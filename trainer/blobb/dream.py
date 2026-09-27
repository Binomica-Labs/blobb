"""Dreaming: train what the blob experienced into its weights.

Inputs (all as chat samples {system, user, assistant}):
  - the instinct curriculum   (web: npm run -s curriculum -> data/curriculum.jsonl)
  - memories exported from the game ("Export memories for dreaming" in settings)

LoRA fine-tune on top of the previous generation, merge, and register the result with Ollama
as blobb-genN.

    python -m blobb.dream --gen 1                                  # hatch: curriculum only
    python -m blobb.dream --gen 2 --memories ~/Downloads/blobb-memories-*.json
"""
import argparse
import fcntl
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .config import BRAINS, CURRICULUM, DATA, DREAM_MAP_FILE, DREAM_STATUS_FILE, HF_BASE, KEEP_GENS, OLLAMA_BASE

# Samples longer than this are skipped: memory grows with length, and real ones are ~450 tokens.
MAX_TOKENS = 1024
# Replies are ~40-70 tokens of JSON; anything far longer is a runaway, and scored tokens are costly.
MAX_REPLY_TOKENS = 256
BATCH = 4

# Memory model for one training step (fp32 on CPU), measured on an i7-1260P with ~1000 packed tokens:
# no checkpointing peaks at 6.8 GB, every layer checkpointed at 3.8 GB.
BASE_GB = 2.6            # weights, runtime, LoRA optimizer state
TOKEN_GB = 4.3e-3        # activations per packed token, nothing checkpointed
TOKEN_GB_CKPT = 1.2e-3   # ... every layer checkpointed
SCORED_GB = 1.8e-3       # per scored reply token: fp32 logits over the 152k vocab, log-softmax, grad
SAFETY = 1.15
DEFAULT_MAX_RAM_GB = 6.0
# Below this, even a fully checkpointed step doesn't fit comfortably.
MIN_FREE_GB = BASE_GB + 1.9


def write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


GEN = 0  # generation being dreamed, stamped on every status for the peek


def status(**kw):
    kw["t"] = time.time()
    kw.setdefault("gen", GEN)
    write_json(DREAM_STATUS_FILE, kw)
    if kw.get("phase") != "dreaming" or kw.get("step", 0) % 10 == 0:
        print(json.dumps(kw), flush=True)


def valid(s) -> bool:
    return isinstance(s, dict) and all(isinstance(s.get(k), str) and s[k] for k in ("system", "user", "assistant"))


def _reward(s: dict) -> float:
    r = s.get("reward", 1)  # older exports only had good memories
    return r if isinstance(r, (int, float)) else 0


def load_samples(curriculum: Path | None, memories: list[Path], max_curriculum: int) -> list[dict]:
    samples: list[dict] = []
    if curriculum and curriculum.exists():
        cur = []
        for n, line in enumerate(curriculum.read_text().splitlines(), 1):
            try:
                s = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                s = None
            if valid(s):
                cur.append(s)
            elif line.strip():
                print(f"skipping bad curriculum line {n}", flush=True)
        random.shuffle(cur)
        samples += cur[:max_curriculum]
    for path in memories:
        try:
            export = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as ex:
            raise SystemExit(f"can't read memories from {path}: {ex}")
        # Only what went well: the game also exports failures and 👎s (for its own brain to avoid),
        # and fine-tuning on those would teach the LLM to repeat them.
        good = [s for s in (export.get("samples") or []) if valid(s) and _reward(s) > 0] if isinstance(export, dict) else []
        if not good:
            print(f"no usable samples in {path}", flush=True)
        for s in good:
            # Praise from the owner counts more than "the plan didn't crash".
            reps = min(3, math.ceil(s.get("reward", 1))) if s.get("owner") else 1
            samples += [s] * reps
    random.shuffle(samples)
    return samples


MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAP_EVERY = 3  # steps between weight-map snapshots


def _cpu_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def performance_cores() -> int:
    """Physical performance cores - the default thread count. A torch step waits for its slowest
    thread, so on hybrid Intel chips the efficiency cores drag everything to their pace: on an
    i7-1260P, its 4 P-cores train as fast as all 16 threads (and leave the rest free to play)."""
    try:
        cpus = _cpu_list(Path("/sys/devices/cpu_core/cpus").read_text())  # hybrid Intel: P-cores only
    except (OSError, ValueError):
        cpus = list(range(os.cpu_count() or 4))
    cores = set()
    for c in cpus:
        topo = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        try:
            cores.add((topo / "physical_package_id").read_text().strip() + ":" + (topo / "core_id").read_text().strip())
        except OSError:
            cores.add(str(c))  # no topology (not Linux): count logical CPUs
    return max(1, len(cores))


def step_gb(tokens: int, scored: int, ckpt_layers: int, layers: int) -> float:
    """Estimated peak RAM of one forward+backward over `tokens` packed tokens, `scored` of them targets."""
    per_token = TOKEN_GB - (TOKEN_GB - TOKEN_GB_CKPT) * ckpt_layers / layers
    return BASE_GB + (tokens * per_token + scored * SCORED_GB) * SAFETY


def layers_to_checkpoint(tokens: int, layers: int, budget_gb: float, scored: int = 0) -> int:
    """How many layers must recompute activations in the backward pass for a step of `tokens`
    packed tokens to fit in `budget_gb`. Checkpointing roughly halves speed, so use as little as fits.
    (If even all of them isn't enough, the batch has to be split - see micro_batches.)"""
    need = step_gb(tokens, scored, 0, layers)
    if need <= budget_gb:
        return 0
    saved_per_layer = tokens * (TOKEN_GB - TOKEN_GB_CKPT) * SAFETY / layers
    return min(layers, math.ceil((need - budget_gb) / saved_per_layer))


def scored(labels: list[int]) -> int:
    return sum(1 for y in labels if y != -100)


def micro_batches(batch: list, layers: int, budget_gb: float) -> list[list]:
    """Split a batch into consecutive packs that each fit the budget with every layer checkpointed.
    Usually that's the whole batch; it splits when samples share little prompt (e.g. memories from a
    renamed blob next to curriculum lessons) or run long. Gradients are accumulated across the parts."""
    out: list[list] = []
    cur: list = []
    for item in batch:
        trial = cur + [item]
        ids, labels, _, _ = pack(trial)
        if cur and step_gb(len(ids), scored(labels), layers, layers) > budget_gb:
            out.append(cur)
            trial = [item]
        cur = trial
    return out + [cur]


def pack(batch: list[tuple[list[int], list[int], int]]) -> tuple[list[int], list[int], list[int], list[int]]:
    """Pack a batch into ONE sequence that shares the prompt prefix every sample has in common (the
    ~290-token system prompt), followed by each sample's own tail. Positions restart after the prefix
    for every tail, and a block mask (see `segments`) keeps tails from seeing each other - so each
    sample sees exactly what it would alone, for about half the tokens.
    Items are (ids, labels, prompt_len). Returns (ids, labels, positions, segment) where segment is
    -1 for the shared prefix and i for sample i's tail."""
    # Every tail keeps at least one prompt token, so no target is ever predicted across a seam.
    lcp = min(plen for _, _, plen in batch) - 1
    first = batch[0][0]
    for ids, _, _ in batch[1:]:
        n = 0
        while n < lcp and ids[n] == first[n]:
            n += 1
        lcp = n
    ids, labels, pos, seg = list(first[:lcp]), [-100] * lcp, list(range(lcp)), [-1] * lcp
    for i, (x, y, _) in enumerate(batch):
        ids += x[lcp:]
        labels += y[lcp:]
        pos += range(lcp, len(x))
        seg += [i] * (len(x) - lcp)
    return ids, labels, pos, seg


def about(sample: dict) -> str:
    """What a sample is 'about', for the dream peek: the wizard's words, or what the blob did alone."""
    m = re.search(r'The wizard says: "(.*)"', sample["user"])
    return m.group(1)[:80] if m else "(looking after itself)"


def weight_map(model, layers: int) -> tuple[list[list[float]], list[int]]:
    """How far each LoRA-adapted matrix has moved from the base: ||scale * B @ A||_F, per layer x module,
    plus each module's weight count (so the peek can show change per weight - big matrices aren't
    more "dreamed" just for being big). Uses r x r products (trace((BᵀB)(AAᵀ))), never the full delta."""
    import torch
    out = [[0.0] * len(MODULES) for _ in range(layers)]
    numel = [0] * len(MODULES)
    with torch.no_grad():
        for name, mod in model.named_modules():
            m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)$", name)
            if not m or not hasattr(mod, "lora_A") or "default" not in mod.lora_A:
                continue
            a, b = mod.lora_A["default"].weight, mod.lora_B["default"].weight
            sq = ((b.T @ b) * (a @ a.T)).sum().clamp(min=0)
            i = MODULES.index(m.group(2))
            out[int(m.group(1))][i] = float(sq.sqrt()) * mod.scaling["default"]
            numel[i] = b.shape[0] * a.shape[1]
    return out, numel


def train(base: str, samples: list[dict], out_dir: Path, epochs: int, lr: float,
          threads: int | None = None, max_ram_gb: float = DEFAULT_MAX_RAM_GB):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    threads = threads or performance_cores()
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.config.use_cache = False
    # Enabled here, then switched per layer each step - only as many layers as the RAM budget needs.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # No LoRA dropout: on CPU the random masks and input copies cost more than half of each step.
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=MODULES))

    layers = model.get_base_model().config.num_hidden_layers
    encoded, skipped = [], 0
    for s in samples:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["user"]}],
            tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        t = tok(s["assistant"] + "<|im_end|>", add_special_tokens=False).input_ids
        # Too long to be real, or too big to train even alone within the RAM budget.
        if (len(p) + len(t) > MAX_TOKENS or len(t) > MAX_REPLY_TOKENS
                or step_gb(len(p) + len(t), len(t), layers, layers) > max_ram_gb):
            skipped += 1
            continue
        encoded.append(((p + t, [-100] * len(p) + t, len(p)), about(s)))
    if skipped:
        print(f"skipping {skipped} samples that are too long", flush=True)
    if not encoded:
        raise ValueError("every sample was too long to dream about")

    total = math.ceil(len(encoded) / BATCH) * epochs
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 1 - s / total))
    model.train()
    losses: list[float] = []

    def snapshot(step: int, total: int):
        norms, numel = weight_map(model, layers)
        write_json(DREAM_MAP_FILE, {"t": time.time(), "step": step, "total": total, "layers": layers,
                                    "modules": [m.removesuffix("_proj") for m in MODULES], "numel": numel,
                                    "norms": [[float(f"{v:.4g}") for v in row] for row in norms], "losses": losses})

    snapshot(0, 0)
    inner = model.get_base_model()  # Qwen2ForCausalLM with the LoRA layers injected
    decoder = inner.model.layers
    print(f"dreaming on {threads} threads, up to {max_ram_gb:.1f} GB", flush=True)
    step, t0, loss_ema = 0, time.time(), None
    for _ in range(epochs):
        # Fresh random batches every epoch. (No need to group by length: packed batches have no padding.)
        random.shuffle(encoded)
        for i in range(0, len(encoded), BATCH):
            batch = encoded[i:i + BATCH]
            # Status goes out before the (slow) step, so the peek shows what it's dreaming about right now.
            eta = (time.time() - t0) / step * (total - step) if step else None
            status(phase="dreaming", step=step, total=total, loss=round(loss_ema, 4) if loss_ema is not None else None,
                   eta=round(eta) if eta is not None else None, about=random.choice(batch)[1])
            items = [e for e, _ in batch]
            n_scored = sum(scored(y) for _, y, _ in items)
            loss_value = 0.0
            for part in micro_batches(items, len(decoder), max_ram_gb):
                ids, labels, pos, seg = pack(part)
                ckpt = layers_to_checkpoint(len(ids), len(decoder), max_ram_gb, scored(labels))
                for n, layer in enumerate(decoder):
                    layer.gradient_checkpointing = n < ckpt
                segs = torch.tensor(seg)
                # Causal, and a tail only sees the shared prefix and itself. Bool mask: sdpa's fast path.
                mask = (torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
                        & ((segs[:, None] == segs[None, :]) | (segs[None, :] == -1)))[None, None]
                hidden = inner.model(input_ids=torch.tensor([ids]), attention_mask=mask,
                                     position_ids=torch.tensor([pos])).last_hidden_state
                # Only the ~40 reply tokens per sample are scored, so only those get projected onto the
                # 152k vocab - full-sequence fp32 logits (+ grads) once OOM'd the box.
                targets = torch.tensor([labels])[:, 1:]
                keep = targets != -100
                logits = inner.lm_head(hidden[:, :-1][keep])
                # Summed and divided by the whole batch's target count: the mean over the batch, however it's split.
                loss = torch.nn.functional.cross_entropy(logits.float(), targets[keep], reduction="sum") / n_scored
                del hidden, logits, mask
                loss.backward()
                loss_value += loss.item()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            loss_ema = loss_value if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value
            losses.append(round(loss_ema, 4))
            if step % MAP_EVERY == 0 or step == total:
                snapshot(step, total)

    snapshot(total, total)
    status(phase="waking", step=total, total=total, loss=round(loss_ema or 0, 4), eta=0)
    model = model.merge_and_unload()
    # Write next to the destination and swap in at the end, so a crash mid-save never leaves a
    # half-written brain that the next generation would train on top of.
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    model.to(torch.bfloat16).save_pretrained(tmp_dir, safe_serialization=True)
    tok.save_pretrained(tmp_dir)
    # transformers 5 nests rope_theta under rope_parameters; Ollama's importer only reads the
    # top-level key and silently falls back to 10000 (-> gibberish), so write it back out.
    cfg_path = tmp_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    for k, v in (cfg.get("rope_parameters") or {}).items():
        cfg.setdefault(k, v)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    # Swap by renames only: set the old one aside, move the new one in, then delete the old.
    old = out_dir.with_name(out_dir.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out_dir.exists():
        out_dir.rename(old)
    tmp_dir.rename(out_dir)
    shutil.rmtree(old, ignore_errors=True)


def free_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("inf")  # not Linux: can't tell, don't block


def ollama_template() -> str:
    """Also a preflight: better to find out Ollama is down now than after hours of dreaming."""
    try:
        r = subprocess.run(["ollama", "show", "--template", OLLAMA_BASE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise SystemExit(f"can't run ollama ({ex}). Is it installed and running?")
    if r.returncode != 0:
        raise SystemExit(f"ollama can't show {OLLAMA_BASE}: {r.stderr.strip()} (try: ollama pull {OLLAMA_BASE})")
    return r.stdout


def register_with_ollama(out_dir: Path, tag: str, template: str):
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f'FROM {out_dir}\nTEMPLATE """{template}"""\n'
                'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n')
        modelfile = f.name
    retry = f"weights are safe in {out_dir}; retry with --gen {tag.removeprefix('blobb-gen')} --register-only"
    try:
        r = subprocess.run(["ollama", "create", tag, "-q", "q8_0", "-f", modelfile], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"ollama create failed: {r.stderr.strip()[-300:]} ({retry})")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ollama create hung for 30 min ({retry})")
    finally:
        os.unlink(modelfile)


def prune(current_gen: int):
    for d in BRAINS.glob("gen*"):
        if d.name.endswith((".partial", ".old")):
            shutil.rmtree(d, ignore_errors=True)  # leftovers from an interrupted save (~1 GB each)
            continue
        if not d.name[3:].isdigit():
            continue
        g = int(d.name[3:])
        if g <= current_gen - KEEP_GENS:
            shutil.rmtree(d, ignore_errors=True)
            try:
                subprocess.run(["ollama", "rm", f"blobb-gen{g}"], capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                print(f"ollama rm blobb-gen{g} timed out; remove it by hand if it's still listed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=int, required=True, help="generation number to create")
    ap.add_argument("--memories", nargs="*", default=[], type=Path, help="memory exports from the game")
    ap.add_argument("--curriculum", type=Path, default=CURRICULUM)
    ap.add_argument("--max-curriculum", type=int, default=600,
                    help="curriculum samples to mix in (keeps skills from being forgotten)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--threads", type=int, default=None,
                    help=f"CPU threads (default: physical performance cores, here {performance_cores()})")
    ap.add_argument("--max-ram", type=float, default=DEFAULT_MAX_RAM_GB,
                    help="GB a training step may use; more = faster (less recomputation)")
    ap.add_argument("--force", action="store_true", help="dream this generation again even though it exists")
    ap.add_argument("--register-only", action="store_true",
                    help="don't train: hand the existing genN to Ollama (e.g. after `ollama create` failed)")
    args = ap.parse_args()
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be at least 1")
    if args.max_ram < MIN_FREE_GB:
        ap.error(f"--max-ram must be at least {MIN_FREE_GB:.1f} GB")
    if args.gen < 1:
        ap.error("--gen starts at 1")
    global GEN
    GEN = args.gen

    def on_term(*_):
        raise KeyboardInterrupt  # a plain `kill` should still leave an honest status behind
    signal.signal(signal.SIGTERM, on_term)

    # Two dreams at once is how the machine ran out of memory - only ever one.
    lock = open(DATA / "dream.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another dream is already running (data/dream.lock)")

    out_dir = BRAINS / f"gen{args.gen}"
    tag = f"blobb-gen{args.gen}"
    if args.register_only:
        if not (out_dir / "config.json").exists():
            raise SystemExit(f"gen{args.gen} isn't on disk - nothing to register")
        register_with_ollama(out_dir, tag, ollama_template())
        prune(args.gen)
        status(phase="done", model=tag, gen=args.gen)
        print(f"registered {tag}", flush=True)
        return
    if (out_dir / "config.json").exists() and not args.force:
        raise SystemExit(f"gen{args.gen} already exists. --force to dream it again (anything built on it "
                         f"keeps the old one), or --gen {args.gen + 1} to build on it.")
    # A fresh peek: don't show the last dream's map under this one's title while the model loads.
    DREAM_MAP_FILE.unlink(missing_ok=True)

    try:
        status(phase="falling asleep", step=0, total=0)
        free = free_gb()
        if free < MIN_FREE_GB:
            raise SystemExit(f"only {free:.1f} GB of RAM free; a dream needs at least {MIN_FREE_GB:.1f}. "
                             "Close something big first.")
        # Never plan to use more than what's actually free, less a margin for everything else.
        budget = max(MIN_FREE_GB, min(args.max_ram, free - 1.0))
        template = ollama_template()
        prev = BRAINS / f"gen{args.gen - 1}"
        if args.gen > 1 and not (prev / "config.json").exists():
            # Silently starting over from the base model would throw away everything learned.
            raise SystemExit(f"gen{args.gen - 1} isn't on disk, so gen{args.gen} has nothing to build on. "
                             f"Existing: {sorted(d.name for d in BRAINS.glob('gen*')) or 'none'}")
        samples = load_samples(args.curriculum, args.memories, args.max_curriculum)
        if not samples:
            raise SystemExit("nothing to dream about (no curriculum or memories)")
        base = str(prev) if args.gen > 1 else HF_BASE
        print(f"dreaming {len(samples)} samples on top of {base}", flush=True)
        train(base, samples, out_dir, args.epochs, args.lr, args.threads, budget)
        register_with_ollama(out_dir, tag, template)
        prune(args.gen)
        status(phase="done", model=tag, gen=args.gen, samples=len(samples))
    except SystemExit as ex:
        status(phase="error", error=str(ex.code)[:500])
        raise
    except KeyboardInterrupt:
        status(phase="error", error="interrupted")
        raise
    except Exception as ex:  # surface failures instead of dying silently
        status(phase="error", error=f"{type(ex).__name__}: {ex}"[:500])
        raise


if __name__ == "__main__":
    main()
