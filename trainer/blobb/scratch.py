"""The blob's own brain, grown from scratch: a ~2.8M-parameter transformer that reads the observation text
and writes the plan as tokens (verb, arg, verb, arg, ..., end). It runs in the browser (web/src/scratch.ts),
and it dreams in minutes instead of the LLM's 45.

Words are embedded as word-id + hashed character n-grams (fastText-style), so a word it never saw
("shroom", "mushrooms") still lands near ones it did. Only legal verb/arg pairs can come out. Thought and
say are picked from lines it learned. No dropout: on CPU its random masks were 38% of a step, and the
paraphrased, typo'd, word-dropped curriculum already keeps it from memorizing.

    # hatch: learn the instinct curriculum from nothing (~45 min on 4 P-cores)
    cd web && npm run -s curriculum -- 80000 > ../trainer/data/scratch/curriculum.jsonl
    cd ../trainer && python -m blobb.scratch hatch

    # dream: learn from what it lived through - praise AND failures and 👎 (a few minutes)
    python -m blobb.scratch dream --gen 1 --memories ~/Downloads/blobb-memories-*.json

Brains are saved the way the browser loads them: brain.json (vocab, heads, shapes) + brain.bin (fp16).
The hatchling lives in web/public/brain; dreams go to data/scratch/brains/genN (served by serve.mjs).
"""
import argparse
import fcntl
import json
import math
import random
import re
import shutil
import signal
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DATA, DREAM_MAP_FILE, DREAM_STATUS_FILE, ROOT

SCRATCH = DATA / "scratch"
SCRATCH_CURRICULUM = SCRATCH / "curriculum.jsonl"
SCRATCH_BRAINS = SCRATCH / "brains"
HATCHLING = ROOT.parent / "web" / "public" / "brain"
FIXTURES = ROOT.parent / "web" / "test" / "fixtures" / "scratch.json"

VERBS = ["move_to", "morph", "push", "grab", "drop", "eat", "use", "deliver", "rest"]  # no "skill" (yet)
ARGS = ["none", "berry", "mushroom", "stick", "rock", "tree", "tower", "gate", "gap", "blob", "ball", "puddle", "spring"]
OUT = ["<pad>", "<bos>", "<end>"] + [f"v:{v}" for v in VERBS] + [f"a:{a}" for a in ARGS]
OUT_ID = {t: i for i, t in enumerate(OUT)}
PAD, BOS, END = 0, 1, 2
MAX_STEPS = 4
MAX_PLAN = 2 * MAX_STEPS + 1  # bos + a verb and arg per step; END is predicted after the last
MAX_LEN = 384  # observation tokens; real ones are ~110-200

BUCKETS = 8192
MAX_NGRAMS = 24
TOKEN_RE = re.compile(r"[a-z]+|\d+|[^\sa-z\d,():.]")  # commas, parens, colons, periods carry no meaning here
WIZARD_RE = re.compile(r'The wizard says: "(.*)"')
MODULES = ["qkv", "proj", "ff1", "ff2"]  # per layer, for the dream peek's weight map


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())[:MAX_LEN]


def ngrams(word: str) -> list[int]:
    if not re.fullmatch(r"[a-z]+", word):  # the same test the browser does
        return []
    w = f"<{word}>"
    grams = [w[i:i + n] for n in (3, 4, 5) for i in range(len(w) - n + 1)]
    return [1 + zlib.crc32(g.encode()) % (BUCKETS - 1) for g in grams[:MAX_NGRAMS]]


class Strings:
    """Every distinct token string seen, with its word id and n-gram buckets precomputed, so a batch is
    just tensor gathers."""

    def __init__(self, words: list[str]):
        self.words = words  # index 0 = <pad>, 1 = <unk>
        self.word_id = {w: i for i, w in enumerate(words)}
        self.index: dict[str, int] = {"": 0}
        self.word = [0]
        self.grams = [[0] * MAX_NGRAMS]
        self._tensors = None

    def get(self, tok: str) -> int:
        i = self.index.get(tok)
        if i is None:
            i = self.index[tok] = len(self.word)
            self.word.append(self.word_id.get(tok, 1))
            g = ngrams(tok)
            self.grams.append(g + [0] * (MAX_NGRAMS - len(g)))
            self._tensors = None
        return i

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """String ids of the tokens, and which ones are the wizard's command (for word dropout)."""
        cmd = WIZARD_RE.search(text)
        cmd_words = set(tokenize(cmd.group(1))) if cmd else set()
        toks = tokenize(text)
        return (torch.tensor([self.get(t) for t in toks], dtype=torch.int32),
                torch.tensor([t in cmd_words for t in toks], dtype=torch.bool))

    def tensors(self):
        if self._tensors is None:
            self._tensors = torch.tensor(self.word), torch.tensor(self.grams)
        return self._tensors


def batchify(encoded: list[tuple[torch.Tensor, torch.Tensor]], strings: Strings, drop: float):
    """`drop` hides words of the wizard's command (word id -> unk, n-grams kept), so it learns to read
    commands from spelling too, not only from exact words."""
    sid = nn.utils.rnn.pad_sequence([e for e, _ in encoded], batch_first=True).long()
    word, grams = strings.tensors()
    ids = word[sid]
    if drop:
        cmd = nn.utils.rnn.pad_sequence([c for _, c in encoded], batch_first=True)
        ids = torch.where(cmd & (torch.rand(ids.shape) < drop), torch.ones_like(ids), ids)
    # N-grams depend only on the string: embed each distinct one once, then gather ([U, G] + [B, L] -> U).
    uniq, where = sid.unique(return_inverse=True)
    return ids, (grams[uniq], where), sid != 0


def plan_tokens(plan: list[dict]) -> list[int] | None:
    """None for a plan it can't learn from: unknown steps (skills), or longer than it can write -
    cutting one short would teach it half a plan."""
    if not isinstance(plan, list) or len(plan) > MAX_STEPS:
        return None
    out = [BOS]
    for s in plan:
        if not isinstance(s, dict):
            return None
        v, a = f"v:{s.get('do')}", f"a:{s.get('arg')}"
        if v not in OUT_ID or a not in OUT_ID:
            return None
        out += [OUT_ID[v], OUT_ID[a]]
    return out if len(out) > 1 else None


def plan_steps(tokens: list[int]) -> list[dict]:
    return [{"do": OUT[tokens[i]][2:], "arg": OUT[tokens[i + 1]][2:]} for i in range(1, len(tokens) - 1, 2)]


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        b, s, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, s, d))
        return x + self.ff(self.n2(x))


class Brain(nn.Module):
    def __init__(self, vocab: int, thoughts: int, says: int, d=160, layers=4, heads=4):
        super().__init__()
        self.word = nn.Embedding(vocab, d, padding_idx=0)
        self.gram = nn.Embedding(BUCKETS, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.out_emb = nn.Embedding(len(OUT), d)
        self.out_pos = nn.Embedding(MAX_PLAN, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.plan_head = nn.Linear(d, len(OUT))
        self.thought_head = nn.Linear(d, thoughts)
        self.say_head = nn.Linear(d, says)

    def forward(self, ids, grams, obs_mask, plan):
        """ids [B, L] word ids; grams ([U, G] n-grams of each distinct string, [B, L] index into U);
        obs_mask [B, L] true for real tokens; plan [B, P]. Returns the plan positions' states [B, P, d].
        Prefix-LM: observation tokens see all of each other; plan tokens see it and the plan so far."""
        b, l = ids.shape
        p = plan.shape[1]
        uniq, where = grams
        g = self.gram(uniq).sum(1) / (uniq != 0).sum(1, keepdim=True).clamp(min=1)
        x = torch.cat([self.word(ids) + g[where] + self.pos(torch.arange(l)),
                       self.out_emb(plan) + self.out_pos(torch.arange(p))], 1)
        s = l + p
        allowed = torch.zeros(b, s, s, dtype=torch.bool)
        allowed[:, :, :l] = obs_mask[:, None, :]
        allowed[:, l:, l:] = torch.ones(p, p, dtype=torch.bool).tril()
        allowed[:, :l, l:] = False
        for blk in self.blocks:
            x = blk(x, allowed[:, None])
        return self.norm(x[:, l:])


class Model:
    """The network plus what it needs to answer: its words, its lines, and which verb/arg pairs are legal."""

    def __init__(self, net: Brain, words: list[str], thoughts: list[str], says: list[str], pairs: dict[int, list[int]], cfg: dict):
        self.net, self.thoughts, self.says, self.pairs, self.cfg = net, thoughts, says, pairs, cfg
        self.strings = Strings(words)

    def save(self, out: Path, extra: dict[str, str] | None = None):
        """brain.json + brain.bin (fp16, little-endian, in manifest order) - what the browser loads - plus
        any `extra` files. Written next to `out` and swapped in, so a crash never leaves half a brain."""
        tmp = out.with_name(out.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        tensors, blobs, offset = {}, [], 0
        for name, t in self.net.state_dict().items():
            a = t.detach().float().numpy().astype("<f2")
            tensors[name] = {"offset": offset, "shape": list(a.shape)}
            blobs.append(a.tobytes())
            offset += a.size
        (tmp / "brain.bin").write_bytes(b"".join(blobs))
        manifest = {"version": 1, "cfg": {**self.cfg, "maxLen": MAX_LEN, "maxPlan": MAX_PLAN, "buckets": BUCKETS, "maxNgrams": MAX_NGRAMS},
                    "words": self.strings.words, "out": OUT, "thoughts": self.thoughts, "says": self.says,
                    "pairs": {str(k): v for k, v in self.pairs.items()}, "tensors": tensors}
        (tmp / "brain.json").write_text(json.dumps(manifest, separators=(",", ":")))
        for name, text in (extra or {}).items():
            (tmp / name).write_text(text)
        old = out.with_name(out.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if out.exists():
            out.rename(old)
        tmp.rename(out)
        shutil.rmtree(old, ignore_errors=True)

    @staticmethod
    def open(path: Path) -> "Model":
        m = json.loads((path / "brain.json").read_text())
        if m.get("version") != 1 or m["out"] != OUT:
            raise SystemExit(f"{path} is a brain this trainer doesn't understand")
        halves = np.frombuffer((path / "brain.bin").read_bytes(), dtype="<f2")
        cfg = {k: m["cfg"][k] for k in ("d", "layers", "heads")}
        net = Brain(len(m["words"]), len(m["thoughts"]), len(m["says"]), **cfg)
        state = {name: torch.from_numpy(halves[t["offset"]:t["offset"] + math.prod(t["shape"])].astype(np.float32).reshape(t["shape"]))
                 for name, t in m["tensors"].items()}
        net.load_state_dict(state)
        net.eval()
        return Model(net, m["words"], m["thoughts"], m["says"], {int(k): v for k, v in m["pairs"].items()}, cfg)

    @torch.no_grad()
    def decide(self, user: str, temperature: float = 0.0) -> dict:
        ids, grams, m = batchify([self.strings.encode(user)], self.strings, 0.0)
        plan = [BOS]
        while True:
            h = self.net(ids, grams, m, torch.tensor([plan]))[0, -1]
            logits = self.net.plan_head(h)
            allow = torch.full_like(logits, -math.inf)
            if len(plan) % 2 == 1:  # verb position
                steps = (len(plan) - 1) // 2
                if steps == MAX_STEPS:
                    break
                for v in VERBS:  # only verbs it learned an argument for
                    if self.pairs.get(OUT_ID[f"v:{v}"]):
                        allow[OUT_ID[f"v:{v}"]] = 0
                if steps:
                    allow[END] = 0
            else:
                for a in self.pairs.get(plan[-1], []):
                    allow[a] = 0
            t = choose(logits + allow, temperature)
            if t == END:
                break
            plan.append(t)
        return {"thought": self.thoughts[choose(self.net.thought_head(h), 0.0)], "plan": plan_steps(plan),
                "say": self.says[choose(self.net.say_head(h), temperature)]}


def choose(logits, temperature: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))


# ---------- data ----------

def sample(user: str, assistant: str, weight: float = 1.0, negative: bool = False, failed_step=None, id=None,
           owner: bool = False) -> dict | None:
    """A plan to learn, or with `negative` one to avoid: just the step where the world stopped it
    (`failed_step`), or - the wizard's 👎 - every choice in it."""
    try:
        a = json.loads(assistant)
    except json.JSONDecodeError:
        return None
    toks = plan_tokens(a.get("plan") or [])
    if toks is None or not isinstance(user, str):
        return None
    # Target positions to push down: step k's verb and arg are predicted at 2k and 2k+1. Never the END
    # (at len - 1): it ends every plan, and lowering it is the cheapest way to make any one plan
    # unlikely - it would just learn to ramble on.
    steps = (len(toks) - 1) // 2
    avoid = (2 * failed_step, 2 * failed_step + 2) if isinstance(failed_step, int) and 0 <= failed_step < steps else (0, len(toks) - 1)
    return {"user": user, "plan": toks, "thought": a.get("thought", ""), "say": a.get("say", ""), "weight": weight,
            "negative": negative, "avoid": avoid, "id": id if isinstance(id, str) else None, "owner": owner}


def load_curriculum(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no curriculum at {path} - make one: cd web && npm run -s curriculum -- 80000 > {path}")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    random.shuffle(lines)
    out = []
    for line in lines[:limit]:
        s = json.loads(line)
        x = sample(s["user"], s["assistant"])
        if x:
            out.append(x)
    return out


def load_memories(paths: list[Path]) -> list[dict]:
    """What the blob lived through. Good attempts are copied; the wizard's praise counts up to 3x. Bad
    ones - plans that failed, or that he 👎'd - are pushed away from, which the LLM's dream can't do."""
    out = []
    for path in paths:
        try:
            export = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as ex:
            raise SystemExit(f"can't read memories from {path}: {ex}")
        n = 0
        for s in (export.get("samples") or []) if isinstance(export, dict) else []:
            if not isinstance(s, dict):
                continue
            r = s.get("reward", 1)
            if not isinstance(r, (int, float)) or r == 0:
                continue
            if r > 0:
                x = sample(s.get("user"), s.get("assistant", ""), min(3, math.ceil(r)) if s.get("owner") else 1, id=s.get("id"),
                           owner=bool(s.get("owner")))
            else:
                x = sample(s.get("user"), s.get("assistant", ""), 1.0 if s.get("owner") else 0.5, negative=True,
                           failed_step=None if s.get("owner") else s.get("failedStep"), id=s.get("id"), owner=bool(s.get("owner")))
            if x:
                out.append(x)
                n += 1
        print(f"{n} memories from {path}", flush=True)
    return out


# ---------- training ----------

def fit(model: Model, items: list[dict], epochs: int, lr: float, batch: int, word_drop: float,
        on_step=lambda step, total, loss, items: None):
    """Positives: learn the plan (and its thought and say). Negatives: push down each choice it shouldn't
    have made, -log(1 - p) - it eases off once they're unlikely, so it can't wreck the rest."""
    net = model.net
    t_id = {t: i for i, t in enumerate(model.thoughts)}
    s_id = {s: i for i, s in enumerate(model.says)}
    for it in items:
        it["enc"] = model.strings.encode(it["user"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    total = math.ceil(len(items) / batch) * epochs
    warm = max(1, total // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    step = 0
    for _ in range(epochs):
        net.train()
        random.shuffle(items)
        for i in range(0, len(items), batch):
            part = items[i:i + batch]
            ids, grams, m = batchify([it["enc"] for it in part], model.strings, word_drop)
            p = max(len(it["plan"]) for it in part)
            plan = torch.full((len(part), p), PAD, dtype=torch.long)
            tgt = torch.full((len(part), p), -100, dtype=torch.long)
            last = torch.zeros(len(part), dtype=torch.long)
            for r, it in enumerate(part):
                x = it["plan"]
                plan[r, :len(x)] = torch.tensor(x)
                tgt[r, :len(x)] = torch.tensor(x[1:] + [END])
                if it["negative"]:
                    lo, hi = it["avoid"]
                    tgt[r, :lo] = -100
                    tgt[r, hi:] = -100
                last[r] = len(x) - 1
            h = net(ids, grams, m, plan)
            tok_nll = F.cross_entropy(net.plan_head(h).transpose(1, 2), tgt, reduction="none")  # [B, P], 0 where ignored
            scored = tgt != -100
            n_tok = scored.sum(1).clamp(min=1)
            neg = torch.tensor([it["negative"] for it in part])
            w = torch.tensor([it["weight"] for it in part], dtype=torch.float)
            pos_loss = tok_nll.sum(1) / n_tok
            # Each bad choice pushed down given the ones before it: -log(1 - p). Per token, not per plan -
            # a whole plan can be made unlikely by bending one cheap late step ("morph spring, morph spring,
            # eat berry"), which dodges the loss and keeps the habit. Clamped: big but finite for a sure thing.
            unlikely = -torch.log1p(-torch.exp(-tok_nll).clamp(max=1 - 1e-4))
            neg_loss = (unlikely * scored).sum(1) / n_tok
            per = torch.where(neg, neg_loss, pos_loss)
            hl = h[torch.arange(len(part)), last]
            ti = torch.tensor([-100 if it["negative"] else t_id.get(it["thought"], -100) for it in part])
            si = torch.tensor([-100 if it["negative"] else s_id.get(it["say"], -100) for it in part])
            if (ti != -100).any():
                per = per + 0.3 * F.cross_entropy(net.thought_head(hl), ti, reduction="none", ignore_index=-100)
            if (si != -100).any():
                per = per + 0.3 * F.cross_entropy(net.say_head(hl), si, reduction="none", ignore_index=-100)
            loss = (per * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            on_step(step, total, loss.item(), part)
    net.eval()


def exact(model: Model, items: list[dict]) -> int:
    return sum(model.decide(it["user"])["plan"] == plan_steps(it["plan"]) for it in items)


def hatch(args):
    torch.manual_seed(0)
    random.seed(0)
    data = load_curriculum(args.curriculum)
    val, tr = data[:args.val], data[args.val:]
    if args.base:
        # Keep growing an existing brain on revised lessons (say, after a fix to the island), instead of
        # starting from nothing. Its words, lines and legal steps stay as they were.
        model = Model.open(args.base)
        print(f"{len(tr)} lessons ({len(val)} held out), on top of {args.base}; "
              f"exact plan before: {exact(model, val[:1000])}/{min(1000, len(val))}", flush=True)
    else:
        model = new_brain(tr, args)
        print(f"{len(tr)} lessons ({len(val)} held out), {len(model.strings.words)} words, {len(model.thoughts)} thoughts, "
              f"{len(model.says)} says, {sum(p.numel() for p in model.net.parameters()) / 1e6:.2f}M weights", flush=True)
    t0 = time.time()

    def log(step, total, loss, _):
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {loss:.4f} ({el / step:.2f}s/step, {el / step * (total - step) / 60:.0f} min left)", flush=True)

    fit(model, tr, args.epochs, args.lr, args.batch, args.word_drop, log)
    print(f"exact plan on held-out lessons: {exact(model, val[:1000])}/{min(1000, len(val))}", flush=True)
    model.save(args.out)
    if args.out.resolve() == HATCHLING.resolve():  # the browser's tests check the brain it ships with
        write_fixtures(Model.open(args.out), val)
    print(f"hatched in {(time.time() - t0) / 60:.0f} min -> {args.out}", flush=True)


def new_brain(tr: list[dict], args) -> Model:
    counts: dict[str, int] = {}
    for it in tr:
        for w in tokenize(it["user"]):
            counts[w] = counts.get(w, 0) + 1
    words = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)
    thoughts, says = sorted({it["thought"] for it in tr}), sorted({it["say"] for it in tr})
    pairs: dict[int, set[int]] = {}
    for it in tr:
        for i in range(1, len(it["plan"]), 2):
            pairs.setdefault(it["plan"][i], set()).add(it["plan"][i + 1])
    cfg = {"d": args.d, "layers": args.layers, "heads": args.heads}
    return Model(Brain(len(words), len(thoughts), len(says), **cfg), words, thoughts, says, {k: sorted(v) for k, v in pairs.items()}, cfg)


def write_fixtures(model: Model, items: list[dict], n: int = 20):
    """Decisions the browser must reproduce exactly (test/scratch.test.ts), from the saved fp16 brain."""
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps([{"user": it["user"], "decision": model.decide(it["user"])} for it in items[:n]], indent=1))


# ---------- dreaming ----------

MEMORIES = "memories.jsonl"  # next to a dreamed brain: everything it has ever dreamed about
CHECK = 300                  # held-out instinct lessons, to measure what a dream made it forget
FORGOT = 0.05                # more lost than this gets a warning
MAX_LIFE = 5000              # oldest memories fade past this many


def life(brain: Path) -> list[dict]:
    """The memories a brain was dreamed from. Every dream re-learns all of them, not just the new night's:
    otherwise replaying the instinct lessons quietly un-learns what earlier dreams taught - a blob the
    wizard broke of a habit went straight back to it one dream later."""
    f = brain / MEMORIES
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(m, dict) and isinstance(m.get("user"), str) and isinstance(m.get("plan"), list) and isinstance(m.get("avoid"), list):
            out.append({**m, "avoid": tuple(m["avoid"])})
    return out


def merge_life(old: list[dict], new: list[dict]) -> list[dict]:
    """Old memories then new, the same memory (by id) only once - a re-export or a re-run dream - and only
    the most recent MAX_LIFE."""
    seen, out = set(), []
    for m in reversed(old + new):
        if m.get("id"):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
        out.append(m)
    return list(reversed(out))[-MAX_LIFE:]

def status(**kw):
    kw["t"] = time.time()
    tmp = DREAM_STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(kw))
    tmp.replace(DREAM_STATUS_FILE)


def weight_map(net: Brain, base: dict[str, torch.Tensor]) -> tuple[list[list[float]], list[int]]:
    """||W - W_before|| per layer x module, and each module's weight count, for the dream peek."""
    names = {"qkv": "qkv.weight", "proj": "proj.weight", "ff1": "ff.0.weight", "ff2": "ff.2.weight"}
    state = net.state_dict()
    norms, numel = [], []
    for i in range(len(net.blocks)):
        row = []
        for mod in MODULES:
            k = f"blocks.{i}.{names[mod]}"
            row.append(float(f"{float((state[k] - base[k]).norm()):.4g}"))
            if i == 0:
                numel.append(state[k].numel())
        norms.append(row)
    return norms, numel


def to_taste(replay: list[dict], memories: list[dict]) -> list[dict]:
    """The wizard's taste outranks the textbook: once he's 👎'd a plan (and not 👍'd it too), stop replaying
    lessons that teach exactly that plan - or they drag it back every night, and the dream settles on a
    mangled half of it ("morph spring, use berry")."""
    disliked = {tuple(m["plan"]) for m in memories if m.get("owner") and m["negative"]}
    disliked -= {tuple(m["plan"]) for m in memories if m.get("owner") and not m["negative"]}
    return [x for x in replay if tuple(x["plan"]) not in disliked]


def dream(args):
    lock = open(DATA / "dream.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another dream is already running (data/dream.lock)")

    def on_term(*_):
        raise KeyboardInterrupt  # a plain `kill` should still leave an honest status behind
    signal.signal(signal.SIGTERM, on_term)
    gen, tag = args.gen, f"gen{args.gen}"
    out = SCRATCH_BRAINS / tag
    if (out / "brain.json").exists() and not args.force:
        raise SystemExit(f"{tag} already exists. --force to dream it again, or --gen {gen + 1} to build on it.")
    base = args.base or (HATCHLING if gen == 1 else SCRATCH_BRAINS / f"gen{gen - 1}")
    if not (base / "brain.json").exists():
        raise SystemExit(f"nothing to build {tag} on: {base} has no brain")
    DREAM_MAP_FILE.unlink(missing_ok=True)
    try:
        status(phase="falling asleep", step=0, total=0, gen=gen)
        torch.manual_seed(gen)
        random.seed(gen)
        model = Model.open(base)
        # An export from the game, or an old brain's directory: its whole life, e.g. onto a new release.
        new = [m for p in args.memories if p.is_dir() for m in life(p)] + load_memories([p for p in args.memories if not p.is_dir()])
        if not new:
            raise SystemExit("nothing to dream about: no usable memories")
        memories = merge_life(life(base), new)
        # Replay some instinct lessons so it doesn't forget everything else while it learns these - and
        # keep a few it never replays, to check afterwards how much it forgot anyway.
        lessons = load_curriculum(args.curriculum, args.replay + CHECK)
        check, replay = lessons[:CHECK], lessons[CHECK:]
        kept = to_taste(replay, memories)
        if len(kept) < len(replay):
            print(f"not replaying {len(replay) - len(kept)} lessons that teach plans the wizard 👎'd", flush=True)
        replay = kept
        instinct_before = exact(model, check)
        reps = max(1, min(10, round(len(replay) / 4 / len(memories))))
        items = [{**m, "memory": True} for m in memories for _ in range(reps)] + replay
        n_neg = sum(m["negative"] for m in memories)
        print(f"dreaming {len(memories)} memories ({len(new)} new, {n_neg} to avoid) x{reps} among {len(replay)} lessons, on top of {base}", flush=True)
        before = {k: v.clone() for k, v in model.net.state_dict().items()}
        losses: list[float] = []
        t0, ema = time.time(), None

        def about(it) -> str:
            said = WIZARD_RE.search(it["user"])
            return said.group(1)[:80] if said else "(looking after itself)"

        def on_step(step, total, loss, part):
            nonlocal ema
            ema = loss if ema is None else 0.9 * ema + 0.1 * loss
            losses.append(round(ema, 4))
            mem = [it for it in part if it.get("memory")] or part
            el = time.time() - t0
            status(phase="dreaming", step=step, total=total, loss=round(ema, 4), eta=round(el / step * (total - step)),
                   about=about(random.choice(mem)), gen=gen)
            if step % 5 == 0 or step == total:
                norms, numel = weight_map(model.net, before)
                tmp = DREAM_MAP_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps({"t": time.time(), "step": step, "total": total, "layers": len(model.net.blocks),
                                           "modules": MODULES, "numel": numel, "norms": norms, "losses": losses}))
                tmp.replace(DREAM_MAP_FILE)

        fit(model, items, args.epochs, args.lr, args.batch, 0.1, on_step)
        status(phase="waking", step=1, total=1, gen=gen)
        model.save(out, {MEMORIES: "".join(json.dumps({k: v for k, v in m.items() if k not in ("enc", "memory")}) + "\n" for m in memories)})
        print(f"remembers {exact(model, [m for m in memories if not m['negative']])}/{len(memories) - n_neg} good memories; "
              f"repeats {sum(model.decide(m['user'])['plan'] == plan_steps(m['plan']) for m in memories if m['negative'])}/{n_neg} bad ones", flush=True)
        instinct = exact(model, check)
        drop = (instinct_before - instinct) / max(1, len(check))
        print(f"instinct: {instinct_before}/{len(check)} lessons right before the dream, {instinct} after", flush=True)
        if drop > FORGOT:
            print(f"WARNING: it forgot {drop:.0%} of its instincts - maybe too much to avoid and too little to do. "
                  f"The dream is saved as {tag}; the old brain is still in {base}.", flush=True)
        status(phase="done", model=f"own:{tag}", gen=gen, samples=len(memories),
               instinct=[instinct_before, instinct, len(check)])
        print(f"dreamed {tag} in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
    except SystemExit as ex:
        status(phase="error", error=str(ex.code)[:500], gen=gen)
        raise
    except KeyboardInterrupt:
        status(phase="error", error="interrupted", gen=gen)
        raise
    except Exception as ex:
        status(phase="error", error=f"{type(ex).__name__}: {ex}"[:500], gen=gen)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (physical performance cores are best)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hatch", help="grow a brain from nothing on the instinct curriculum")
    h.add_argument("--curriculum", type=Path, default=SCRATCH_CURRICULUM)
    h.add_argument("--out", type=Path, default=HATCHLING)
    h.add_argument("--from", dest="base", type=Path, default=None, help="keep training this brain instead of starting fresh")
    h.add_argument("--epochs", type=int, default=4)
    h.add_argument("--batch", type=int, default=64)
    h.add_argument("--lr", type=float, default=1e-3)
    h.add_argument("--d", type=int, default=160)
    h.add_argument("--layers", type=int, default=4)
    h.add_argument("--heads", type=int, default=4)
    h.add_argument("--val", type=int, default=2000)
    h.add_argument("--word-drop", type=float, default=0.15)
    d = sub.add_parser("dream", help="learn from memories exported by the game")
    d.add_argument("--gen", type=int, required=True, help="generation to create (1 builds on the hatchling)")
    d.add_argument("--memories", nargs="+", type=Path, required=True,
                   help="exports from the game (⚙ → Export memories), and/or old brains' directories (their whole life)")
    d.add_argument("--from", dest="base", type=Path, default=None,
                   help="brain to build on (default: gen N-1, or the hatchling for gen 1) - e.g. a new release's "
                        "hatchling, to carry a blob's memories over to it")
    d.add_argument("--curriculum", type=Path, default=SCRATCH_CURRICULUM)
    d.add_argument("--replay", type=int, default=4000, help="instinct lessons mixed in so it doesn't forget them")
    d.add_argument("--epochs", type=int, default=3)
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--lr", type=float, default=3e-4)
    d.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    SCRATCH_BRAINS.mkdir(parents=True, exist_ok=True)
    {"hatch": hatch, "dream": dream}[args.cmd](args)


if __name__ == "__main__":
    main()
