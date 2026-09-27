# Blobb

You are a very old wizard, and you are done with people. So you opened your spellbook and pulled a
squishy blob out of another dimension to run the villagers' errands for you.

The blob's mind is a tiny language model. You tell it what to do in plain words; it thinks up a short
plan, tries it in the world, fails, hears why, and tries again. It learns in three ways:

- **Memories** - every attempt, what went wrong, and what you thought of it (👍/👎). Similar situations
  bring the right memories back into its head.
- **Skills** - name a plan that worked (⭐) and it can do the whole thing as one step.
- **Dreams** (🌙) - when it sleeps, what it lived through is trained into its brain for real: the next
  generation knows it without being reminded.

Villagers write in with requests - a mushroom for soup, a berry from the high meadow, a bridge over the
gap. Each one needs the blob to figure out a different use of its body: a spring to jump up, a puddle to
slide under the low gate, a hard ball to knock fruit out of a tree.

## Play

**In your browser:** https://binomica-labs.github.io/blobb/ - the brain runs on your GPU via WebGPU (Chrome/Edge); the
first summoning downloads about 350 MB, cached after that.

**Locally, with Ollama** (faster, and it can dream):

```sh
ollama pull qwen2.5:0.5b
cd web && npm ci && npm run build && npm run serve    # http://localhost:8000 (add `-- 8000 0.0.0.0` to open it from other devices on your network)
```

On localhost the game talks to Ollama by default. The first time it connects to a model it times a few
thread counts and remembers the fastest (on hybrid Intel laptops Ollama's default is often ~1.7x slower).

## Dreaming

Dreams fine-tune the brain (LoRA on Qwen2.5-0.5B, CPU is fine) and hand the new generation to Ollama.

```sh
python -m venv .venv
.venv/bin/pip install -r trainer/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd web && npm run -s curriculum -- 600 > ../trainer/data/curriculum.jsonl   # verified "instinct" lessons
cd ../trainer
systemd-run --user --scope -p MemoryMax=8G ../.venv/bin/python -m blobb.dream --gen 1
```

Later generations add what the blob lived through: in the game, ⚙ → *Export memories for dreaming*, then

```sh
systemd-run --user --scope -p MemoryMax=8G ../.venv/bin/python -m blobb.dream --gen 2 --memories ~/Downloads/blobb-memories-*.json
```

While it sleeps, tap 🌙 in the game to peek into the dream: how far along it is, what it's dreaming
about, how wrong it still is, and a map of its brain lighting up as the dream rewrites it. When it wakes,
one click switches the game to the new brain.

A gen-1 dream takes about 45 minutes on an i7-1260P and stays under 6 GB of RAM. The trainer packs each
batch around the prompt the samples share, skips LoRA dropout, uses only the performance cores, and
recomputes activations for just as many layers as the memory budget needs. `--max-ram 7.5` trades
memory for speed; `--threads N` overrides the core count. `systemd-run ... MemoryMax` is a seatbelt: if
anything ever does blow up, only the dream dies, not your terminal.

## Develop

```sh
cd web && npm run verify     # typecheck, lint, tests
npm run dev                  # rebuild on change (serve with npm run serve)
npm run eval -- qwen2.5:0.5b blobb-gen1   # exam a brain on fixed scenarios
cd ../trainer && python -m unittest discover tests
```

- `web/src/world.ts`, `actions.ts` - the island and what the blob can do (pure, tested)
- `web/src/perception.ts`, `brain.ts` - what it sees, and the model that decides
- `web/src/game.ts`, `memory.ts`, `errands.ts` - the learning loop, memories, the villagers' letters
- `web/src/render.ts`, `main.ts`, `dream.ts` - the 3D view, the UI, the dream peek
- `web/tools/curriculum.ts` - generates simulator-verified training lessons
- `trainer/blobb/dream.py` - the dream itself

## License

MIT © 2026 Binomica Labs
