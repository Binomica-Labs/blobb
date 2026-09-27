# Blobb

You are a very old wizard, and you are done with people. So you opened your spellbook and pulled a
squishy blob out of another dimension to run the villagers' errands for you.

The blob's mind is its own: a tiny brain grown from scratch on lessons the island itself checks, small
enough to run in any browser. You tell it what to do in plain words; it thinks up a short plan, tries
it in the world, fails, hears why, and tries again. It learns in three ways:

- **Memories** - every attempt, what went wrong, and what you thought of it (👍/👎). Similar situations
  bring the right memories back into its head.
- **Skills** - name a plan that worked (⭐) and it can do the whole thing as one step.
- **Dreams** (🌙) - when it sleeps, what it lived through is trained into its brain for real: what
  worked and what you praised, and - just as much - what failed and what you 👎'd. The next generation
  knows it without being reminded.

Villagers write in with requests - a mushroom for soup, a berry from the high meadow, a bridge over the
gap. Each one needs the blob to figure out a different use of its body: a spring to jump up, a puddle to
slide under the low gate, a hard ball to knock fruit out of a tree.

## Play

**In your browser:** https://binomica-labs.github.io/blobb/ - its brain is a ~6 MB download and thinks
in a fraction of a second, on any device.

**Locally** (so it can dream):

```sh
cd web && npm ci && npm run build && npm run serve    # http://localhost:8000 (add `-- 8000 0.0.0.0` to open it from other devices on your network)
```

### Its own brain

About 2.8 million weights - a small transformer that reads what the blob sees and writes the plan as
steps, so it can only ever say something the island understands. It has never read the internet: it
knows exactly what its lessons taught it. Those come from `web/tools/curriculum.ts`, which makes up
situations, tries plans in the simulator, and keeps the ones that work - and says each request the many
ways a grumpy wizard might (`web/tools/paraphrases.ts`: synonyms, typos, SHOUTING, "you useless jelly").
Words it has never seen still get read by their spelling, so "shroom" lands near "mushroom".

### Or a borrowed LLM

⚙ → Brain can instead use Qwen2.5-0.5B, in the browser via WebGPU (~350 MB once) or through Ollama
(`ollama pull qwen2.5:0.5b`). It understands far stranger wording, and uses recalled memories and skills
in its prompt - but it's slower, and its dreams take 45 minutes instead of a few. The first time it
connects to Ollama it times a few thread counts and remembers the fastest.

## Dreaming

In the game, ⚙ → *Export memories for dreaming*, then:

```sh
python -m venv .venv
.venv/bin/pip install -r trainer/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd web && npm run -s curriculum -- 80000 > ../trainer/data/scratch/curriculum.jsonl   # lessons to replay
cd ../trainer
systemd-run --user --scope -p MemoryMax=4G ../.venv/bin/python -m blobb.scratch dream --gen 1 --memories ~/Downloads/blobb-memories-*.json
```

Good memories are learned (the ones you praised up to 3x); failed plans teach it not to take the step that
failed, and a 👎 puts it off every choice in the plan. Each generation builds on the last (`--gen 2`, ...)
and keeps its whole life's memories next to its weights (`memories.jsonl`), so every dream goes over all
of them again - its instinct lessons are replayed too, and without that they'd quietly wash out what you
taught it a few nights before. A 👎 only says what *not* to do: expect it to fumble for something else
first, and to settle on what you 👍 over the next dream or two.

A new release hatches a new brain. To carry a blob's whole life over to it, dream its last generation's
memories onto the new hatchling:

```sh
python -m blobb.scratch dream --gen 4 --from ../web/public/brain --memories data/scratch/brains/gen3
``` While it sleeps, tap 🌙 in the game
to peek into the dream: how far along it is, what it's dreaming about, how wrong it still is, and a map of
its brain lighting up as the dream rewrites it. When it wakes, one click switches the game to the new
brain.

To grow a new brain from nothing (after changing the island or the lessons), hatch one - about 45 minutes
on four laptop cores; it replaces `web/public/brain`:

```sh
systemd-run --user --scope -p MemoryMax=4G ../.venv/bin/python -m blobb.scratch hatch
```

### The LLM's dreams

These fine-tune Qwen2.5-0.5B (LoRA, CPU is fine) and hand the new generation to Ollama. They learn only
from what went well.

```sh
.venv/bin/pip install -r trainer/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd web && npm run -s curriculum -- 600 > ../trainer/data/curriculum.jsonl   # verified "instinct" lessons
cd ../trainer
systemd-run --user --scope -p MemoryMax=8G ../.venv/bin/python -m blobb.dream --gen 1
systemd-run --user --scope -p MemoryMax=8G ../.venv/bin/python -m blobb.dream --gen 2 --memories ~/Downloads/blobb-memories-*.json
```

A gen-1 dream takes about 45 minutes on an i7-1260P and stays under 6 GB of RAM. The trainer packs each
batch around the prompt the samples share, skips LoRA dropout, uses only the performance cores, and
recomputes activations for just as many layers as the memory budget needs. `--max-ram 7.5` trades
memory for speed; `--threads N` overrides the core count. `systemd-run ... MemoryMax` is a seatbelt: if
anything ever does blow up, only the dream dies, not your terminal.

## Develop

```sh
cd web && npm run verify     # typecheck, lint, tests
npm run dev                  # rebuild on change (serve with npm run serve)
npm run eval -- scratch qwen2.5:0.5b       # exam brains on fixed scenarios; --novel uses wordings no lesson has
cd ../trainer && python -m unittest discover tests
```

- `web/src/world.ts`, `actions.ts` - the island and what the blob can do (pure, tested)
- `web/src/perception.ts`, `scratch.ts`, `brain.ts` - what it sees, its own brain, and the borrowed LLMs
- `web/src/game.ts`, `memory.ts`, `errands.ts` - the learning loop, memories, the villagers' letters
- `web/src/render.ts`, `main.ts`, `dream.ts` - the 3D view, the UI, the dream peek
- `web/tools/curriculum.ts`, `paraphrases.ts` - simulator-verified lessons, and the wizard's many ways of saying things
- `trainer/blobb/scratch.py` - its own brain: hatching and dreaming
- `trainer/blobb/dream.py` - the LLM's dream

## License

MIT © 2026 Binomica Labs
