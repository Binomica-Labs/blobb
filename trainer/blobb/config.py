from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRAINS = DATA / "brains"
DREAM_STATUS_FILE = DATA / "dream_status.json"
DREAM_MAP_FILE = DATA / "dream_map.json"       # the "weight map" the game shows while the blob dreams
CURRICULUM = DATA / "curriculum.jsonl"

OLLAMA_BASE = "qwen2.5:0.5b"                # gen-0 brain, straight from Ollama (template source)
HF_BASE = "Qwen/Qwen2.5-0.5B-Instruct"      # same weights, trainable

# Keep this many trained generations on disk / in Ollama (older ones are pruned).
KEEP_GENS = 2

BRAINS.mkdir(parents=True, exist_ok=True)
