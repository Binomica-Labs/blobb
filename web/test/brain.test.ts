// The Ollama backend's thread tuning, against a fake Ollama.
import { OllamaBackend, threadCandidates } from "../src/brain";

/** Fake Ollama: per-thread-count speed in ms/token; records what was asked. */
function fakeOllama(msPerToken: (threads: number | undefined) => number) {
  const calls: { path: string; body: { options?: { num_thread?: number } } }[] = [];
  const fetchFn = (url: string, init?: { body?: string }): Promise<Response> => {
    const path = new URL(url).pathname;
    const body = JSON.parse(init?.body ?? "{}") as { options?: { num_thread?: number } };
    calls.push({ path, body });
    const json = path === "/api/tags" ? { models: [{ name: "m:latest" }] }
      : path === "/api/generate" ? { eval_count: 24, eval_duration: msPerToken(body.options?.num_thread) * 24 * 1e6 }
      : { message: { content: "{}" } };
    return Promise.resolve(new Response(JSON.stringify(json)));
  };
  return { calls, fetchFn };
}

describe("ollama thread tuning", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("tries the default plus half and three quarters of the CPUs", () => {
    expect(threadCandidates(16)).toEqual([undefined, 8, 12]);
    expect(threadCandidates(2)).toEqual([undefined]);
    expect(threadCandidates(0)).toEqual([undefined]);
  });

  it("picks the fastest, remembers it, and uses it for real requests", async () => {
    const { calls, fetchFn } = fakeOllama((t) => (t === 12 ? 10 : t === 8 ? 14 : 17));
    vi.stubGlobal("fetch", fetchFn);
    vi.stubGlobal("navigator", { hardwareConcurrency: 16 });
    const saved = new Map<string, string>();
    const cache = { get: (k: string) => saved.get(k) ?? null, set: (k: string, v: string) => { saved.set(k, v); } };
    const b = new OllamaBackend("http://x", "m", cache);
    await b.init(() => undefined);
    expect(b.threads).toBe(12);
    await b.complete("s", "u", {});
    expect(calls[calls.length - 1]?.body.options?.num_thread).toBe(12);

    // Next time: no re-tuning, just one warm-up.
    calls.length = 0;
    const again = new OllamaBackend("http://x", "m", cache);
    await again.init(() => undefined);
    expect(again.threads).toBe(12);
    expect(calls.filter((c) => c.path === "/api/generate")).toHaveLength(1);
  });

  it("keeps Ollama's default unless something is clearly faster", async () => {
    const { calls, fetchFn } = fakeOllama((t) => (t === undefined ? 10 : 9.8));
    vi.stubGlobal("fetch", fetchFn);
    vi.stubGlobal("navigator", { hardwareConcurrency: 16 });
    const b = new OllamaBackend("http://x", "m");
    await b.init(() => undefined);
    expect(b.threads).toBeUndefined();
    await b.complete("s", "u", {});
    expect(calls[calls.length - 1]?.body.options?.num_thread).toBeUndefined();
  });
});
