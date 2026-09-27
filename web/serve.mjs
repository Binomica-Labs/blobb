// Tiny static server for public/ - no dependencies, so it also runs in Termux.
//   node serve.mjs [port] [host]    host defaults to this computer only; 0.0.0.0 shares it on your network
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, sep } from "node:path";

const root = join(import.meta.dirname, "public");
// Peeking into the blob's dream: the trainer's progress files, read-only, by fixed name only.
const trainerData = join(import.meta.dirname, "..", "trainer", "data");
const DREAM = { "/dream/status.json": "dream_status.json", "/dream/map.json": "dream_map.json" };
const port = Number(process.argv[2] ?? 8000);
const host = process.argv[3] ?? "127.0.0.1";
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".webmanifest": "application/manifest+json", ".png": "image/png", ".svg": "image/svg+xml", ".map": "application/json",
};

async function handle(req, res) {
  let url;
  try {
    url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  } catch {
    res.writeHead(400).end("bad url"); // malformed %-escape: refuse it, don't crash the server
    return;
  }
  if (Object.hasOwn(DREAM, url)) {
    try {
      const body = await readFile(join(trainerData, DREAM[url]));
      res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" }).end(body);
    } catch {
      res.writeHead(404).end("no dream");
    }
    return;
  }
  const path = normalize(join(root, url.endsWith("/") ? url + "index.html" : url));
  // root + separator, so a sibling like public-old/ doesn't pass as "inside public".
  if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403).end(); return; }
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "application/octet-stream", "cache-control": "no-cache" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}

createServer((req, res) => {
  handle(req, res).catch(() => { if (!res.headersSent) res.writeHead(500); res.end(); });
}).listen(port, host, () => { console.log(`Blobb on http://${host === "0.0.0.0" ? "<this computer's address>" : host}:${port}`); });
