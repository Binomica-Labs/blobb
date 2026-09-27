// Bundle src/main.ts into public/js. WebLLM is split into its own chunk and only
// fetched when the in-browser brain is used.
//   node build.mjs          production build
//   node build.mjs --dev    sourcemaps, no minify
//   node build.mjs --watch  dev + rebuild on change

import { context, build } from "esbuild";
import { rmSync } from "node:fs";

const watch = process.argv.includes("--watch");
const dev = watch || process.argv.includes("--dev");

rmSync("public/js", { recursive: true, force: true });

const options = {
  // The own brain thinks in a worker: its own entry, next to main.js (see ScratchBackend).
  entryPoints: ["src/main.ts", "src/scratch-worker.ts"],
  bundle: true,
  splitting: true,
  format: "esm",
  target: "es2022",
  outdir: "public/js",
  chunkNames: "chunks/[name]-[hash]",
  minify: !dev,
  sourcemap: dev,
  logLevel: "info",
};

if (watch) {
  const ctx = await context(options);
  await ctx.watch();
  console.log("watching src/ ...");
} else {
  await build(options);
}
