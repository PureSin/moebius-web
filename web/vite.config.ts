import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";

const COI = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// Serve the onnxruntime-web runtime assets (.wasm / glue .mjs) as raw static files
// from ./ort-dist at /ort/*. They must NOT live in /public, because ORT loads its glue
// via dynamic import() and Vite would try to transform a /public .mjs as a source module.
function ortStatic(): Plugin {
  const dir = path.resolve(__dirname, "ort-dist");
  const mime: Record<string, string> = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
  };
  const handler = (req: any, res: any, next: any) => {
    const url: string = req.url || "";
    if (!url.startsWith("/ort/")) return next();
    const rel = url.slice("/ort/".length).split("?")[0];
    const fp = path.join(dir, rel);
    if (!fp.startsWith(dir) || !fs.existsSync(fp)) return next();
    res.setHeader("Content-Type", mime[path.extname(fp)] || "application/octet-stream");
    for (const [k, v] of Object.entries(COI)) res.setHeader(k, v);
    fs.createReadStream(fp).pipe(res);
  };
  return {
    name: "ort-static",
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

export default defineConfig({
  plugins: [ortStatic()],
  server: { headers: COI, fs: { allow: [".."] } },
  preview: { headers: COI },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
