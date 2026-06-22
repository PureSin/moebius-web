// Headless verification of the TS numeric port against the validated Python fixture.
// Imports the REAL ddim.ts and replicates pipeline.unetCFG's 9-ch assembly, running the
// actual unet.onnx via onnxruntime-node. Compares final latents to the numpy reference.
import * as ort from "onnxruntime-node";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { makeDDIM, ddimStep } from "../src/ddim.ts";

const dir = path.dirname(fileURLToPath(import.meta.url));
const FX = path.join(dir, "fixture");
const MODELS = "/tmp/Moebius/moebius-web/models";
const LAT = 64;
const HALF_IDS = 10;

const f32 = (name) => new Float32Array(fs.readFileSync(path.join(FX, name)).buffer.slice(0));
const meta = JSON.parse(fs.readFileSync(path.join(FX, "meta.json"), "utf8"));

const maskedLat = f32("masked_lat.bin"); // (1,4,64,64)
const mask64 = f32("mask64.bin"); // (1,1,64,64)
const initLatents = f32("init_latents.bin"); // (1,4,64,64)
const refFinal = f32("final_latents.bin"); // (1,4,64,64)

const plane = LAT * LAT;

async function main() {
  const unet = await ort.InferenceSession.create(`${MODELS}/unet.onnx`);
  const ddim = makeDDIM(meta.steps); // recompute schedule in TS
  // verify the TS schedule matches the fixture's timesteps
  const tsMatch = JSON.stringify(ddim.timesteps) === JSON.stringify(meta.timesteps);
  console.log("[verify] DDIM timesteps match fixture:", tsMatch, ddim.timesteps.slice(0, 4), "...");

  let latents = Float32Array.from(initLatents);
  const guidance = meta.guidance;
  const tl = ddim.timesteps;

  for (let i = 0; i < tl.length; i++) {
    const t = tl[i];
    const prevT = i + 1 < tl.length ? tl[i + 1] : -1;

    // ---- 9-channel assembly (same as pipeline.ts:unetCFG) ----
    const nine = new Float32Array(9 * plane);
    nine.set(latents.subarray(0, 4 * plane), 0);
    nine.set(mask64, 4 * plane);
    nine.set(maskedLat.subarray(0, 4 * plane), 5 * plane);
    const nine2 = new Float32Array(2 * 9 * plane);
    nine2.set(nine, 0);
    nine2.set(nine, 9 * plane);

    const ids = new BigInt64Array(2 * HALF_IDS);
    for (let j = 0; j < HALF_IDS; j++) {
      ids[j] = BigInt(HALF_IDS + j);
      ids[HALF_IDS + j] = BigInt(j);
    }
    const out = await unet.run({
      latent: new ort.Tensor("float32", nine2, [2, 9, LAT, LAT]),
      timesteps: new ort.Tensor("int64", new BigInt64Array([BigInt(t), BigInt(t)]), [2]),
      input_ids: new ort.Tensor("int64", ids, [2, HALF_IDS]),
    });
    const noise = out.noise.data;
    const n = 4 * plane;
    const eps = new Float32Array(n);
    for (let k = 0; k < n; k++) eps[k] = noise[k] + guidance * (noise[n + k] - noise[k]);

    latents = ddimStep(eps, latents, t, prevT, ddim);
  }

  // compare
  let maxd = 0;
  let mean = 0;
  for (let k = 0; k < latents.length; k++) {
    const d = Math.abs(latents[k] - refFinal[k]);
    maxd = Math.max(maxd, d);
    mean += d;
  }
  mean /= latents.length;
  console.log(`[verify] final latents vs numpy reference: max|Δ|=${maxd.toExponential(3)} mean|Δ|=${mean.toExponential(3)}`);
  console.log(maxd < 1e-2 ? "[verify] PASS ✅ TS port matches the validated pipeline" : "[verify] FAIL ❌");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
