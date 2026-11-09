// Diagnostic: run the instrumented graph in onnxruntime-node; dump every output as float32 .bin.
import * as ort from '/Users/bill/projects/TwixT_Game/node_modules/onnxruntime-node/dist/index.js';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const [dir, pid, outDir] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const raw = (n) => new Uint8Array(readFileSync(`${dir}/${pid}_${n}.bin`)).buffer;
const f32 = (n) => new Float32Array(raw(n));
const i64 = (n) => new BigInt64Array(raw(n));
const s = await ort.InferenceSession.create(`${dir}/model_instr.onnx`);
const r = await s.run({ board: new ort.Tensor('float32', f32('board'), [1, 30, 24, 24]), move_rows: new ort.Tensor('int64', i64('rows'), [576]), move_cols: new ort.Tensor('int64', i64('cols'), [576]), move_mask: new ort.Tensor('float32', f32('mask'), [576]) });
const meta = {};
for (const [k, t] of Object.entries(r)) {
  const safe = k.replace(/[^A-Za-z0-9_.-]/g, '_');
  const data = t.type === 'float32' ? t.data : Float32Array.from(t.data, Number);
  writeFileSync(`${outDir}/${safe}.bin`, Buffer.from(new Float32Array(data).buffer));
  meta[k] = { file: safe, dims: t.dims, type: t.type };
}
writeFileSync(`${outDir}/meta.json`, JSON.stringify(meta));
console.log('node dumped', Object.keys(meta).length);
