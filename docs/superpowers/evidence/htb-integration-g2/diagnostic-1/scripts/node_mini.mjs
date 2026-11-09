import * as ort from '/Users/bill/projects/TwixT_Game/node_modules/onnxruntime-node/dist/index.js';
import { readFileSync } from 'node:fs';
const [path, xpath] = process.argv.slice(2);
const buf = readFileSync(xpath); const hdr = buf.readUInt16LE(8); const x = new Float32Array(new Uint8Array(buf.subarray(10 + hdr)).buffer);
const s = await ort.InferenceSession.create(path);
const out = await s.run({ x: new ort.Tensor('float32', x, [1, 128, 24, 24]) });
const a = out.y0.data, b = out.y1.data; let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
console.log(path.split('/').pop(), 'node: max |y0 - y1| =', d.toExponential(3), '(0 would mean the two convs were conflated)');
