/**
 * ONNX Runtime shutdown probe: one tiny synthetic inference, then exit in the chosen way.
 *
 * Why: Diagnostic O's driver (ledger entry 64) wrote a COMPLETED record and then aborted at `process.exit(0)` with
 * "libc++abi: ... mutex lock failed: Invalid argument" (exit 134). The user reproduced it on onnxruntime-node 1.23.2:
 * a forced exit crashed 3/3 times; setting `process.exitCode` and letting Node exit naturally succeeded 3/3 times;
 * releasing the session before a forced exit still crashed. It matches ONNX Runtime's documented macOS shutdown bug
 * (microsoft/onnxruntime#26445).
 *
 * Usage: node tests/fixtures/onnx_shutdown/probe.mjs <natural|forced|release-forced>
 *   natural         set process.exitCode = 0 and return (the required pattern for drivers that load ONNX Runtime)
 *   forced          process.exit(0)                      (expected to abort on an affected runtime; manual use only)
 *   release-forced  session.release(), then process.exit(0)  (likewise)
 * Prints "inference ok" after a correct inference. server/test_ort_shutdown.js runs the natural mode.
 */
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';

const MODEL = join(dirname(fileURLToPath(import.meta.url)), 'tiny_add.onnx');
const mode = process.argv[2];
if (!['natural', 'forced', 'release-forced'].includes(mode)) {
  console.error('usage: probe.mjs <natural|forced|release-forced>');
  process.exitCode = 2;
} else {
  const session = await ort.InferenceSession.create(MODEL); // the product's call (server/inference.js), default options
  const x = new ort.Tensor('float32', Float32Array.from([0, 1, 2, 3]), [1, 4]);
  const { y } = await session.run({ x });
  if (Array.from(y.data).join(',') !== '1,2,3,4') {
    console.error(`wrong output ${Array.from(y.data)}`);
    process.exitCode = 1;
  } else {
    console.log('inference ok');
    if (mode === 'natural') process.exitCode = 0;
    else {
      if (mode === 'release-forced') await session.release();
      process.exit(0);
    }
  }
}
