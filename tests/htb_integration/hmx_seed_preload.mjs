/**
 * TEST-ONLY seed preload for the H-MX integration check (proposal revision 1, §5.2). Never imported by product code.
 *
 *   HMX_SEED_QUEUE=<seeds.json> node --import ./tests/htb_integration/hmx_seed_preload.mjs server/index.js
 *
 * server/index.js never passes seeds to protocolMove, which draws two fresh 48-bit streams per hard request with
 * `randomInt(0, 2 ** 48 - 1)` (server/protocol_move.js:42, search then readout at :58). This preload replaces
 * crypto.randomInt for exactly that call, serving it from the queue file (a JSON array of integers, consumed in
 * order), then calls syncBuiltinESMExports() so that `import { randomInt } from 'node:crypto'` sees the replacement.
 * Every other randomInt call passes through unchanged.
 *
 * Fail-closed:
 * - it refuses to load unless HMX_SEED_QUEUE names a readable JSON array of safe non-negative integers below 2**48;
 * - every value served is logged to stderr as `HMX_SEED <index> <value>`, so the driver can check the order;
 * - an exhausted queue throws (the request fails; nothing is drawn fresh in its place).
 */
import crypto from 'node:crypto';
import { readFileSync } from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';

export const SEED_MAX = 2 ** 48 - 1;

const path = process.env.HMX_SEED_QUEUE;
if (!path) throw new Error('hmx_seed_preload: HMX_SEED_QUEUE is not set; refusing to load');
const queue = JSON.parse(readFileSync(path, 'utf8'));
if (!Array.isArray(queue) || !queue.every((v) => Number.isSafeInteger(v) && v >= 0 && v <= SEED_MAX)) {
  throw new Error(`hmx_seed_preload: ${path} is not an array of integers in [0, 2**48 - 1]`);
}

const original = crypto.randomInt;
let next = 0;
crypto.randomInt = function hmxRandomInt(...args) {
  if (args.length === 2 && args[0] === 0 && args[1] === SEED_MAX) {
    if (next >= queue.length) throw new Error(`hmx_seed_preload: seed queue exhausted after ${queue.length} values`);
    const v = queue[next];
    process.stderr.write(`HMX_SEED ${next} ${v}\n`);
    next += 1;
    return v;
  }
  return original.apply(crypto, args);
};
syncBuiltinESMExports();
process.stderr.write(`HMX_SEED_PRELOAD ${queue.length} values from ${path}\n`);
