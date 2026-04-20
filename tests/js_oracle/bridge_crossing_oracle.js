#!/usr/bin/env node
/**
 * Bridge Crossing Oracle - Node.js script for cross-validation with Python
 *
 * Reads JSON test cases from stdin, outputs results to stdout.
 * Used by test_bridge_crossing_oracle.py to verify JS matches Python.
 *
 * Input format (JSON):
 * {
 *   "test_cases": [
 *     {
 *       "bridges": [[[r1,c1], [r2,c2]], ...],  // existing bridges
 *       "candidate": [[r1,c1], [r2,c2]]        // candidate bridge to test
 *     },
 *     ...
 *   ]
 * }
 *
 * Output format (JSON):
 * {
 *   "results": [true/false, ...]  // does candidate cross any existing bridge?
 * }
 */

import * as readline from 'readline';

// ============================================================================
// Bridge Crossing Logic (matches twixtGame.js)
// ============================================================================

/**
 * Orientation test for three points.
 * Returns: 1 if CCW, -1 if CW, 0 if collinear
 */
function orient(ax, ay, bx, by, cx, cy) {
  const v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return v > 0 ? 1 : v < 0 ? -1 : 0;
}

/**
 * Fast proper intersection test for TwixT knight-edges.
 *
 * For knight-move segments (delta ±1,±2 or ±2,±1):
 * - Collinear overlaps cannot happen between distinct knight-edges
 * - No interior lattice points exist (gcd(1,2)=1)
 * - Only need pure orientation test
 */
function properIntersectKnight(x1, y1, x2, y2, x3, y3, x4, y4) {
  const o1 = orient(x1, y1, x2, y2, x3, y3);
  const o2 = orient(x1, y1, x2, y2, x4, y4);
  if (o1 === 0 || o2 === 0 || o1 === o2) return false;

  const o3 = orient(x3, y3, x4, y4, x1, y1);
  const o4 = orient(x3, y3, x4, y4, x2, y2);
  if (o3 === 0 || o4 === 0 || o3 === o4) return false;

  return true;
}

/**
 * Check if candidate bridge crosses any existing bridge.
 *
 * @param {Array} bridges - Array of [[r1,c1], [r2,c2]] bridge endpoints
 * @param {Array} candidate - [[r1,c1], [r2,c2]] candidate bridge endpoints
 * @returns {boolean} - True if candidate crosses any existing bridge
 */
function bridgesCross(bridges, candidate) {
  if (bridges.length === 0) return false;

  const [[r1, c1], [r2, c2]] = candidate;

  // Candidate endpoints (x=col, y=row)
  const a1x = c1,
    a1y = r1;
  const a2x = c2,
    a2y = r2;

  // Candidate bbox
  const a_minx = a1x < a2x ? a1x : a2x;
  const a_maxx = a1x < a2x ? a2x : a1x;
  const a_miny = a1y < a2y ? a1y : a2y;
  const a_maxy = a1y < a2y ? a2y : a1y;

  for (const bridge of bridges) {
    const [[br1, bc1], [br2, bc2]] = bridge;

    // Sharing an endpoint is legal, not a crossing
    if (
      (r1 === br1 && c1 === bc1) ||
      (r1 === br2 && c1 === bc2) ||
      (r2 === br1 && c2 === bc1) ||
      (r2 === br2 && c2 === bc2)
    ) {
      continue;
    }

    // Bbox rejection (cheap - skips most bridges)
    const b_minx = bc1 < bc2 ? bc1 : bc2;
    const b_maxx = bc1 < bc2 ? bc2 : bc1;
    if (b_maxx < a_minx || b_minx > a_maxx) continue;

    const b_miny = br1 < br2 ? br1 : br2;
    const b_maxy = br1 < br2 ? br2 : br1;
    if (b_maxy < a_miny || b_miny > a_maxy) continue;

    // Proper intersection only (fast for knight edges)
    if (properIntersectKnight(a1x, a1y, a2x, a2y, bc1, br1, bc2, br2)) {
      return true;
    }
  }
  return false;
}

// ============================================================================
// Main: Read stdin, process, write stdout
// ============================================================================

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  let inputData = '';

  for await (const line of rl) {
    inputData += line;
  }

  try {
    const input = JSON.parse(inputData);
    const results = [];

    for (const testCase of input.test_cases) {
      const { bridges, candidate } = testCase;
      const crosses = bridgesCross(bridges, candidate);
      results.push(crosses);
    }

    console.log(JSON.stringify({ results }));
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
