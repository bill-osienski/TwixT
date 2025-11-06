#!/usr/bin/env node
/**
 * Symmetric baseline tuner for TwixT AI heuristics.
 *
 * Iterates through a small grid of neutral parameters, runs short self-play
 * batches (depth-2 and depth-3, 12 games each), and reports the win splits.
 *
 * Results are logged to stdout; the best configuration (closest to parity
 * across both depths) is highlighted. The script restores the original
 * search.json at the end.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const searchPath = path.join(projectRoot, 'assets/js/ai/search.json');
const selfPlayPath = path.join(projectRoot, 'selfplay.json');
const tempDir = path.join(projectRoot, 'temp');
const logsDir = path.join(projectRoot, 'logs');
const nextSweepPath = path.join(logsDir, 'next-sweep.json');

const originalConfig = JSON.parse(fs.readFileSync(searchPath, 'utf8'));

const edgeOffense = originalConfig.rewards.edge.offense;
const finishPenaltyBase = edgeOffense.finishPenaltyBase;
const gapDecayBase = edgeOffense.gapDecay;
const spanGainBase = edgeOffense.spanGainBase;
const redFinishPenaltyFactorBase = edgeOffense.redFinishPenaltyFactor;
const blackFinishScaleMultiplierBase = edgeOffense.blackFinishScaleMultiplier;
const redSpanGainMultiplierBase = edgeOffense.redSpanGainMultiplier;
const blackSpanGainMultiplierBase = edgeOffense.blackSpanGainMultiplier;
const redDoubleCoverageBonusBase = edgeOffense.redDoubleCoverageBonus ?? 0;
const blackDoubleCoverageScaleBase =
  edgeOffense.blackDoubleCoverageScale ?? 1.0;
const OPTIONAL_OFFENSE_KEYS = [
  'connectorBonus',
  'finishThreshold',
  'finishBonusBase',
  'connectorTargetBonus',
  'doubleCoverageBase',
  'finishGapSlope',
  'nearFinishBonus',
  'redFinishExtra',
  'redGapDecayMultiplier',
];
const optionalOffenseDefaults = OPTIONAL_OFFENSE_KEYS.reduce((acc, key) => {
  acc[key] = edgeOffense[key];
  return acc;
}, {});

const firstEdgePairs = [
  { red: 415, black: 455 },
  { red: 420, black: 455 },
  { red: 420, black: 460 },
];
const finishPenaltyOptions = [1181];
const gapDecayOptions = [gapDecayBase];
const stallOptions = [
  {
    redFinishPenaltyFactor: Number(
      (redFinishPenaltyFactorBase - 0.2).toFixed(2)
    ),
    blackFinishScaleMultiplier: Number(
      blackFinishScaleMultiplierBase.toFixed(2)
    ),
  },
  {
    redFinishPenaltyFactor: Number(
      (redFinishPenaltyFactorBase - 0.15).toFixed(2)
    ),
    blackFinishScaleMultiplier: Number(
      blackFinishScaleMultiplierBase.toFixed(2)
    ),
  },
];
const spanOptions = [
  {
    spanGainBase: spanGainBase,
    redSpanGainMultiplier: Number(redSpanGainMultiplierBase.toFixed(2)),
    blackSpanGainMultiplier: Number(blackSpanGainMultiplierBase.toFixed(2)),
  },
  {
    spanGainBase: spanGainBase,
    redSpanGainMultiplier: Number(
      (redSpanGainMultiplierBase + 0.15).toFixed(2)
    ),
    blackSpanGainMultiplier: Number(
      (blackSpanGainMultiplierBase - 0.15).toFixed(2)
    ),
  },
];
const doubleCoverageOptions = [
  {
    redDoubleCoverageBonus: 0,
    blackDoubleCoverageScale: 1.0,
  },
  {
    redDoubleCoverageBonus: 1500,
    blackDoubleCoverageScale: 0.6,
  },
];

function stableStringify(value) {
  if (Array.isArray(value)) {
    return '[' + value.map((v) => stableStringify(v)).join(',') + ']';
  }
  if (value && typeof value === 'object') {
    return (
      '{' +
      Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
        .join(',') +
      '}'
    );
  }
  return JSON.stringify(value);
}

function computeConfigHash(combo) {
  const payload = {
    firstEdgeTouchRed: combo.firstEdge.red,
    firstEdgeTouchBlack: combo.firstEdge.black,
    finishPenaltyBase: combo.finishPenalty,
    redFinishPenaltyFactor: combo.redFinishPenaltyFactor,
    blackFinishScaleMultiplier: combo.blackFinishScaleMultiplier,
    redSpanGainMultiplier: combo.redSpanGainMultiplier,
    blackSpanGainMultiplier: combo.blackSpanGainMultiplier,
    redDoubleCoverageBonus: combo.redDoubleCoverageBonus,
    blackDoubleCoverageScale: combo.blackDoubleCoverageScale,
  };
  for (const key of OPTIONAL_OFFENSE_KEYS) {
    payload[key] = combo[key];
  }
  const json = stableStringify(payload);
  return crypto.createHash('sha1').update(json).digest('hex');
}

function coerceNumber(value, fallback) {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  const asNumber = Number(value);
  return Number.isFinite(asNumber) ? asNumber : fallback;
}

function applyOptionalDefaults(target, source) {
  for (const key of OPTIONAL_OFFENSE_KEYS) {
    if (
      Object.prototype.hasOwnProperty.call(source, key) &&
      source[key] !== undefined &&
      source[key] !== null
    ) {
      target[key] = source[key];
    } else if (!Object.prototype.hasOwnProperty.call(target, key)) {
      target[key] = optionalOffenseDefaults[key];
    }
  }
}

function buildDefaultCombos() {
  const defaults = [];
  for (const firstEdge of firstEdgePairs) {
    for (const finishPenalty of finishPenaltyOptions) {
      for (const gapDecay of gapDecayOptions) {
        for (const stall of stallOptions) {
          for (const span of spanOptions) {
            for (const coverage of doubleCoverageOptions) {
              const entry = {
                firstEdge,
                finishPenalty,
                gapDecay,
                redFinishPenaltyFactor: stall.redFinishPenaltyFactor,
                blackFinishScaleMultiplier: stall.blackFinishScaleMultiplier,
                spanGainBase: span.spanGainBase,
                redSpanGainMultiplier: span.redSpanGainMultiplier,
                blackSpanGainMultiplier: span.blackSpanGainMultiplier,
                redDoubleCoverageBonus: coverage.redDoubleCoverageBonus,
                blackDoubleCoverageScale: coverage.blackDoubleCoverageScale,
                origin: 'grid',
              };
              applyOptionalDefaults(entry, {});
              defaults.push(entry);
            }
          }
        }
      }
    }
  }
  return defaults;
}

function loadPlannedCombos() {
  if (!fs.existsSync(nextSweepPath)) {
    return null;
  }
  try {
    const payload = JSON.parse(fs.readFileSync(nextSweepPath, 'utf8'));
    const planned = Array.isArray(payload.combos) ? payload.combos : [];
    if (!planned.length) {
      return null;
    }
    const mapped = planned.map((raw) => {
      const firstEdgeRed = coerceNumber(
        raw.firstEdgeRed ?? raw.firstEdge?.red,
        edgeOffense.firstEdgeTouchRed
      );
      const firstEdgeBlack = coerceNumber(
        raw.firstEdgeBlack ?? raw.firstEdge?.black,
        edgeOffense.firstEdgeTouchBlack
      );
      const entry = {
        firstEdge: { red: firstEdgeRed, black: firstEdgeBlack },
        finishPenalty: coerceNumber(
          raw.finishPenalty ?? raw.finishPenaltyBase,
          finishPenaltyBase
        ),
        gapDecay: coerceNumber(raw.gapDecay, gapDecayBase),
        redFinishPenaltyFactor: coerceNumber(
          raw.redFinishPenaltyFactor,
          redFinishPenaltyFactorBase
        ),
        blackFinishScaleMultiplier: coerceNumber(
          raw.blackFinishScaleMultiplier,
          blackFinishScaleMultiplierBase
        ),
        spanGainBase: coerceNumber(raw.spanGainBase, spanGainBase),
        redSpanGainMultiplier: coerceNumber(
          raw.redSpanGainMultiplier,
          redSpanGainMultiplierBase
        ),
        blackSpanGainMultiplier: coerceNumber(
          raw.blackSpanGainMultiplier,
          blackSpanGainMultiplierBase
        ),
        redDoubleCoverageBonus: coerceNumber(
          raw.redDoubleCoverageBonus,
          redDoubleCoverageBonusBase
        ),
        blackDoubleCoverageScale: coerceNumber(
          raw.blackDoubleCoverageScale,
          blackDoubleCoverageScaleBase
        ),
        origin: raw.origin || 'plan',
        sourceSweep: raw.sourceSweep || null,
      };
      applyOptionalDefaults(entry, raw);
      if (raw.configHash) {
        entry.configHash = raw.configHash;
      }
      return entry;
    });
    console.log(
      `Loaded ${mapped.length} combos from ${path.relative(projectRoot, nextSweepPath)}`
    );
    return mapped;
  } catch (err) {
    console.warn(
      `Failed to parse ${path.relative(projectRoot, nextSweepPath)}. Falling back to default grid.`,
      err
    );
    return null;
  }
}

let combos = loadPlannedCombos();
if (!combos) {
  combos = buildDefaultCombos();
  console.log(`Using default sweep grid (${combos.length} combos).`);
}

function writeConfig(config) {
  fs.writeFileSync(searchPath, JSON.stringify(config, null, 2));
}

function cleanOutputs() {
  if (fs.existsSync(selfPlayPath)) {
    fs.rmSync(selfPlayPath);
  }
  if (fs.existsSync(tempDir)) {
    const entries = fs.readdirSync(tempDir);
    for (const entry of entries) {
      fs.rmSync(path.join(tempDir, entry), { recursive: true, force: true });
    }
  }
}

function runSelfPlay() {
  const cmd =
    'node scripts/selfPlayParallel.js --depth-config "2:10,3:10" --workers 10 --verbose';
  execSync(cmd, {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

function tallyGames(games) {
  let red = 0;
  let black = 0;
  let draw = 0;
  let redStarts = 0;
  let blackStarts = 0;
  for (const g of games) {
    const summary = g.summary || {};
    if (summary.draw) draw += 1;
    else if (summary.winner === 'red') red += 1;
    else if (summary.winner === 'black') black += 1;
    if (summary.startingPlayer === 'red') redStarts += 1;
    else if (summary.startingPlayer === 'black') blackStarts += 1;
  }
  return { red, black, draw, redStarts, blackStarts };
}

function evaluateSelfPlay() {
  const data = JSON.parse(fs.readFileSync(selfPlayPath, 'utf8'));
  const games = data.games || [];
  const half = games.length / 2;
  const depth2Games = games.slice(0, half);
  const depth3Games = games.slice(half);
  return {
    depth2: tallyGames(depth2Games),
    depth3: tallyGames(depth3Games),
  };
}

function scoreResult(result) {
  const d2Diff = Math.abs(result.depth2.red - result.depth2.black);
  const d3Diff = Math.abs(result.depth3.red - result.depth3.black);
  return d2Diff + d3Diff;
}

function cloneConfig() {
  return JSON.parse(JSON.stringify(originalConfig));
}

const results = [];

console.log('Starting symmetric baseline sweep...\n');

for (let i = 0; i < combos.length; i++) {
  const combo = combos[i];
  const candidate = cloneConfig();

  candidate.rewards.general.redGlobalMultiplier = 1.0;
  candidate.rewards.general.blackGlobalScale = 1.0;
  candidate.rewards.general.redBaseBonus = 0;
  candidate.rewards.general.blackBasePenalty = 0;

  const offense = candidate.rewards.edge.offense;
  offense.firstEdgeTouchRed = combo.firstEdge.red;
  offense.firstEdgeTouchBlack = combo.firstEdge.black;
  offense.finishPenaltyBase = combo.finishPenalty;
  offense.gapDecay = combo.gapDecay;
  offense.redFinishExtra = 0;
  offense.redSpanGainMultiplier = combo.redSpanGainMultiplier;
  offense.redGapDecayMultiplier = 1.0;
  offense.blackFinishScaleMultiplier = combo.blackFinishScaleMultiplier;
  offense.blackSpanGainMultiplier = combo.blackSpanGainMultiplier;
  offense.blackDoubleCoverageScale = combo.blackDoubleCoverageScale;
  offense.redDoubleCoverageBonus = combo.redDoubleCoverageBonus;
  offense.redFinishPenaltyFactor = combo.redFinishPenaltyFactor;
  offense.spanGainBase = combo.spanGainBase;
  for (const key of OPTIONAL_OFFENSE_KEYS) {
    if (combo[key] !== undefined) {
      offense[key] = combo[key];
    }
  }

  writeConfig(candidate);
  cleanOutputs();

  const originLabel = combo.origin ? `[${combo.origin}] ` : '';
  console.log(
    `Combo ${i + 1}/${combos.length}: ${originLabel}` +
      `firstEdgeRed=${combo.firstEdge.red}, firstEdgeBlack=${combo.firstEdge.black}, finishPenalty=${combo.finishPenalty}, gapDecay=${combo.gapDecay}, ` +
      `redPenaltyFactor=${combo.redFinishPenaltyFactor}, blackFinishScale=${combo.blackFinishScaleMultiplier}, ` +
      `spanBase=${combo.spanGainBase}, redSpanMult=${combo.redSpanGainMultiplier}, blackSpanMult=${combo.blackSpanGainMultiplier}, ` +
      `redDoubleCov=${combo.redDoubleCoverageBonus}, blackDoubleCovScale=${combo.blackDoubleCoverageScale}`
  );
  runSelfPlay();

  const evaluation = evaluateSelfPlay();
  const score = scoreResult(evaluation);
  const configHash = combo.configHash || computeConfigHash(combo);
  results.push({ combo, evaluation, score, configHash });

  console.log(
    `  Depth2: red=${evaluation.depth2.red}, black=${evaluation.depth2.black}, draw=${evaluation.depth2.draw}`
  );
  console.log(
    `  Depth3: red=${evaluation.depth3.red}, black=${evaluation.depth3.black}, draw=${evaluation.depth3.draw}`
  );
  console.log(`  Score (lower is better): ${score}\n`);
}

writeConfig(originalConfig);

const completedAllCombos = results.length === combos.length;
results.sort((a, b) => a.score - b.score);

console.log('\n=== Sweep complete ===\n');
console.log('Top configurations (sorted by score):\n');
for (const { combo, evaluation, score } of results.slice(0, 10)) {
  console.log(
    `firstEdgeRed=${combo.firstEdge.red}, firstEdgeBlack=${combo.firstEdge.black}, finishPenalty=${combo.finishPenalty}, gapDecay=${combo.gapDecay}, ` +
      `redPenaltyFactor=${combo.redFinishPenaltyFactor}, blackFinishScale=${combo.blackFinishScaleMultiplier}, ` +
      `spanBase=${combo.spanGainBase}, redSpanMult=${combo.redSpanGainMultiplier}, blackSpanMult=${combo.blackSpanGainMultiplier}, ` +
      `redDoubleCov=${combo.redDoubleCoverageBonus}, blackDoubleCovScale=${combo.blackDoubleCoverageScale}, score=${score}, configHash=${computeConfigHash(combo)}`
  );
  console.log(
    `  Depth2 => ${evaluation.depth2.red}-${evaluation.depth2.black}-${evaluation.depth2.draw}`
  );
  console.log(
    `  Depth3 => ${evaluation.depth3.red}-${evaluation.depth3.black}-${evaluation.depth3.draw}\n`
  );
}

console.log('Original search.json restored.');

if (!completedAllCombos) {
  console.warn(
    `Sweep aborted: completed ${results.length} of ${combos.length} combos. ` +
      'Partial results were discarded.'
  );
  process.exit(1);
}

try {
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir);
  }
  const timestamp = new Date().toISOString();
  const consolidatedPath = path.join(logsDir, 'sweep-results.json');
  const consolidated = fs.existsSync(consolidatedPath)
    ? JSON.parse(fs.readFileSync(consolidatedPath, 'utf8'))
    : { sweeps: [] };
  consolidated.sweeps.push({
    timestamp,
    combos: results.map(({ combo, evaluation, score, configHash }) => ({
      firstEdgeRed: combo.firstEdge.red,
      firstEdgeBlack: combo.firstEdge.black,
      finishPenalty: combo.finishPenalty,
      gapDecay: combo.gapDecay,
      redFinishPenaltyFactor: combo.redFinishPenaltyFactor,
      blackFinishScaleMultiplier: combo.blackFinishScaleMultiplier,
      spanGainBase: combo.spanGainBase,
      redSpanGainMultiplier: combo.redSpanGainMultiplier,
      blackSpanGainMultiplier: combo.blackSpanGainMultiplier,
      redDoubleCoverageBonus: combo.redDoubleCoverageBonus,
      blackDoubleCoverageScale: combo.blackDoubleCoverageScale,
      ...OPTIONAL_OFFENSE_KEYS.reduce((acc, key) => {
        acc[key] = combo[key];
        return acc;
      }, {}),
      evaluation,
      score,
      configHash,
    })),
  });
  fs.writeFileSync(consolidatedPath, JSON.stringify(consolidated, null, 2));
  console.log(
    `Sweep results appended to ${path.relative(projectRoot, consolidatedPath)}`
  );
} catch (err) {
  console.error('Failed to write sweep log:', err);
}
