#!/usr/bin/env node
/**
 * Direct comparison of two captured golden corpora, each validated under its own stage, on the frozen compared
 * fields with exact equality.
 *
 *   node tests/mcts_golden/compare_pair.mjs <reference_dir> --stage <name> <subject_dir> --stage <name>
 *
 * Built for the current-environment equivalence control (ledger decision note of 2026-11-09, after entry 59):
 * old code (a fresh capture at aad5796, stage lazy) against the integrated code (V7's capture, stage
 * protocol-hard-pinned), in the SAME environment. It reuses compare.mjs's exported compareArtifacts
 * (COMPARED_FIELDS, EXCLUDED_FIELDS) and cases.mjs's validateCorpus, unedited. Neither side is the eager standard.
 *
 * Exit: 0 EXACT MATCH, 1 MISMATCH, 2 usage, 3 corpus invalid, 4 error.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { EXPECTED_CASE_COUNT, artifactName, describeThrown, enumerateCases, stageConfig, validateCorpus } from './cases.mjs';
import { COMPARED_FIELDS, EXCLUDED_FIELDS, compareArtifacts, corpusFingerprint } from './compare.mjs';
import { deriveExpectedFixtures } from './worker.mjs';

function loadValidated(dir, stage, expectedFixtures) {
  stageConfig(stage);
  const first = enumerateCases()[0];
  const commit = JSON.parse(readFileSync(join(dir, artifactName(first.caseId)), 'utf8')).capture_commit;
  const failures = validateCorpus(dir, { mode: 'capture', stage, expectedCaptureCommit: commit, expectedFixtures, readdirSync, readFileSync });
  return { commit, failures, fingerprint: corpusFingerprint(dir) };
}

export function comparePair({ refDir, refStage, subDir, subStage }) {
  const fx = deriveExpectedFixtures();
  const ref = loadValidated(refDir, refStage, fx);
  const sub = loadValidated(subDir, subStage, fx);
  if (ref.failures.length || sub.failures.length) {
    return { outcome: 'invalid', refFailures: ref.failures.length, subFailures: sub.failures.length };
  }
  const mismatches = [];
  let compared = 0;
  for (const c of enumerateCases()) {
    const name = artifactName(c.caseId);
    mismatches.push(
      ...compareArtifacts(c.caseId, JSON.parse(readFileSync(join(refDir, name), 'utf8')), JSON.parse(readFileSync(join(subDir, name), 'utf8')))
    );
    compared += 1;
  }
  if (compared !== EXPECTED_CASE_COUNT) throw new Error(`compared ${compared}`);
  return {
    outcome: mismatches.length ? 'mismatch' : 'match',
    comparedCases: compared,
    comparedFields: [...COMPARED_FIELDS],
    excludedFields: [...EXCLUDED_FIELDS],
    reference: { stage: refStage, captureCommit: ref.commit, fingerprint: ref.fingerprint },
    subject: { stage: subStage, captureCommit: sub.commit, fingerprint: sub.fingerprint },
    mismatches,
  };
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const a = process.argv.slice(2);
  if (a.length !== 6 || a[1] !== '--stage' || a[4] !== '--stage') {
    console.error('usage: compare_pair.mjs <reference_dir> --stage <name> <subject_dir> --stage <name>');
    process.exitCode = 2;
  } else {
    try {
      const r = comparePair({ refDir: a[0], refStage: a[2], subDir: a[3], subStage: a[5] });
      console.log(JSON.stringify({ ...r, mismatches: (r.mismatches ?? []).slice(0, 20) }, null, 2));
      if (r.outcome === 'invalid') {
        console.log('CORPUS INVALID');
        process.exitCode = 3;
      } else if (r.outcome === 'match') {
        console.log(`EXACT MATCH: ${r.comparedCases} cases on ${r.comparedFields.join(', ')}; excluded ${r.excludedFields.join(', ')}`);
        process.exitCode = 0;
      } else {
        console.log(`MISMATCH: ${r.mismatches.length} compared-field difference(s)`);
        process.exitCode = 1;
      }
    } catch (err) {
      console.error(`ERROR: ${describeThrown(err)}`);
      process.exitCode = 4;
    }
  }
}
