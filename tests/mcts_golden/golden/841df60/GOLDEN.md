# Eager golden corpus — CAPTURED and VALID

**Date:** 2026-08-16 · **Outcome: 92 / 92 captured, corpus VALID.**

The behavioural baseline for the MCTS memory remediation
(`docs/superpowers/2026-08-16-mcts-memory-remediation-design.md` §4). These are traces of the
**unmodified eager implementation**, against which a lazy child-state implementation must
reproduce §4.3's compared fields exactly.

**This is a baseline, not a result.** It establishes no equivalence, licenses no `server/mcts.js`
change, and authorizes no falsification run, lazy capture, timing smoke, `P` selection or match.

## Provenance

| | |
|---|---|
| command | `node tests/mcts_golden/capture.mjs capture runs/mcts_golden_eager_841df60` |
| capture commit | `841df6040a740a4b9f1753253e0e8bfc63e15366` (pushed before the run) |
| pinned surface commit | `74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e` |
| execution-surface sha256 | `228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd` |
| worktree at launch | clean |
| output directory | `runs/mcts_golden_eager_841df60` (absent beforehand) |
| configuration | default Node heap, product ORT configuration, no session options |
| execution | one fresh process per case, sequential |
| started (UTC) | `2026-08-16T23:58:08Z` |
| finished (UTC) | `2026-08-17T00:00:17Z` (~2 min 09 s) |
| node | `v26.7.0` |
| onnxruntime-node | `1.23.2` |

## Result

| observation | value |
|---|---|
| **exit status** | **`0`** |
| **signal** | **`null`** — returned, not signalled |
| stdout | `capture.log`, 21,496 bytes, sha256 `13663d12fd51288188841cc9a23e37b7720fe5d6e5db2d44188f21456d7bfa95` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| artifacts | **92**, all `.json` |
| `.tmp` files | **0** |
| non-JSON entries | **0** |
| orchestrator verdict | `corpus VALID: 92 cases verified against the matrix` |
| independent re-validation | **0 failures** — `validateCorpus` re-run in a separate process, re-deriving fixtures from the pinned sidecars |

Status and signal were captured directly into a file by a subshell, with no pipeline able to
substitute another command's status. Empty stderr is a positive result: no
`SESSION_RELEASE_FAILED`, no `SESSION_RELEASE_UNAVAILABLE`, no `SECONDARY` line, and no native
abort message from any of the 92 workers.

## Corpus-level facts, independently recomputed

| quantity | value |
|---|---|
| total simulations backed up | **15,755** = `18 positions × 875` (ladder `1+2+8+64+800`) `+ 5` for A2 |
| total visit-count entries | 45,876 |
| `A1` visit-count sum | **0** — aborted before the first simulation, empty map by contract |
| `A2` visit-count sum | **5** — aborted from the progress callback at `done === 5` |

Both abort contracts held in real captures, not only against the fake used in tests.

## Corpus fingerprint

`sha256` over the sorted `filename:sha256` manifest of all 92 artifacts:

```
9e3a9037409c6eb4e72206a8b01697c1c138b0a90291462d713116106f2239f6
```

Recompute with:

```
cd tests/mcts_golden/golden/841df60/artifacts && \
  for f in $(ls *.json | sort); do printf '%s:%s\n' "$f" "$(shasum -a 256 "$f" | cut -d' ' -f1)"; done \
  | shasum -a 256
```

## Per-artifact hashes

| artifact | sha256 |
|---|---|
| `A1.json` | `11c9185d095d91a75a27dc885859728e65915ffd807bd5382b25e2101cb35b5a` |
| `A2.json` | `7a57fed0fad2f76cc5a31bcd5c6033f953a7db95ee047da6ca0a54f70bac3154` |
| `G_P01_baseline_s1.json` | `374fde34109796c91ad72f86c7d1a14b7725afedae0046c3ab3cefacf1970259` |
| `G_P01_baseline_s2.json` | `1a55872c4c15472669f523c120ffe7641e7228585d94b236133358431d7ec6a2` |
| `G_P01_baseline_s64.json` | `4d543d419786f26bf4af430bec73bbcb31ab5c45417b3216f5b09f6b6b1ac324` |
| `G_P01_baseline_s8.json` | `65eef1a3242be075ca35b74ece5d12ba37fa0af1e1f2826a3a4db4e0d195051a` |
| `G_P01_baseline_s800.json` | `3f93d7b84c35d023cac9ca7abba4b58d6a007ec4c08c41ef99b8eebe2863979f` |
| `G_P02_baseline_s1.json` | `dca1ee0f8c86bbda48cdd833b972ba47a70da84d32e5bc88e3cce15570478630` |
| `G_P02_baseline_s2.json` | `42435c84e1fe824b3e69c631f7cc54427dcddc4b99197f7634154aa665593f81` |
| `G_P02_baseline_s64.json` | `cedd06bbc68a380094897bd84c958de9343e5fd549b34e9eb4b73677ff86d1ba` |
| `G_P02_baseline_s8.json` | `e8e91098d3a8f11ea9b5cdce35d86ff90e149f6dad11b1fc4af5919f1fa07b68` |
| `G_P02_baseline_s800.json` | `c620161741dde86be265d6589b1ad6c27a97fccb27352a3a21dd2d3340beee13` |
| `G_P02_candidate_s1.json` | `6a0daf7ec3627f5b6851cc628b3baf4a74ef23b4a3482d6e8383d0f7c5496e57` |
| `G_P02_candidate_s2.json` | `70ab8dc6d6f749801129f798feb7f7190f026f7513381cd768e92656b77d28b2` |
| `G_P02_candidate_s64.json` | `9dae0ad62b571b421e5051855508caeb3cfa573a5beb8d9bf23981ca552fc5d2` |
| `G_P02_candidate_s8.json` | `fd101d1218945469395a26e87d8ec343e66c6567f8887581531684618d8c3014` |
| `G_P02_candidate_s800.json` | `69a5ce961ba585bd91d1626199d00f88168e2f669dded5124957ef7e64c34af1` |
| `G_P03_baseline_s1.json` | `86c901664484369fd156173b5d11d3dca326ee683be435ec0be96a7e1f4d9271` |
| `G_P03_baseline_s2.json` | `5b557ba92e267a61005c76727b909f495dd126bdd2050e1d5ca26819061a3dfe` |
| `G_P03_baseline_s64.json` | `2feda19a5377279d84af914d66ff2863a0877196254977abd40452b854c1df61` |
| `G_P03_baseline_s8.json` | `02ead052d422381bb19f19b3e795c0eb791665fc7dd24e92b3b25c1e4f993ee9` |
| `G_P03_baseline_s800.json` | `398dbdf5240f7600becfe546c4f394740918455f62c4f85e6665cf3d7f6f10aa` |
| `G_P04_baseline_s1.json` | `07a61b13fe437ff01fd52d4fa467597abddbaec4ca8c24ee461da08b2aa00130` |
| `G_P04_baseline_s2.json` | `5a80289ce593856a9c9693a775f09b9a2a7530852ad79881f4c6199ad6c9530a` |
| `G_P04_baseline_s64.json` | `af3e1100889e69a36449b82d1d2adf35f6869ca4ae71ef63831690f4e4297669` |
| `G_P04_baseline_s8.json` | `67858c5e66c16406e5a93c96d448bacc1aa5d2d49c537a817ba352b8aab58418` |
| `G_P04_baseline_s800.json` | `5ff557b06f8b9a5a0b17fbe9e7dee7d58f5c875b003426cdcee3c0478916789a` |
| `G_P05_baseline_s1.json` | `90c6efb3ce1d5a6ebb19eeb879077f9bb1b91baa400f8a1a09d9e0b32a40993d` |
| `G_P05_baseline_s2.json` | `a6d33a7d0786de4dd8e46840a8746fdfd44246d3ed6e29c779e7df82ab4d8463` |
| `G_P05_baseline_s64.json` | `1100653b3d07d260be2791901dd8b46e44e19d592b82f892b54ad513bbe0b796` |
| `G_P05_baseline_s8.json` | `19d83b17e58d1dd26a2a103f8bb5065f2f40ad567b9bef6669e283fe7752df8b` |
| `G_P05_baseline_s800.json` | `0faa661a81bc79f85e8a9569f93c158503ff163600d9a617200f8561cd19d08d` |
| `G_P06_baseline_s1.json` | `b68eec9ad74490ce81e03bb2fb05eed7cefe936cf81c1d236e6a0547f9426ea6` |
| `G_P06_baseline_s2.json` | `65f5bf1c851dd9feac70f5ba2806afcf8be038537ee6a84f12b08d81e67efccb` |
| `G_P06_baseline_s64.json` | `8a8abe7906c0f6c944fa95f30e89a887fdb79671c5476e385de7eeb3168d4853` |
| `G_P06_baseline_s8.json` | `2a1f56569eeade63d045eb6ec915d93356619948c7f6f7279e1b8fbc4608ee49` |
| `G_P06_baseline_s800.json` | `72b16d58126b026cec20414b55dd2d044b1ddb849aaef3c1409c36767faeb75e` |
| `G_P07_baseline_s1.json` | `562df246b460553e4a4a3b6b623c387db333a8c0bd298ff2126c4413d47beab7` |
| `G_P07_baseline_s2.json` | `5f6d98275c7b4a8b073df8075ac8148167983f4b7ec952a8893de4ee0030f871` |
| `G_P07_baseline_s64.json` | `99a73e6d7d9953c20ad8a086440bc9cd0689c9fd5fa2a9ec6af02bfa8db802d3` |
| `G_P07_baseline_s8.json` | `64406fe8e539ed0c9d84e5587231f498e5863f453ee03f0484effd81b58c8e19` |
| `G_P07_baseline_s800.json` | `fb8fb7d1c93757fa5823c2ce507324a271814cbd74e2d7f4f68b07949611ba4d` |
| `G_P08_baseline_s1.json` | `2ada4ade5e4bc46df7449c77057fcccdfe16382bd5bb69212b481c2c19f956cd` |
| `G_P08_baseline_s2.json` | `e8b63360813050195fe32f8b2cdc7db7f23f3d8aa99e66fea9aebb4f8a9d19bb` |
| `G_P08_baseline_s64.json` | `f71a999eb45c81f201d01b6c9b6f849a5ef114e9fafa7100637c0e7429a017d0` |
| `G_P08_baseline_s8.json` | `2d423417bc4dca10aafcaf3de1b0987f1645f6ae6e210e09eccefc7154fe83f8` |
| `G_P08_baseline_s800.json` | `389b4ae21ebf30fa72aa32937da4470b477f69b8e619667fa33d6bb1035b8748` |
| `G_P09_baseline_s1.json` | `c3559102cef0a294757cccc6a49a63a8edacdd4113019ad58279d580a24cd17a` |
| `G_P09_baseline_s2.json` | `8fa24b53423ac4b594a9b1c5bca6d3edbbd97e771ce56b87ca8509922b9a8050` |
| `G_P09_baseline_s64.json` | `3fed3a13088b8282397ed9104a0c133877b7e9966996c5cdecd857e48ff95df1` |
| `G_P09_baseline_s8.json` | `b6c74b026cfa2d8f4c7917e6b1174d834af112e109f24f3010977ac639c2d204` |
| `G_P09_baseline_s800.json` | `1430eb990c41a20967c9070beda079bc4e9a1bfd1d1f6e83f9b4e37dfdad728a` |
| `G_P10_baseline_s1.json` | `537e6d8622f761e5260d3082375f4e33f08ba549968db982a77d4c0951426187` |
| `G_P10_baseline_s2.json` | `215c71aed6964c375bd8ce6f5d065d72ffc728c7aff674c7be466ffe512e4632` |
| `G_P10_baseline_s64.json` | `a96419886b25083d46dad26ed0c945ccd3a5cc3b5b45f13904cbf7161ecec336` |
| `G_P10_baseline_s8.json` | `2c041cf78902a63f8a5248f1a01df5601775d5ca474ef7b536666b12fdde5a82` |
| `G_P10_baseline_s800.json` | `37591176a8cb9f6ac64c4f6f2a561402e54968ef6957eb9ffcce5cd8cf8624cd` |
| `G_P11_baseline_s1.json` | `86588937b629ba401d85c9e1825b3f37134f6c6951c3af23c7f807e0909bed93` |
| `G_P11_baseline_s2.json` | `f9dc9223c338d5b90a99a05977c292841db18999c72b8fedfee42e8918ba76d8` |
| `G_P11_baseline_s64.json` | `90aab21ef8511baa304ba63c14a0136eb04521eb14f5e79d11a7d8a59e05d58f` |
| `G_P11_baseline_s8.json` | `25bac1b65f0c17aed735499665dc4673b5f9e2563c4767dd4232b6f2fcd989bb` |
| `G_P11_baseline_s800.json` | `1e9153bec60285610ce021eccfa5a6f883297ed16c4745209d9f0e11c9077033` |
| `G_P11_candidate_s1.json` | `ed98628a3abbc3736177cfec825262b66446fc606aed078c0cc39d3f4e28af10` |
| `G_P11_candidate_s2.json` | `91b6da096c471ce191d8e1e584b11ac213e09454a11d2fa956733a9085f0d8f1` |
| `G_P11_candidate_s64.json` | `14d1f222853b83bacbb98185513bbc18518bbd6f13a71897babd5a36259d7d1b` |
| `G_P11_candidate_s8.json` | `82fcf56d2eadad9d762c85683d132192b3bfc3026df2a9c5f4ec448fdfcc7ffa` |
| `G_P11_candidate_s800.json` | `ed24e851b5da9959501cdee7632d7164f53eb85ec766fc676a7f1e23657fbf0a` |
| `G_P12_baseline_s1.json` | `cb24b01f4232d9b8729560de6edc5cc32add1db295512e428b240a10f55da652` |
| `G_P12_baseline_s2.json` | `c8c03fdab2c45e115b9b4ff7713222555b4a4647a01582e5caa8aa4c1efa7798` |
| `G_P12_baseline_s64.json` | `4046acfec29d74c67ef585e542fbf03cf1d4b89356351806ae9f9d1598341a3b` |
| `G_P12_baseline_s8.json` | `029fd17bbe48e866a8f570af5e3937fe9829aeb6079558a925553b54d12030d4` |
| `G_P12_baseline_s800.json` | `db98a483375daeabf133da3533d09d5adc8fd2e0c9607de36dc80ef665227988` |
| `G_P13_baseline_s1.json` | `5469113746a193d2039e1946f45f2be8d0c7589b057f8cd45aa22d9b75845943` |
| `G_P13_baseline_s2.json` | `3e8329a6204f8fe59a44f34aa484ceab6a6aa28aaea58d4e13bd29c67547f398` |
| `G_P13_baseline_s64.json` | `277575be76e603bcc03a308d130d79ac039ee76c4875c8b3d0187306dc0b2035` |
| `G_P13_baseline_s8.json` | `38730fccc44c3c5b566839738cf4f9c0100355c1285837cdac7809b759e53e3d` |
| `G_P13_baseline_s800.json` | `8a444fe86c36b562b876d099a8d9c1ce941124a7ee5a4b8830e8713c71bd1cba` |
| `G_P14_baseline_s1.json` | `0b006b77406f00093aa5b72f0816ac833e1195ab70eff386dc0a59c254d7e5ea` |
| `G_P14_baseline_s2.json` | `b6b6270066931fb067ebfb342a4b9a7d0c21b08e41dde391eeb70ae9eec29b91` |
| `G_P14_baseline_s64.json` | `e93742023fe9ef0c8da151d4f029b46865d84919644e1fc5e9637defa5075328` |
| `G_P14_baseline_s8.json` | `964c11a39d896ff0f3753c0d0786b6245a9d241d74c3cab2a1cbbb8794803fec` |
| `G_P14_baseline_s800.json` | `81948dbb09bef77e927772f527364bb889896a2791d4e3d1215a1f49e23f0d14` |
| `G_P15_baseline_s1.json` | `4d3e2299417a354d034497d6d444228041ba475f16f11869524f1fb725f555ea` |
| `G_P15_baseline_s2.json` | `becb4c111d45c3f0a342ac5960352e74ff56aad7352e77bc25f5962c0702b8f5` |
| `G_P15_baseline_s64.json` | `2db0d63f32fb6611e372e9c6630052e794c8bbb0f9144306fdba498322ebb940` |
| `G_P15_baseline_s8.json` | `48473ece033e8eafe26e5545b8d15e9ba334866c425fd02824d1321482986236` |
| `G_P15_baseline_s800.json` | `2ac2c7d6cdac8a94cb5d173e28ebe0e86f308345f9f0d595b6bd23e60cc9cef5` |
| `G_P16_baseline_s1.json` | `89f2d7b7b73af75bfdeb0cef2522971831f5776fedf7311ef52771f407b7725d` |
| `G_P16_baseline_s2.json` | `18cf3c917926ca9f70c37198f3b23972abd5c870daf9fd35441a027d6ffc3215` |
| `G_P16_baseline_s64.json` | `a2a9dcfdcb83bf5ab8e8502b5114b32165e26fcd2bffcc5d2980c431897d2a6e` |
| `G_P16_baseline_s8.json` | `fed198b9c99601a1e66533a9392f985a7f68baf2253c79b46ea7a40b9e5a0ad7` |
| `G_P16_baseline_s800.json` | `8c687bde734c13fd40458463b2ecd7e7e6a4b8b6b0b856011345d78c65b69c55` |

## Scope

- These traces describe the **eager** implementation as committed at execution surface
  `228f57b5…` (`74dca6e`). They are the comparison target, not a verdict.
- The lazy implementation must reproduce, exactly, the fields §4.3 lists as compared:
  `visit_counts` (values and order), `root_value`, `selected_move`, and `progress`
  (`done`/`total`/`valueEstimate`). `progress_elapsed_ms` is metadata and is **not** compared —
  it derives from `Date.now()` and no two runs reproduce it.
- Editing `server/mcts.js` moves the execution-surface digest off `228f57b5…`. That is expected
  for the remedy, and it means the *timing* smoke must be retaken; it does **not** invalidate
  this corpus, which records what the eager code did at that surface.
