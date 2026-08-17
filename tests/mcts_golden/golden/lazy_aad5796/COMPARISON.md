# Lazy corpus and the exact comparison — EXACT MATCH

**Date:** 2026-08-17 · **Outcome: 92 / 92 captured and validated; comparison EXACT MATCH.**

The lazy implementation's traces, and the result of comparing them against the frozen eager
corpus (`../841df60/`, fingerprint `9e3a9037…`) on the fields design §4.3 fixes.

## PRECOMMITTED INTERPRETATION — fixed before this ran

**A PASS means exact agreement on those fields, across these 92 cases only.**

It does **not** establish global implementation equivalence, heap safety, performance, strength,
or correctness outside the frozen corpus. This wording was committed with the comparator
(`aad5796`) before any lazy trace existed, and is printed by the tool itself.

## The capture

| | |
|---|---|
| command | `node tests/mcts_golden/capture.mjs capture runs/mcts_golden_lazy_aad5796 --stage lazy` |
| capture commit | `aad579675967ba86a21e4b5faf826e685941bb3e` |
| stage | `lazy` |
| execution surface | `d7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769` |
| surface origin commit | `85894b93392e63ce8f6e008f368ff7e798f91853` |
| artifact schema | `twixt-mcts-golden/2` (carries `stage`) |
| worktree at launch | clean |
| output directory | `runs/mcts_golden_lazy_aad5796` (absent beforehand) |
| configuration | default Node heap, product ORT configuration, no session options |
| execution | one fresh process per case, sequential |
| started / finished (UTC) | `2026-08-17T15:18:25Z` / `2026-08-17T15:19:57Z` (~1 min 32 s) |
| node · onnxruntime-node | `v26.7.0` · `1.23.2` |

| observation | value |
|---|---|
| **exit status** | **`0`** · signal `null` |
| stdout | `capture.log`, 21,424 bytes, sha256 `cf8770b4969758a2f4a090fb9dcc378aa7cb4f1cea5a4c44e850e1f191677be4` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| artifacts | **92**, all `.json`; `.tmp` **0**; non-JSON **0** |
| orchestrator verdict | `corpus VALID: 92 cases verified against the matrix` |

## The comparison

| | |
|---|---|
| command | `node tests/mcts_golden/compare.mjs runs/mcts_golden_lazy_aad5796` |
| run (UTC) | `2026-08-17T15:20:11Z` |
| **exit status** | **`0`** — `EXACT MATCH` · signal `null` |
| stdout | `compare.stdout.txt`, 821 bytes, sha256 `edd39f20d98d9a0ef0af3baeda7b4570b05955581623dad29f90708ad884d338` |
| stderr | **0 bytes** |
| cases compared | **92** · mismatches **0** |
| eager standard | fingerprint `9e3a9037…`, capture commit `841df60…`, stage `eager` |
| lazy subject | capture commit `aad5796…`, stage `lazy` |

**Compared:** `visit_counts` values **and order**, `root_value`, `selected_move`, and `progress`
restricted to `done` / `total` / `valueEstimate`.

**Excluded:** `progress_elapsed_ms` — wall-clock metadata derived from `Date.now()`, which no two
runs reproduce. Requiring it would fail a correct implementation.

Re-running the comparator against the **preserved copy** in `artifacts/` reproduces `EXACT MATCH`.

## Corpus fingerprint

`sha256` over the sorted `filename:sha256` manifest of all 92 lazy artifacts:

```
5082ac2dbd0fd49c1489f4eeace3ba3ff675ae1a0f611752a9b63c46c7483cfc
```

It necessarily differs from the eager corpus's `9e3a9037…`: the artifacts record a different
stage, schema, surface, capture commit, pids and `progress_elapsed_ms`. **What matches is the
compared projection, not the files.**

Recompute with:

```
cd tests/mcts_golden/golden/lazy_aad5796/artifacts && \
  for f in $(ls *.json | sort); do printf '%s:%s\n' "$f" "$(shasum -a 256 "$f" | cut -d' ' -f1)"; done \
  | shasum -a 256
```

## What this does NOT establish

- **Not global equivalence.** 16 positions, five simulation counts, two models, two abort cases.
  Nothing outside the frozen corpus was compared.
- **Nothing about heap safety.** §6's default-heap measurement at 800 simulations against the
  ≤ 512 MB ceiling has not been run.
- **Nothing about performance or timing.** No throughput was measured; the ten-game smoke has not
  been re-run at this surface.
- **Nothing about playing strength**, which this programme never measured on either side.

## Per-artifact hashes

| artifact | sha256 |
|---|---|
| `A1.json` | `4bf9addc0afa2deee6b0c3b80ab586dd3b03de9ed5aa315c0478cee41d92800a` |
| `A2.json` | `bf277eb0572cc50f94006f3fd26f8b89c2442be0153ac0f9fa9c9d74d487c286` |
| `G_P01_baseline_s1.json` | `d67f3cceb53914dacbd788ce9047a96a22d7cd03ff9717845c7e0eee6df712d5` |
| `G_P01_baseline_s2.json` | `e3f626e4dea0a91fdc8a3244ec1eec66421962b91b80046de582862f9c1643de` |
| `G_P01_baseline_s64.json` | `6cb94f1b4a98a61379510172cc69395f61885a6899eac4c0c8f68cfab145aa4a` |
| `G_P01_baseline_s8.json` | `4b293b2814ba21892af1e561a916bb3bf0f5c1145451e0ecdb9a9e7f17f1b423` |
| `G_P01_baseline_s800.json` | `5675f073834bb71db2e854ba8e38fde6193b1f91a8bb642709a7392e9534d793` |
| `G_P02_baseline_s1.json` | `2f70c5ddb17ce5917b458b9a3096e34b8a3d15a8d1be1a8d207185c4d4fb0022` |
| `G_P02_baseline_s2.json` | `0bab41a119d16a1f66ba144aeafe62deb808899ac73057fd3cf3562c86bb71cb` |
| `G_P02_baseline_s64.json` | `8a3da980849e6fbef1df8a2cd75559d7fee037e2d6d6483e356bdaed102ff317` |
| `G_P02_baseline_s8.json` | `1ebc74f89c6db36958fd6f59f9f318d43f97fe5af57b12fe80c1295a6d760e06` |
| `G_P02_baseline_s800.json` | `d98702d9244151459b20297fa232a1ef214a021f91fcd24cb09b934815b85707` |
| `G_P02_candidate_s1.json` | `3ac0100b581ddb393ea400d90e51b9d6051eeacb70c5f8acb2592c64fa1b6b9a` |
| `G_P02_candidate_s2.json` | `e41f639a5e2bca1b8dc4a6477727864516d9d0e1c040f98902d888239ba80692` |
| `G_P02_candidate_s64.json` | `e659daa4b08a04c4a39bdd787ea7152e9baa12a51ae6668e8acbf2f371e31ce2` |
| `G_P02_candidate_s8.json` | `dd235fcff6d32e7326620523899a0700a39fa7cd7d7b91b71fc79c0b5574c6ee` |
| `G_P02_candidate_s800.json` | `c91321bed3a5f4774f55d8b8ec5bbe5f6332105ec43bce164cde2b1d22896bde` |
| `G_P03_baseline_s1.json` | `6c42d56f94c8d695fb6d70c5b2fad1a38ae0054a1e10c3bdb57c005ee9faac15` |
| `G_P03_baseline_s2.json` | `caf517690f33d170d9d6333e33775ab4b4221e17984cc3a124f6e69433863d20` |
| `G_P03_baseline_s64.json` | `1df56edc189d2c95497e32bf2c4236f4e286837f4ce97e94430a095a4d0d67a0` |
| `G_P03_baseline_s8.json` | `6d667f123c74554fb0c78ce868fd1856fbe4cdeafa20ddff6062b6fd93b2bcc4` |
| `G_P03_baseline_s800.json` | `48d8de4501b1a654ea0165b54f5a02d5e298f50537848ef5ea8db564594a55e0` |
| `G_P04_baseline_s1.json` | `ab5043b67dabd48caf7d158143b62cbe06670282f306ae36cdec18d3fbbbe498` |
| `G_P04_baseline_s2.json` | `aaca6e0aee28c853a8066f7a2a1e1be5af66f4e38c6b183049e8b01a56e972f3` |
| `G_P04_baseline_s64.json` | `a29174c8aeab8597ab8bc8373d42e33edcde196d0398acaddb4f39043ac594ae` |
| `G_P04_baseline_s8.json` | `9bd12d5383f7d0b80477300f475e5ada471fbdade01be74b89fb27bcdee19316` |
| `G_P04_baseline_s800.json` | `17318cc97163a1db41127efa6d47e61571decc0d4cf509b37338c0a200f0e165` |
| `G_P05_baseline_s1.json` | `6063692a74ab4cb30685492e0277f3dafc2d7fcafccaf1105421fa94d2ca0d43` |
| `G_P05_baseline_s2.json` | `f3909b0edeb800fbe2150f958d0727f35659f8164f8a371b7a0f020fad9bc0ca` |
| `G_P05_baseline_s64.json` | `9481bddc253c85dbd73d153a92cc07ffefe953a62b44a6a5920ffcca5b7845b2` |
| `G_P05_baseline_s8.json` | `8152095b705e3f8a5abe0162b640770ff220e5a057f780fd2dbda63f571112c2` |
| `G_P05_baseline_s800.json` | `bb81175abbe1ad892a1b1a7d33f7824e2b728b929325e2eb531d50c945125291` |
| `G_P06_baseline_s1.json` | `0bb4e88ca1033804cf319a94cce77411bdf6c53dd31541f5c74e70e1450ba949` |
| `G_P06_baseline_s2.json` | `b51338910b891aa3db8e7d329daa8f9a4f5b2aed75c51f830300de6a391d25d2` |
| `G_P06_baseline_s64.json` | `b92a6d1b23a9a30597a1004b395e716eb658d3a1fea3da6b5ed32449288752ee` |
| `G_P06_baseline_s8.json` | `5a1175365ae74b8a3f991ea96daad0f7ba62f8729e5357ab4218b8bfb51b796c` |
| `G_P06_baseline_s800.json` | `c1253dac089b3258ac97b08f256173c4d811acd9190d1d84bb70dea6a67c65a1` |
| `G_P07_baseline_s1.json` | `de49f725e98b56bc785d90e3513fc6e78f393f86503c8ca82120d84737be9e3f` |
| `G_P07_baseline_s2.json` | `522d40a71522f53142751b44fd579703250c1e275fad594071f16f6c8cd3b285` |
| `G_P07_baseline_s64.json` | `dfbf716d9d9e125a9c4100560d60a82c67599c4ca43d890e8e8f357c2923d506` |
| `G_P07_baseline_s8.json` | `d8fb9d5c917f27c3b0db8668739f69fd8fac5181b8de0d0bcfb7412cfcb54a92` |
| `G_P07_baseline_s800.json` | `e72f8d0c310851fed462dcdaa46bddc24fbd6d044a67fbd61ba4538d30a16213` |
| `G_P08_baseline_s1.json` | `d8e16841ca6a3c5070e92bc7ba64335a49b82951d21500c9c293a05a9ce993ba` |
| `G_P08_baseline_s2.json` | `ee13ffabf9abf0953379ce3ac7ec912c7f4d91e0f3c053952160e11edcc015d8` |
| `G_P08_baseline_s64.json` | `e6301a6cdce739515b42209df7e5332d8b0d2011450957b38c4bda7c66362585` |
| `G_P08_baseline_s8.json` | `bedd6eeefdb1c13739d0150f26037d65b6146db534736cb39596561e205ef3ea` |
| `G_P08_baseline_s800.json` | `3004ca5584b1b781f525b9ac5ed7679ccf8ab2d2f1129de3373fe743fb014fcd` |
| `G_P09_baseline_s1.json` | `89b53c764a0257a827dd1567f4c1be4518149ee4266b0a703fb1fefebe43f540` |
| `G_P09_baseline_s2.json` | `bfc1baf1c6aee545e9368bdd24783ea9adc3eb775d5974b8e166af72f04fa4f4` |
| `G_P09_baseline_s64.json` | `6bdb559a60273fe7259eac6f81f6c64e9114a7800b75dd06bb7b1a5fafe426de` |
| `G_P09_baseline_s8.json` | `b2b3d2c284b9ac57f8e79029b30ab0a0ba5f6da1f4b1fdf26d7e28242b634a9a` |
| `G_P09_baseline_s800.json` | `54c079be59403fc5e87ee40d3aca5957f44c7ed71ea6c9b67a3537738e7ad53b` |
| `G_P10_baseline_s1.json` | `44daa90c7a1b9d223f916a193983931121b6a1b1df34cd35b6ad2ac70c5e20b0` |
| `G_P10_baseline_s2.json` | `828c5d2354ac2e31613ca6cd981132258bd8193f92f778c09500cdea9934089e` |
| `G_P10_baseline_s64.json` | `ef8b2be22df0c6649ff07742162f0d07769e3de2d6c597019bb0109c06ba0859` |
| `G_P10_baseline_s8.json` | `98c3c73868a3b535420de7c31b4ec3aef7cf9f7ea6ce621316659cbb8bccf045` |
| `G_P10_baseline_s800.json` | `fc69e6b27ae05678fd51fa626fe2328ee9908f60d0179610883d985e5b19ad9d` |
| `G_P11_baseline_s1.json` | `07ff7dc763818023be623da6740a9fc0380981150f1b670378d711906fb28983` |
| `G_P11_baseline_s2.json` | `2b583a7ccf16340b60681c9f43cc0138609341baf617eed0d30b2fb2da864ff4` |
| `G_P11_baseline_s64.json` | `3e328e22bb11cbbd3fe40471b51fb6736657eaaa8590691b29b8274d5995eb2d` |
| `G_P11_baseline_s8.json` | `774f3562bb01b71d88069330a9c44c9cce0f68a0e98f7597952edd57737c58aa` |
| `G_P11_baseline_s800.json` | `8f6c6c1111682cf34d1fbbe092884246e21b8ea69772524768fd0d603458b78e` |
| `G_P11_candidate_s1.json` | `6ba32ce091e3c2a5a6cc3410ab6bddc0e90e4528cc6a23c3a9e785ed2c89382a` |
| `G_P11_candidate_s2.json` | `4a7db619ac5bd73eb4c34844601e24c5b7960759cc014a16437cda32b5925649` |
| `G_P11_candidate_s64.json` | `c4d2f2e1c4efde3f9955944897dd2bdb2f2c5017ac36718c4c37da76d5f1c0df` |
| `G_P11_candidate_s8.json` | `fceaba79954ac300b4aa7ef3c1ab9532428e516fd4a2baf47d46fb3daced94e5` |
| `G_P11_candidate_s800.json` | `cda750d9c05fb6f12ab1d1faf3fbc573cd6fe1ee3f80052ef70d027e9113494c` |
| `G_P12_baseline_s1.json` | `eee5e315a2ab1147437e3a5af9ad3af20e1aff2b8596b2ab7d7de575458a9861` |
| `G_P12_baseline_s2.json` | `f3371441aacc80a0e932f8de2d644eeab9423642329cd2d35bbb146cd540144f` |
| `G_P12_baseline_s64.json` | `f02d7ea08f36c86c96e2bce6f1b0d50ecad688dbf4f62c82a4e653134821e8c1` |
| `G_P12_baseline_s8.json` | `2a5e4e378e0627737ee0865f353d59984a6194e768f03702d82613a0b37d502f` |
| `G_P12_baseline_s800.json` | `7ab3f34d268619ae1366a8cb8415f3f694db0aa4a199b6b063e29e102f2f86fd` |
| `G_P13_baseline_s1.json` | `1528836c5927290919f8bc4e494a3da02c91581f9603eef7d04b0290a165eec7` |
| `G_P13_baseline_s2.json` | `2f6284f12d571e3412d6eddf0e7df60fc6c22f291350595689f4b1c1048aad27` |
| `G_P13_baseline_s64.json` | `9ed768f19a4bd9df9def7c75ef0512cc75da8ffeb8811f89e5fdbae3484fd8a7` |
| `G_P13_baseline_s8.json` | `1035316e4b93f072a9a76e4bd95d46752d253128e9148f50381f8c66f9a5a8f5` |
| `G_P13_baseline_s800.json` | `cc54a482129361ee9e6060754231848eb0493ec48ade009bd5de5f373cd025a2` |
| `G_P14_baseline_s1.json` | `64ef8691ac3f7f783799b333cb337666a032a17951045b1e7fc3470f5835206a` |
| `G_P14_baseline_s2.json` | `4b95c378381e7f2b149b18e7b0b7470440fee65f669d8883f4d4091b6545c4e8` |
| `G_P14_baseline_s64.json` | `aaf6b714395c3755fc856bce5f24c85c6965f0befc2a14066cf6195f484f9e4a` |
| `G_P14_baseline_s8.json` | `50f32c38ecaf429fd9a639d458ebe869c96e02615d49a7dc1850f82f6aeea917` |
| `G_P14_baseline_s800.json` | `622befad1d14c1dda18d4b41ee23fba6ec808d845f8175d55b165ffd3685ee36` |
| `G_P15_baseline_s1.json` | `e5dea1fb1e5b747bba7a420a5de3f01aa952d870245d3896ded84849f15d5dc8` |
| `G_P15_baseline_s2.json` | `b0405c9acfcb95eb165c0b15f0a822181709a743c8b37a94845a3ef1a528bc8c` |
| `G_P15_baseline_s64.json` | `ff7ebdc82530d65cde76d029c7c0b7665f891fa1621927fb2082773115fe375a` |
| `G_P15_baseline_s8.json` | `d679714d6fdda7fd06317af31b42f75e4d213fe6529cb2c6c450c1b08233e996` |
| `G_P15_baseline_s800.json` | `5f507b2cac2002acbf944c91638c6fcc70ecb3fa9b8ef8f87117720764223e8b` |
| `G_P16_baseline_s1.json` | `bda244b1a045edffd7722b5b31a1896c561b84868233f61af9a04f639bc69c06` |
| `G_P16_baseline_s2.json` | `310076a6bd65a17940fee9635f5b3d538ee1e66f40480f11546888be3a4af1b7` |
| `G_P16_baseline_s64.json` | `ccfe1f1672845c29358bfd26e87df3f254e368b0f61c62d10272270090ebc168` |
| `G_P16_baseline_s8.json` | `43a57267f7579eba59c8e29c3d89ed61ef36c192c6045c6426438238146c8642` |
| `G_P16_baseline_s800.json` | `61ee8553f2f4ab4c918396a87de9fa460477ef062eea45c07947260e035300d2` |
