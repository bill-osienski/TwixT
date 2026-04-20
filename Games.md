============================================================
Iteration 21/200
  Curriculum: active_size=20, max_moves=320
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 2848 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 3193 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 3444 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 3629 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 3813 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1330 positions
  Results: Red=15, Black=10, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 3813
  Balance: black=40.0%, red=60.0%, draw=0.0% (decisive=25/25, window=2/20)
  Backups: 199500, Leaf evals: 129279, NN batches: 11277
  Avg batch: 11.5, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=6044, stall=3979, tail=1254
  Avg plies: 53.2

  Sanity (1330 positions):
    z: mean=0.015, std=1.000, [+/0/-]=675/0/655
    z by to_move: red=0.225 (+/0/-=409/0/259), black=-0.196 (+/0/-=266/0/396)
    v: mean=-0.134, std=0.771, range=[-0.99,1.00], mse=0.5747, sign_agree=80.5%
    v pretanh: range=[-2.86,3.27], p99=3.11, frac_sat=0.000, zv_corr=0.519 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.6976
  Step 40/160, Loss: 3.5763
  Step 60/160, Loss: 3.5533
  Step 80/160, Loss: 3.5310
  Step 100/160, Loss: 3.5204
  Step 120/160, Loss: 3.4885
  Step 140/160, Loss: 3.4786
  Step 160/160, Loss: 3.4614
  Average loss: 3.4614 (policy=3.2553, value=0.0986, l2=0.1815)
  State: size=20 sims=150 factor=1.00 frozen=False promo_streak=1 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0021.safetensors
  Curriculum: active_size=20, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=305.0s, selfplay=249.7s, train=55.3s

============================================================
Iteration 22/200
  Curriculum: active_size=20, max_moves=320
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 4157 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 4615 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 4764 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 4982 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 5228 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1415 positions
  Results: Red=11, Black=14, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 5228
  Balance: black=56.0%, red=44.0%, draw=0.0% (decisive=25/25, window=3/20)
  Backups: 212250, Leaf evals: 155177, NN batches: 12401
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=7846, stall=3204, tail=1351
  Avg plies: 56.6

  Sanity (1415 positions):
    z: mean=0.012, std=1.000, [+/0/-]=716/0/699
    z by to_move: red=-0.018 (+/0/-=347/0/360), black=0.042 (+/0/-=369/0/339)
    v: mean=0.053, std=0.854, range=[-1.00,1.00], mse=0.4478, sign_agree=86.3%
    v pretanh: range=[-3.34,5.11], p99=4.81, frac_sat=0.039, zv_corr=0.642 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.6157
  Step 40/160, Loss: 3.6175
  Step 60/160, Loss: 3.5852
  Step 80/160, Loss: 3.5597
  Step 100/160, Loss: 3.5474
  Step 120/160, Loss: 3.5285
  Step 140/160, Loss: 3.5253
  Step 160/160, Loss: 3.4974
  Average loss: 3.4974 (policy=3.3071, value=0.1077, l2=0.1633)
  State: size=20 sims=150 factor=1.00 frozen=False promo_streak=2 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0022.safetensors
  Curriculum: active_size=20, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=554.4s, selfplay=499.5s, train=54.9s

============================================================
Iteration 23/200
  Curriculum: active_size=20, max_moves=320
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 5341 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 5448 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 5580 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 5765 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 5914 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 686 positions
  Results: Red=16, Black=9, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 5914
  Balance: black=36.0%, red=64.0%, draw=0.0% (decisive=25/25, window=4/20)
  Backups: 102900, Leaf evals: 75434, NN batches: 6167
  Avg batch: 12.2, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=4069, stall=1452, tail=646
  Avg plies: 27.4

  Sanity (686 positions):
    z: mean=0.026, std=1.000, [+/0/-]=352/0/334
    z by to_move: red=0.407 (+/0/-=242/0/102), black=-0.357 (+/0/-=110/0/232)
    v: mean=0.000, std=0.892, range=[-1.00,1.00], mse=0.3197, sign_agree=91.4%
    v pretanh: range=[-3.53,4.80], p99=4.57, frac_sat=0.055, zv_corr=0.738 (n=256)
    TRIPWIRE: Value head saturating (p99=4.57, sat=0.055)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4992
  Step 40/160, Loss: 3.4716
  Step 60/160, Loss: 3.4326
  Step 80/160, Loss: 3.4041
  Step 100/160, Loss: 3.4155
  Step 120/160, Loss: 3.3876
  Step 140/160, Loss: 3.3809
  Step 160/160, Loss: 3.3752
  Average loss: 3.3752 (policy=3.2025, value=0.0873, l2=0.1509)
  State: size=20 sims=150 factor=1.00 frozen=False promo_streak=3 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0023.safetensors
  Curriculum: active_size=20, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=169.7s, selfplay=114.7s, train=55.0s

============================================================
Iteration 24/200
  Curriculum: active_size=20, max_moves=320
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 6060 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 6188 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 6424 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 6649 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 6760 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 846 positions
  Results: Red=15, Black=10, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 6760
  Balance: black=40.0%, red=60.0%, draw=0.0% (decisive=25/25, window=5/20)
  Backups: 126900, Leaf evals: 75804, NN batches: 7160
  Avg batch: 10.6, Avg waiters: 1.6, Max waiters: 17
  Flushes: full=3516, stall=2834, tail=810
  Avg plies: 33.8

  Sanity (846 positions):
    z: mean=0.021, std=1.000, [+/0/-]=432/0/414
    z by to_move: red=0.154 (+/0/-=244/0/179), black=-0.111 (+/0/-=188/0/235)
    v: mean=-0.300, std=0.792, range=[-1.00,1.00], mse=1.0557, sign_agree=68.0%
    v pretanh: range=[-3.61,4.38], p99=4.07, frac_sat=0.020, zv_corr=0.330 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5150
  Step 40/160, Loss: 3.4228
  Step 60/160, Loss: 3.4107
  Step 80/160, Loss: 3.4087
  Step 100/160, Loss: 3.3948
  Step 120/160, Loss: 3.3869
  Step 140/160, Loss: 3.3900
  Step 160/160, Loss: 3.3824
  Average loss: 3.3824 (policy=3.2100, value=0.1181, l2=0.1429)
  State: size=20 sims=150 factor=1.00 frozen=False promo_streak=4 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0024.safetensors
  Curriculum: active_size=20, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=198.6s, selfplay=143.4s, train=55.2s

============================================================
Iteration 25/200
  Curriculum: active_size=20, max_moves=320
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 6898 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 7050 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 7239 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 7364 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 7488 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 728 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 7488
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=6/20)
  Backups: 109200, Leaf evals: 80271, NN batches: 6536
  Avg batch: 12.3, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=4302, stall=1539, tail=695
  Avg plies: 29.1

  Sanity (728 positions):
    z: mean=0.022, std=1.000, [+/0/-]=372/0/356
    z by to_move: red=0.656 (+/0/-=303/0/63), black=-0.619 (+/0/-=69/0/293)
    v: mean=-0.092, std=0.880, range=[-1.00,1.00], mse=0.3183, sign_agree=88.7%
    v pretanh: range=[-3.93,5.02], p99=4.79, frac_sat=0.090, zv_corr=0.732 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3280
  Step 40/160, Loss: 3.2811
  Step 60/160, Loss: 3.2670
  Step 80/160, Loss: 3.2797
  Step 100/160, Loss: 3.2938
  Step 120/160, Loss: 3.2813
  Step 140/160, Loss: 3.2828
  Step 160/160, Loss: 3.2876
  Average loss: 3.2876 (policy=3.1264, value=0.0978, l2=0.1367)

*** CURRICULUM PROMOTED: active_size=24 ***
    Metrics: timeout=0.0%, plies_ratio=0.09
    Next iter: max_moves=420, base_sims=150
  State: size=20 sims=150 factor=1.00 frozen=False promo_streak=0 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0025.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=182.8s, selfplay=127.9s, train=54.9s
  Size changed: 20 -> 24, streaks reset

============================================================
Iteration 26/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 7725 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 7927 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 8090 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 8263 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 8428 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 940 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 8428
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=7/20)
  Backups: 141000, Leaf evals: 113442, NN batches: 8681
  Avg batch: 13.1, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6336, stall=1440, tail=905
  Avg plies: 37.6

  Sanity (940 positions):
    z: mean=0.023, std=1.000, [+/0/-]=481/0/459
    z by to_move: red=0.353 (+/0/-=320/0/153), black=-0.310 (+/0/-=161/0/306)
    v: mean=-0.055, std=0.876, range=[-1.00,1.00], mse=0.2148, sign_agree=93.8%
    v pretanh: range=[-3.57,3.79], p99=3.55, frac_sat=0.000, zv_corr=0.778 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.7967
  Step 40/160, Loss: 3.6746
  Step 60/160, Loss: 3.6166
  Step 80/160, Loss: 3.5917
  Step 100/160, Loss: 3.5585
  Step 120/160, Loss: 3.5513
  Step 140/160, Loss: 3.5433
  Step 160/160, Loss: 3.5375
  Average loss: 3.5375 (policy=3.3948, value=0.0284, l2=0.1356)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=1 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0026.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=253.8s, selfplay=198.4s, train=55.5s

============================================================
Iteration 27/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 8577 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 8792 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 8964 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 9138 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 9296 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 868 positions
  Results: Red=16, Black=9, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 9296
  Balance: black=36.0%, red=64.0%, draw=0.0% (decisive=25/25, window=8/20)
  Backups: 130200, Leaf evals: 110213, NN batches: 8181
  Avg batch: 13.5, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=6347, stall=998, tail=836
  Avg plies: 34.7

  Sanity (868 positions):
    z: mean=0.025, std=1.000, [+/0/-]=445/0/423
    z by to_move: red=0.358 (+/0/-=296/0/140), black=-0.310 (+/0/-=149/0/283)
    v: mean=-0.030, std=0.969, range=[-1.00,1.00], mse=0.3590, sign_agree=89.8%
    v pretanh: range=[-5.29,6.26], p99=5.87, frac_sat=0.531, zv_corr=0.790 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5061
  Step 40/160, Loss: 3.4171
  Step 60/160, Loss: 3.4003
  Step 80/160, Loss: 3.4042
  Step 100/160, Loss: 3.4166
  Step 120/160, Loss: 3.4217
  Step 140/160, Loss: 3.4243
  Step 160/160, Loss: 3.4255
  Average loss: 3.4255 (policy=3.2823, value=0.0635, l2=0.1274)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=2 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0027.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=241.1s, selfplay=185.6s, train=55.5s

============================================================
Iteration 28/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 9480 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 9643 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 9799 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 9937 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 10165 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 869 positions
  Results: Red=24, Black=1, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 10165
  Balance: black=4.0%, red=96.0%, draw=0.0% (decisive=25/25, window=9/20)
  Backups: 130350, Leaf evals: 111193, NN batches: 8206
  Avg batch: 13.6, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=6423, stall=941, tail=842
  Avg plies: 34.8

  Sanity (869 positions):
    z: mean=0.022, std=1.000, [+/0/-]=444/0/425
    z by to_move: red=0.788 (+/0/-=396/0/47), black=-0.775 (+/0/-=48/0/378)
    v: mean=0.083, std=0.956, range=[-1.00,1.00], mse=0.3412, sign_agree=89.8%
    v pretanh: range=[-4.50,5.28], p99=4.82, frac_sat=0.289, zv_corr=0.790 (n=256)
    TRIPWIRE: Value head saturating (p99=4.82, sat=0.289)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5430
  Step 40/160, Loss: 3.4908
  Step 60/160, Loss: 3.4713
  Step 80/160, Loss: 3.4565
  Step 100/160, Loss: 3.4369
  Step 120/160, Loss: 3.4429
  Step 140/160, Loss: 3.4243
  Step 160/160, Loss: 3.4253
  Average loss: 3.4253 (policy=3.2918, value=0.0507, l2=0.1208)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=3 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0028.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=251.6s, selfplay=195.9s, train=55.7s

============================================================
Iteration 29/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 10314 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 10474 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 10608 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 10738 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 10869 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 704 positions
  Results: Red=23, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 10869
  Balance: black=8.0%, red=92.0%, draw=0.0% (decisive=25/25, window=10/20)
  Backups: 105600, Leaf evals: 91027, NN batches: 6688
  Avg batch: 13.6, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=5331, stall=672, tail=685
  Avg plies: 28.2

  Sanity (704 positions):
    z: mean=0.023, std=1.000, [+/0/-]=360/0/344
    z by to_move: red=0.866 (+/0/-=334/0/24), black=-0.850 (+/0/-=26/0/320)
    v: mean=0.006, std=0.953, range=[-1.00,1.00], mse=0.1542, sign_agree=94.9%
    v pretanh: range=[-4.08,4.68], p99=4.49, frac_sat=0.098, zv_corr=0.877 (n=256)
    TRIPWIRE: Value head saturating (p99=4.49, sat=0.098)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3549
  Step 40/160, Loss: 3.3384
  Step 60/160, Loss: 3.3647
  Step 80/160, Loss: 3.3504
  Step 100/160, Loss: 3.3336
  Step 120/160, Loss: 3.3303
  Step 140/160, Loss: 3.3177
  Step 160/160, Loss: 3.3140
  Average loss: 3.3140 (policy=3.1873, value=0.0467, l2=0.1150)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=4 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0029.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=205.6s, selfplay=150.0s, train=55.6s

============================================================
Iteration 30/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 11086 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 11396 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 11552 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 11710 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 11990 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1121 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 11990
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=11/20)
  ⚠️  BALANCE WARNING: black decisive win rate 12.0% (<25%) for 3 consecutive iters | decisive=25/25 (100%), draw=0%
  Backups: 168150, Leaf evals: 128645, NN batches: 10358
  Avg batch: 12.4, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=7135, stall=2143, tail=1080
  Avg plies: 44.8

  Sanity (1121 positions):
    z: mean=0.013, std=1.000, [+/0/-]=568/0/553
    z by to_move: red=0.561 (+/0/-=441/0/124), black=-0.543 (+/0/-=127/0/429)
    v: mean=-0.133, std=0.896, range=[-1.00,1.00], mse=0.8184, sign_agree=73.4%
    v pretanh: range=[-4.66,4.41], p99=4.38, frac_sat=0.125, zv_corr=0.501 (n=256)
    TRIPWIRE: Value head saturating (p99=4.38, sat=0.125)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5391
  Step 40/160, Loss: 3.5050
  Step 60/160, Loss: 3.4730
  Step 80/160, Loss: 3.4704
  Step 100/160, Loss: 3.4580
  Step 120/160, Loss: 3.4526
  Step 140/160, Loss: 3.4463
  Step 160/160, Loss: 3.4460
  Average loss: 3.4460 (policy=3.3056, value=0.1110, l2=0.1127)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=5 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0030.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=452.0s, selfplay=396.3s, train=55.7s

============================================================
Iteration 31/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 12231 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 12502 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 12629 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 12912 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 13051 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1061 positions
  Results: Red=14, Black=11, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 13051
  Balance: black=44.0%, red=56.0%, draw=0.0% (decisive=25/25, window=12/20)
  Backups: 159150, Leaf evals: 109619, NN batches: 9539
  Avg batch: 11.5, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=5777, stall=2751, tail=1011
  Avg plies: 42.4

  Sanity (1061 positions):
    z: mean=0.022, std=1.000, [+/0/-]=542/0/519
    z by to_move: red=0.461 (+/0/-=388/0/143), black=-0.419 (+/0/-=154/0/376)
    v: mean=-0.158, std=0.789, range=[-1.00,1.00], mse=0.5659, sign_agree=75.0%
    v pretanh: range=[-3.73,5.08], p99=4.71, frac_sat=0.027, zv_corr=0.541 (n=256)
    TRIPWIRE: Value head saturating (p99=4.71, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3938
  Step 40/160, Loss: 3.3567
  Step 60/160, Loss: 3.3098
  Step 80/160, Loss: 3.3234
  Step 100/160, Loss: 3.3595
  Step 120/160, Loss: 3.3696
  Step 140/160, Loss: 3.3658
  Step 160/160, Loss: 3.3573
  Average loss: 3.3573 (policy=3.2264, value=0.0767, l2=0.1117)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=6 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0031.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=470.3s, selfplay=414.3s, train=56.0s

============================================================
Iteration 32/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 13269 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 13456 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 13596 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 13764 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 13908 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 857 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 13908
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=13/20)
  Backups: 128550, Leaf evals: 101172, NN batches: 7936
  Avg batch: 12.7, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5670, stall=1428, tail=838
  Avg plies: 34.3

  Sanity (857 positions):
    z: mean=0.020, std=1.000, [+/0/-]=437/0/420
    z by to_move: red=0.797 (+/0/-=390/0/44), black=-0.778 (+/0/-=47/0/376)
    v: mean=-0.006, std=0.912, range=[-1.00,1.00], mse=0.2037, sign_agree=91.8%
    v pretanh: range=[-4.06,4.24], p99=3.99, frac_sat=0.039, zv_corr=0.814 (n=256)
    TRIPWIRE: Value head saturating (p99=3.99, sat=0.039)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2518
  Step 40/160, Loss: 3.3187
  Step 60/160, Loss: 3.3474
  Step 80/160, Loss: 3.3627
  Step 100/160, Loss: 3.3590
  Step 120/160, Loss: 3.3489
  Step 140/160, Loss: 3.3495
  Step 160/160, Loss: 3.3472
  Average loss: 3.3472 (policy=3.2218, value=0.0646, l2=0.1093)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=7 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0032.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=241.4s, selfplay=185.6s, train=55.8s

============================================================
Iteration 33/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 14057 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 14224 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 14433 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 14581 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 14866 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 958 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 14866
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=14/20)
  Backups: 143700, Leaf evals: 118966, NN batches: 8962
  Avg batch: 13.3, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6757, stall=1295, tail=910
  Avg plies: 38.3

  Sanity (958 positions):
    z: mean=0.021, std=1.000, [+/0/-]=489/0/469
    z by to_move: red=0.619 (+/0/-=391/0/92), black=-0.587 (+/0/-=98/0/377)
    v: mean=0.008, std=0.910, range=[-1.00,1.00], mse=0.6679, sign_agree=78.5%
    v pretanh: range=[-4.65,4.21], p99=4.31, frac_sat=0.098, zv_corr=0.580 (n=256)
    TRIPWIRE: Value head saturating (p99=4.31, sat=0.098)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3521
  Step 40/160, Loss: 3.3370
  Step 60/160, Loss: 3.3482
  Step 80/160, Loss: 3.3543
  Step 100/160, Loss: 3.3663
  Step 120/160, Loss: 3.3652
  Step 140/160, Loss: 3.3741
  Step 160/160, Loss: 3.3758
  Average loss: 3.3758 (policy=3.2480, value=0.0824, l2=0.1073)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=8 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0033.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=317.7s, selfplay=261.6s, train=56.0s

============================================================
Iteration 34/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 15203 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 15362 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 15511 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 15649 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 15867 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1001 positions
  Results: Red=24, Black=1, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 15867
  Balance: black=4.0%, red=96.0%, draw=0.0% (decisive=25/25, window=15/20)
  ⚠️  BALANCE WARNING: black decisive win rate 4.0% (<25%) for 3 consecutive iters | decisive=25/25 (100%), draw=0%
  Backups: 150150, Leaf evals: 118920, NN batches: 9223
  Avg batch: 12.9, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6624, stall=1662, tail=937
  Avg plies: 40.0

  Sanity (1001 positions):
    z: mean=0.015, std=1.000, [+/0/-]=508/0/493
    z by to_move: red=0.913 (+/0/-=485/0/22), black=-0.907 (+/0/-=23/0/471)
    v: mean=0.040, std=0.898, range=[-1.00,1.00], mse=0.5174, sign_agree=83.2%
    v pretanh: range=[-4.04,3.83], p99=3.78, frac_sat=0.012, zv_corr=0.645 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4106
  Step 40/160, Loss: 3.4105
  Step 60/160, Loss: 3.4207
  Step 80/160, Loss: 3.4046
  Step 100/160, Loss: 3.4252
  Step 120/160, Loss: 3.4325
  Step 140/160, Loss: 3.4201
  Step 160/160, Loss: 3.4115
  Average loss: 3.4115 (policy=3.2843, value=0.0833, l2=0.1063)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=9 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0034.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=353.4s, selfplay=297.2s, train=56.2s

============================================================
Iteration 35/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 16070 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 16258 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 16416 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 16573 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 16727 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 860 positions
  Results: Red=25, Black=0, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 16727
  Balance: black=0.0%, red=100.0%, draw=0.0% (decisive=25/25, window=16/20)
  ⚠️  BALANCE WARNING: black decisive win rate 0.0% (<25%) for 4 consecutive iters | decisive=25/25 (100%), draw=0%
  Backups: 129000, Leaf evals: 106844, NN batches: 8106
  Avg batch: 13.2, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6150, stall=1136, tail=820
  Avg plies: 34.4

  Sanity (860 positions):
    z: mean=0.014, std=1.000, [+/0/-]=436/0/424
    z by to_move: red=1.000 (+/0/-=436/0/0), black=-1.000 (+/0/-=0/0/424)
    v: mean=0.014, std=0.927, range=[-1.00,1.00], mse=0.1162, sign_agree=94.1%
    v pretanh: range=[-4.01,4.64], p99=4.00, frac_sat=0.035, zv_corr=0.872 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2395
  Step 40/160, Loss: 3.2867
  Step 60/160, Loss: 3.3212
  Step 80/160, Loss: 3.3185
  Step 100/160, Loss: 3.3082
  Step 120/160, Loss: 3.3333
  Step 140/160, Loss: 3.3387
  Step 160/160, Loss: 3.3426
  Average loss: 3.3426 (policy=3.2184, value=0.0761, l2=0.1052)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=10 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0035.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=241.0s, selfplay=185.2s, train=55.8s

============================================================
Iteration 36/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 16917 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 17060 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 17241 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 17386 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 17640 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 913 positions
  Results: Red=21, Black=4, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 17640
  Balance: black=16.0%, red=84.0%, draw=0.0% (decisive=25/25, window=17/20)
  ⚠️  BALANCE WARNING: black decisive win rate 16.0% (<25%) for 5 consecutive iters | decisive=25/25 (100%), draw=0%
  Backups: 136950, Leaf evals: 113275, NN batches: 8577
  Avg batch: 13.2, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6491, stall=1203, tail=883
  Avg plies: 36.5

  Sanity (913 positions):
    z: mean=0.023, std=1.000, [+/0/-]=467/0/446
    z by to_move: red=0.767 (+/0/-=409/0/54), black=-0.742 (+/0/-=58/0/392)
    v: mean=0.035, std=0.946, range=[-1.00,1.00], mse=0.0994, sign_agree=96.9%
    v pretanh: range=[-4.77,4.69], p99=4.63, frac_sat=0.137, zv_corr=0.898 (n=256)
    TRIPWIRE: Value head saturating (p99=4.63, sat=0.137)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3083
  Step 40/160, Loss: 3.3008
  Step 60/160, Loss: 3.3255
  Step 80/160, Loss: 3.3459
  Step 100/160, Loss: 3.3368
  Step 120/160, Loss: 3.3603
  Step 140/160, Loss: 3.3701
  Step 160/160, Loss: 3.3850
  Average loss: 3.3850 (policy=3.2662, value=0.0623, l2=0.1032)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=11 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0036.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=256.2s, selfplay=200.5s, train=55.8s

============================================================
Iteration 37/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 18237 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 18458 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 18717 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 19028 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 19185 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1545 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 19185
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=18/20)
  ⚠️  BALANCE WARNING: black decisive win rate 20.0% (<25%) for 6 consecutive iters | decisive=25/25 (100%), draw=0%
  Backups: 231750, Leaf evals: 166743, NN batches: 13726
  Avg batch: 12.1, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=8594, stall=3652, tail=1480
  Avg plies: 61.8

  Sanity (1545 positions):
    z: mean=0.008, std=1.000, [+/0/-]=779/0/766
    z by to_move: red=0.577 (+/0/-=611/0/164), black=-0.564 (+/0/-=168/0/602)
    v: mean=-0.212, std=0.785, range=[-1.00,1.00], mse=0.9135, sign_agree=68.8%
    v pretanh: range=[-4.78,4.12], p99=4.33, frac_sat=0.066, zv_corr=0.374 (n=256)
    TRIPWIRE: Value head saturating (p99=4.33, sat=0.066)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4038
  Step 40/160, Loss: 3.3805
  Step 60/160, Loss: 3.4174
  Step 80/160, Loss: 3.3881
  Step 100/160, Loss: 3.3884
  Step 120/160, Loss: 3.3955
  Step 140/160, Loss: 3.4043
  Step 160/160, Loss: 3.3979
  Average loss: 3.3979 (policy=3.2694, value=0.1038, l2=0.1025)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=12 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0037.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=1027.2s, selfplay=971.2s, train=56.0s

============================================================
Iteration 38/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 19358 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 19560 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 19735 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 19991 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 20166 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 981 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 20166
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=19/20)
  ⚠️  BALANCE WARNING: black decisive win rate 12.0% (<25%) for 7 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 147150, Leaf evals: 107136, NN batches: 8854
  Avg batch: 12.1, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5749, stall=2169, tail=936
  Avg plies: 39.2

  Sanity (981 positions):
    z: mean=0.013, std=1.000, [+/0/-]=497/0/484
    z by to_move: red=0.704 (+/0/-=421/0/73), black=-0.688 (+/0/-=76/0/411)
    v: mean=-0.202, std=0.858, range=[-1.00,1.00], mse=0.6346, sign_agree=80.1%
    v pretanh: range=[-4.04,3.82], p99=3.83, frac_sat=0.027, zv_corr=0.571 (n=256)
    TRIPWIRE: Value head saturating (p99=3.83, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3966
  Step 40/160, Loss: 3.3762
  Step 60/160, Loss: 3.4140
  Step 80/160, Loss: 3.4111
  Step 100/160, Loss: 3.4258
  Step 120/160, Loss: 3.4503
  Step 140/160, Loss: 3.4276
  Step 160/160, Loss: 3.4318
  Average loss: 3.4318 (policy=3.3008, value=0.1146, l2=0.1024)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=13 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0038.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=257.5s, selfplay=201.3s, train=56.1s

============================================================
Iteration 39/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 20301 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 20559 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 20700 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 20871 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 21026 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 860 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 21026
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 24.0% (<25%) for 8 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 129000, Leaf evals: 95344, NN batches: 7789
  Avg batch: 12.2, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5154, stall=1813, tail=822
  Avg plies: 34.4

  Sanity (860 positions):
    z: mean=0.019, std=1.000, [+/0/-]=438/0/422
    z by to_move: red=0.224 (+/0/-=265/0/168), black=-0.190 (+/0/-=173/0/254)
    v: mean=-0.069, std=0.887, range=[-1.00,1.00], mse=0.7970, sign_agree=75.8%
    v pretanh: range=[-4.20,4.47], p99=4.14, frac_sat=0.043, zv_corr=0.497 (n=256)
    TRIPWIRE: Value head saturating (p99=4.14, sat=0.043)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3264
  Step 40/160, Loss: 3.3286
  Step 60/160, Loss: 3.3875
  Step 80/160, Loss: 3.3733
  Step 100/160, Loss: 3.3848
  Step 120/160, Loss: 3.3908
  Step 140/160, Loss: 3.4062
  Step 160/160, Loss: 3.3953
  Average loss: 3.3953 (policy=3.2646, value=0.1142, l2=0.1022)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=14 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0039.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=236.6s, selfplay=180.8s, train=55.8s

============================================================
Iteration 40/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 21407 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 21551 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 21679 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 21859 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 22162 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1136 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 22162
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 170400, Leaf evals: 114945, NN batches: 9923
  Avg batch: 11.6, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=5765, stall=3085, tail=1073
  Avg plies: 45.4

  Sanity (1136 positions):
    z: mean=0.014, std=1.000, [+/0/-]=576/0/560
    z by to_move: red=0.190 (+/0/-=338/0/230), black=-0.162 (+/0/-=238/0/330)
    v: mean=-0.225, std=0.730, range=[-1.00,1.00], mse=0.8119, sign_agree=66.8%
    v pretanh: range=[-4.25,3.74], p99=3.70, frac_sat=0.008, zv_corr=0.386 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3279
  Step 40/160, Loss: 3.3500
  Step 60/160, Loss: 3.3695
  Step 80/160, Loss: 3.3616
  Step 100/160, Loss: 3.3551
  Step 120/160, Loss: 3.3360
  Step 140/160, Loss: 3.3333
  Step 160/160, Loss: 3.3363
  Average loss: 3.3363 (policy=3.2061, value=0.1138, l2=0.1018)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=15 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0040.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=424.2s, selfplay=368.1s, train=56.1s

============================================================
Iteration 41/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 22335 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 22475 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 22625 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 22755 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 22916 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 754 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 22916
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 113100, Leaf evals: 90810, NN batches: 6996
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5150, stall=1123, tail=723
  Avg plies: 30.2

  Sanity (754 positions):
    z: mean=0.021, std=1.000, [+/0/-]=385/0/369
    z by to_move: red=0.655 (+/0/-=317/0/66), black=-0.633 (+/0/-=68/0/303)
    v: mean=-0.045, std=0.934, range=[-1.00,1.00], mse=0.2138, sign_agree=93.4%
    v pretanh: range=[-4.06,4.51], p99=4.00, frac_sat=0.043, zv_corr=0.830 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3470
  Step 40/160, Loss: 3.3459
  Step 60/160, Loss: 3.3392
  Step 80/160, Loss: 3.3640
  Step 100/160, Loss: 3.3412
  Step 120/160, Loss: 3.3431
  Step 140/160, Loss: 3.3398
  Step 160/160, Loss: 3.3459
  Average loss: 3.3459 (policy=3.2205, value=0.0958, l2=0.1014)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=16 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0041.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=213.7s, selfplay=157.9s, train=55.8s

============================================================
Iteration 42/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 23097 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 23269 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 23428 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 23652 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 23891 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 975 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 23891
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 146250, Leaf evals: 96589, NN batches: 8620
  Avg batch: 11.2, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=4951, stall=2739, tail=930
  Avg plies: 39.0

  Sanity (975 positions):
    z: mean=0.015, std=1.000, [+/0/-]=495/0/480
    z by to_move: red=0.276 (+/0/-=314/0/178), black=-0.251 (+/0/-=181/0/302)
    v: mean=-0.300, std=0.805, range=[-1.00,1.00], mse=0.6648, sign_agree=77.0%
    v pretanh: range=[-4.09,3.67], p99=3.99, frac_sat=0.020, zv_corr=0.537 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3236
  Step 40/160, Loss: 3.2832
  Step 60/160, Loss: 3.3283
  Step 80/160, Loss: 3.3485
  Step 100/160, Loss: 3.3558
  Step 120/160, Loss: 3.3670
  Step 140/160, Loss: 3.3633
  Step 160/160, Loss: 3.3566
  Average loss: 3.3566 (policy=3.2306, value=0.1019, l2=0.1006)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=17 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0042.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=274.9s, selfplay=218.8s, train=56.1s

============================================================
Iteration 43/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 24036 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 24220 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 24600 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 24776 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 24931 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1040 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 24931
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 24.0% (<25%) for 3 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 156000, Leaf evals: 113740, NN batches: 9334
  Avg batch: 12.2, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6014, stall=2326, tail=994
  Avg plies: 41.6

  Sanity (1040 positions):
    z: mean=0.017, std=1.000, [+/0/-]=529/0/511
    z by to_move: red=0.587 (+/0/-=415/0/108), black=-0.559 (+/0/-=114/0/403)
    v: mean=-0.091, std=0.841, range=[-1.00,1.00], mse=0.6455, sign_agree=79.7%
    v pretanh: range=[-4.20,5.30], p99=4.77, frac_sat=0.113, zv_corr=0.535 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3358
  Step 40/160, Loss: 3.3989
  Step 60/160, Loss: 3.3553
  Step 80/160, Loss: 3.3539
  Step 100/160, Loss: 3.3389
  Step 120/160, Loss: 3.3255
  Step 140/160, Loss: 3.3181
  Step 160/160, Loss: 3.3228
  Average loss: 3.3228 (policy=3.1952, value=0.1078, l2=0.1006)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=18 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0043.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=399.8s, selfplay=343.7s, train=56.1s

============================================================
Iteration 44/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 25081 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 25237 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 25402 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 25533 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 25826 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 895 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 25826
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 12.0% (<25%) for 4 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 134250, Leaf evals: 100442, NN batches: 8138
  Avg batch: 12.3, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5458, stall=1821, tail=859
  Avg plies: 35.8

  Sanity (895 positions):
    z: mean=0.017, std=1.000, [+/0/-]=455/0/440
    z by to_move: red=0.457 (+/0/-=330/0/123), black=-0.434 (+/0/-=125/0/317)
    v: mean=-0.079, std=0.875, range=[-1.00,1.00], mse=0.2343, sign_agree=89.8%
    v pretanh: range=[-4.34,4.15], p99=4.19, frac_sat=0.074, zv_corr=0.769 (n=256)
    TRIPWIRE: Value head saturating (p99=4.19, sat=0.074)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2973
  Step 40/160, Loss: 3.2788
  Step 60/160, Loss: 3.2837
  Step 80/160, Loss: 3.3058
  Step 100/160, Loss: 3.3105
  Step 120/160, Loss: 3.3494
  Step 140/160, Loss: 3.3303
  Step 160/160, Loss: 3.3330
  Average loss: 3.3330 (policy=3.2080, value=0.1005, l2=0.0998)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=19 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0044.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=286.6s, selfplay=230.4s, train=56.2s

============================================================
Iteration 45/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 25999 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 26232 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 26496 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 26673 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 26827 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1001 positions
  Results: Red=14, Black=11, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 26827
  Balance: black=44.0%, red=56.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 150150, Leaf evals: 118333, NN batches: 9230
  Avg batch: 12.8, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6566, stall=1711, tail=953
  Avg plies: 40.0

  Sanity (1001 positions):
    z: mean=0.017, std=1.000, [+/0/-]=509/0/492
    z by to_move: red=0.032 (+/0/-=258/0/242), black=0.002 (+/0/-=251/0/250)
    v: mean=0.099, std=0.839, range=[-1.00,1.00], mse=1.0309, sign_agree=66.8%
    v pretanh: range=[-3.32,4.20], p99=3.85, frac_sat=0.012, zv_corr=0.342 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4025
  Step 40/160, Loss: 3.3978
  Step 60/160, Loss: 3.4030
  Step 80/160, Loss: 3.4002
  Step 100/160, Loss: 3.3958
  Step 120/160, Loss: 3.3910
  Step 140/160, Loss: 3.3950
  Step 160/160, Loss: 3.3773
  Average loss: 3.3773 (policy=3.2437, value=0.1366, l2=0.0995)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=20 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0045.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=298.6s, selfplay=242.1s, train=56.5s

============================================================
Iteration 46/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 27063 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 27262 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 27394 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 27603 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 27779 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 952 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 27779
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 142800, Leaf evals: 93932, NN batches: 8343
  Avg batch: 11.3, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=4751, stall=2680, tail=912
  Avg plies: 38.1

  Sanity (952 positions):
    z: mean=0.021, std=1.000, [+/0/-]=486/0/466
    z by to_move: red=0.461 (+/0/-=350/0/129), black=-0.425 (+/0/-=136/0/337)
    v: mean=-0.293, std=0.800, range=[-1.00,1.00], mse=0.5861, sign_agree=78.9%
    v pretanh: range=[-3.80,3.95], p99=3.73, frac_sat=0.004, zv_corr=0.570 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3947
  Step 40/160, Loss: 3.3380
  Step 60/160, Loss: 3.3534
  Step 80/160, Loss: 3.3234
  Step 100/160, Loss: 3.3216
  Step 120/160, Loss: 3.3433
  Step 140/160, Loss: 3.3491
  Step 160/160, Loss: 3.3541
  Average loss: 3.3541 (policy=3.2261, value=0.1137, l2=0.0996)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=21 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0046.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=242.3s, selfplay=186.4s, train=55.9s

============================================================
Iteration 47/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 27955 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 28174 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 28350 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 28495 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 28832 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1053 positions
  Results: Red=15, Black=10, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 28832
  Balance: black=40.0%, red=60.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 157950, Leaf evals: 122666, NN batches: 9699
  Avg batch: 12.6, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6764, stall=1916, tail=1019
  Avg plies: 42.1

  Sanity (1053 positions):
    z: mean=0.018, std=1.000, [+/0/-]=536/0/517
    z by to_move: red=-0.070 (+/0/-=245/0/282), black=0.106 (+/0/-=291/0/235)
    v: mean=0.111, std=0.869, range=[-1.00,1.00], mse=0.7148, sign_agree=77.3%
    v pretanh: range=[-3.79,4.78], p99=4.18, frac_sat=0.039, zv_corr=0.527 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3553
  Step 40/160, Loss: 3.3551
  Step 60/160, Loss: 3.3570
  Step 80/160, Loss: 3.3779
  Step 100/160, Loss: 3.3750
  Step 120/160, Loss: 3.3542
  Step 140/160, Loss: 3.3526
  Step 160/160, Loss: 3.3520
  Average loss: 3.3520 (policy=3.2222, value=0.1217, l2=0.0994)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=22 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0047.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=333.7s, selfplay=277.3s, train=56.4s

============================================================
Iteration 48/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 28970 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 29111 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 29245 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 29390 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 29690 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 858 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 29690
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 128700, Leaf evals: 99484, NN batches: 7872
  Avg batch: 12.6, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5512, stall=1539, tail=821
  Avg plies: 34.3

  Sanity (858 positions):
    z: mean=0.019, std=1.000, [+/0/-]=437/0/421
    z by to_move: red=0.407 (+/0/-=306/0/129), black=-0.381 (+/0/-=131/0/292)
    v: mean=0.022, std=0.902, range=[-1.00,1.00], mse=0.3552, sign_agree=88.3%
    v pretanh: range=[-3.78,5.27], p99=4.28, frac_sat=0.059, zv_corr=0.729 (n=256)
    TRIPWIRE: Value head saturating (p99=4.28, sat=0.059)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4265
  Step 40/160, Loss: 3.3697
  Step 60/160, Loss: 3.4359
  Step 80/160, Loss: 3.4195
  Step 100/160, Loss: 3.4276
  Step 120/160, Loss: 3.4014
  Step 140/160, Loss: 3.4006
  Step 160/160, Loss: 3.4034
  Average loss: 3.4034 (policy=3.2780, value=0.1045, l2=0.0993)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=23 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0048.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=365.6s, selfplay=309.2s, train=56.4s

============================================================
Iteration 49/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 29889 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 30034 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 30185 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 30365 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 30537 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 847 positions
  Results: Red=15, Black=10, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 30537
  Balance: black=40.0%, red=60.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 127050, Leaf evals: 101869, NN batches: 7818
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5688, stall=1310, tail=820
  Avg plies: 33.9

  Sanity (847 positions):
    z: mean=0.022, std=1.000, [+/0/-]=433/0/414
    z by to_move: red=0.052 (+/0/-=223/0/201), black=-0.007 (+/0/-=210/0/213)
    v: mean=0.038, std=0.890, range=[-1.00,1.00], mse=0.3203, sign_agree=88.3%
    v pretanh: range=[-4.05,5.08], p99=4.90, frac_sat=0.141, zv_corr=0.736 (n=256)
    TRIPWIRE: Value head saturating (p99=4.90, sat=0.141)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3855
  Step 40/160, Loss: 3.3771
  Step 60/160, Loss: 3.4008
  Step 80/160, Loss: 3.3766
  Step 100/160, Loss: 3.4008
  Step 120/160, Loss: 3.3870
  Step 140/160, Loss: 3.3774
  Step 160/160, Loss: 3.3787
  Average loss: 3.3787 (policy=3.2531, value=0.1054, l2=0.0992)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=24 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0049.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=235.9s, selfplay=179.6s, train=56.3s

============================================================
Iteration 50/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 30685 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 30862 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 31074 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 31210 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 31404 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 867 positions
  Results: Red=23, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 31404
  Balance: black=8.0%, red=92.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 130050, Leaf evals: 102880, NN batches: 8036
  Avg batch: 12.8, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5758, stall=1430, tail=848
  Avg plies: 34.7

  Sanity (867 positions):
    z: mean=0.015, std=1.000, [+/0/-]=440/0/427
    z by to_move: red=0.768 (+/0/-=388/0/51), black=-0.757 (+/0/-=52/0/376)
    v: mean=-0.064, std=0.922, range=[-1.00,1.00], mse=0.1767, sign_agree=94.1%
    v pretanh: range=[-3.90,4.63], p99=4.24, frac_sat=0.070, zv_corr=0.838 (n=256)
    TRIPWIRE: Value head saturating (p99=4.24, sat=0.070)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2814
  Step 40/160, Loss: 3.3387
  Step 60/160, Loss: 3.3794
  Step 80/160, Loss: 3.3913
  Step 100/160, Loss: 3.3853
  Step 120/160, Loss: 3.3938
  Step 140/160, Loss: 3.3727
  Step 160/160, Loss: 3.3742
  Average loss: 3.3742 (policy=3.2480, value=0.1096, l2=0.0988)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=25 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0050.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=239.3s, selfplay=182.7s, train=56.7s

============================================================
Iteration 51/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 31546 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 31770 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 31958 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 32118 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 32246 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 842 positions
  Results: Red=25, Black=0, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 32246
  Balance: black=0.0%, red=100.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 126300, Leaf evals: 103846, NN batches: 7851
  Avg batch: 13.2, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5899, stall=1142, tail=810
  Avg plies: 33.7

  Sanity (842 positions):
    z: mean=0.010, std=1.000, [+/0/-]=425/0/417
    z by to_move: red=1.000 (+/0/-=425/0/0), black=-1.000 (+/0/-=0/0/417)
    v: mean=-0.040, std=0.902, range=[-1.00,1.00], mse=0.1988, sign_agree=92.6%
    v pretanh: range=[-4.13,4.46], p99=4.27, frac_sat=0.070, zv_corr=0.808 (n=256)
    TRIPWIRE: Value head saturating (p99=4.27, sat=0.070)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3463
  Step 40/160, Loss: 3.3477
  Step 60/160, Loss: 3.3627
  Step 80/160, Loss: 3.3614
  Step 100/160, Loss: 3.3783
  Step 120/160, Loss: 3.3612
  Step 140/160, Loss: 3.3574
  Step 160/160, Loss: 3.3460
  Average loss: 3.3460 (policy=3.2205, value=0.1080, l2=0.0985)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=26 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0051.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=248.4s, selfplay=192.1s, train=56.3s

============================================================
Iteration 52/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 32402 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 32558 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 32728 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 32873 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 33016 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 770 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 33016
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 115500, Leaf evals: 96126, NN batches: 7219
  Avg batch: 13.3, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5532, stall=957, tail=730
  Avg plies: 30.8

  Sanity (770 positions):
    z: mean=0.023, std=1.000, [+/0/-]=394/0/376
    z by to_move: red=0.407 (+/0/-=273/0/115), black=-0.366 (+/0/-=121/0/261)
    v: mean=-0.012, std=0.914, range=[-1.00,1.00], mse=0.3844, sign_agree=87.9%
    v pretanh: range=[-4.52,4.26], p99=4.30, frac_sat=0.062, zv_corr=0.725 (n=256)
    TRIPWIRE: Value head saturating (p99=4.30, sat=0.062)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4171
  Step 40/160, Loss: 3.3830
  Step 60/160, Loss: 3.3935
  Step 80/160, Loss: 3.3649
  Step 100/160, Loss: 3.3687
  Step 120/160, Loss: 3.3727
  Step 140/160, Loss: 3.3762
  Step 160/160, Loss: 3.3750
  Average loss: 3.3750 (policy=3.2516, value=0.1016, l2=0.0980)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=27 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0052.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=222.8s, selfplay=166.2s, train=56.6s

============================================================
Iteration 53/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 33273 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 33409 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 33556 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 33738 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 33901 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 885 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 33901
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 132750, Leaf evals: 106175, NN batches: 8197
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5970, stall=1380, tail=847
  Avg plies: 35.4

  Sanity (885 positions):
    z: mean=0.017, std=1.000, [+/0/-]=450/0/435
    z by to_move: red=0.682 (+/0/-=376/0/71), black=-0.662 (+/0/-=74/0/364)
    v: mean=-0.007, std=0.941, range=[-1.00,1.00], mse=0.2654, sign_agree=91.8%
    v pretanh: range=[-4.57,5.07], p99=4.73, frac_sat=0.105, zv_corr=0.810 (n=256)
    TRIPWIRE: Value head saturating (p99=4.73, sat=0.105)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4377
  Step 40/160, Loss: 3.3974
  Step 60/160, Loss: 3.3961
  Step 80/160, Loss: 3.3804
  Step 100/160, Loss: 3.3646
  Step 120/160, Loss: 3.3643
  Step 140/160, Loss: 3.3627
  Step 160/160, Loss: 3.3734
  Average loss: 3.3734 (policy=3.2501, value=0.1029, l2=0.0976)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=28 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0053.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=256.3s, selfplay=199.9s, train=56.4s

============================================================
Iteration 54/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 34064 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 34196 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 34473 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 34670 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 34796 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 895 positions
  Results: Red=23, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 34796
  Balance: black=8.0%, red=92.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 134250, Leaf evals: 101341, NN batches: 8138
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5526, stall=1746, tail=866
  Avg plies: 35.8

  Sanity (895 positions):
    z: mean=0.015, std=1.000, [+/0/-]=454/0/441
    z by to_move: red=0.792 (+/0/-=406/0/47), black=-0.783 (+/0/-=48/0/394)
    v: mean=-0.034, std=0.876, range=[-1.00,1.00], mse=0.3902, sign_agree=85.2%
    v pretanh: range=[-4.31,5.68], p99=4.78, frac_sat=0.090, zv_corr=0.689 (n=256)
    TRIPWIRE: Value head saturating (p99=4.78, sat=0.090)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3279
  Step 40/160, Loss: 3.3517
  Step 60/160, Loss: 3.3415
  Step 80/160, Loss: 3.3369
  Step 100/160, Loss: 3.3074
  Step 120/160, Loss: 3.3111
  Step 140/160, Loss: 3.3120
  Step 160/160, Loss: 3.3231
  Average loss: 3.3231 (policy=3.1996, value=0.1037, l2=0.0976)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=29 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0054.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=261.8s, selfplay=205.7s, train=56.2s

============================================================
Iteration 55/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 34977 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 35140 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 35505 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 35658 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 35850 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1054 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 35850
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 20.0% (<25%) for 3 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 158100, Leaf evals: 116001, NN batches: 9391
  Avg batch: 12.4, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6094, stall=2289, tail=1008
  Avg plies: 42.2

  Sanity (1054 positions):
    z: mean=0.013, std=1.000, [+/0/-]=534/0/520
    z by to_move: red=0.554 (+/0/-=411/0/118), black=-0.531 (+/0/-=123/0/402)
    v: mean=-0.037, std=0.837, range=[-1.00,1.00], mse=0.4686, sign_agree=83.6%
    v pretanh: range=[-4.26,5.08], p99=4.72, frac_sat=0.047, zv_corr=0.617 (n=256)
    TRIPWIRE: Value head saturating (p99=4.72, sat=0.047)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3640
  Step 40/160, Loss: 3.3875
  Step 60/160, Loss: 3.3569
  Step 80/160, Loss: 3.3530
  Step 100/160, Loss: 3.3477
  Step 120/160, Loss: 3.3580
  Step 140/160, Loss: 3.3765
  Step 160/160, Loss: 3.3693
  Average loss: 3.3693 (policy=3.2454, value=0.1061, l2=0.0974)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=30 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0055.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=315.9s, selfplay=259.5s, train=56.4s

============================================================
Iteration 56/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 36354 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 36750 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 36908 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 37068 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 37274 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1424 positions
  Results: Red=21, Black=4, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 37274
  Balance: black=16.0%, red=84.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 16.0% (<25%) for 4 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 213600, Leaf evals: 136225, NN batches: 12139
  Avg batch: 11.2, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=6383, stall=4388, tail=1368
  Avg plies: 57.0

  Sanity (1424 positions):
    z: mean=0.011, std=1.000, [+/0/-]=720/0/704
    z by to_move: red=0.640 (+/0/-=587/0/129), black=-0.624 (+/0/-=133/0/575)
    v: mean=-0.192, std=0.809, range=[-1.00,1.00], mse=0.8609, sign_agree=73.4%
    v pretanh: range=[-4.06,5.04], p99=4.58, frac_sat=0.074, zv_corr=0.416 (n=256)
    TRIPWIRE: Value head saturating (p99=4.58, sat=0.074)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2917
  Step 40/160, Loss: 3.3442
  Step 60/160, Loss: 3.3717
  Step 80/160, Loss: 3.3623
  Step 100/160, Loss: 3.3790
  Step 120/160, Loss: 3.3968
  Step 140/160, Loss: 3.3908
  Step 160/160, Loss: 3.3969
  Average loss: 3.3969 (policy=3.2701, value=0.1181, l2=0.0974)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=31 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0056.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=1259.3s, selfplay=1202.4s, train=56.9s

============================================================
Iteration 57/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 37416 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 37895 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 38096 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 38248 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 38645 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1371 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 38645
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 205650, Leaf evals: 162375, NN batches: 12640
  Avg batch: 12.8, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=8952, stall=2369, tail=1319
  Avg plies: 54.8

  Sanity (1371 positions):
    z: mean=0.011, std=1.000, [+/0/-]=693/0/678
    z by to_move: red=0.125 (+/0/-=386/0/300), black=-0.104 (+/0/-=307/0/378)
    v: mean=0.089, std=0.824, range=[-1.00,1.00], mse=1.2632, sign_agree=59.0%
    v pretanh: range=[-3.10,5.05], p99=4.60, frac_sat=0.086, zv_corr=0.212 (n=256)
    TRIPWIRE: Value head saturating (p99=4.60, sat=0.086)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2021
  Step 40/160, Loss: 3.2679
  Step 60/160, Loss: 3.2736
  Step 80/160, Loss: 3.3165
  Step 100/160, Loss: 3.3202
  Step 120/160, Loss: 3.3340
  Step 140/160, Loss: 3.3561
  Step 160/160, Loss: 3.3479
  Average loss: 3.3479 (policy=3.2215, value=0.1153, l2=0.0975)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=32 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0057.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=850.0s, selfplay=793.6s, train=56.4s

============================================================
Iteration 58/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 38794 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 38947 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 39084 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 39238 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 39387 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 742 positions
  Results: Red=23, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 39387
  Balance: black=8.0%, red=92.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 111300, Leaf evals: 95803, NN batches: 7032
  Avg batch: 13.6, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=5588, stall=723, tail=721
  Avg plies: 29.7

  Sanity (742 positions):
    z: mean=0.019, std=1.000, [+/0/-]=378/0/364
    z by to_move: red=0.814 (+/0/-=341/0/35), black=-0.798 (+/0/-=37/0/329)
    v: mean=0.004, std=0.956, range=[-1.00,1.00], mse=0.0833, sign_agree=96.9%
    v pretanh: range=[-4.01,5.70], p99=5.08, frac_sat=0.125, zv_corr=0.915 (n=256)
    TRIPWIRE: Value head saturating (p99=5.08, sat=0.125)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3891
  Step 40/160, Loss: 3.3726
  Step 60/160, Loss: 3.3570
  Step 80/160, Loss: 3.3720
  Step 100/160, Loss: 3.3795
  Step 120/160, Loss: 3.3804
  Step 140/160, Loss: 3.3812
  Step 160/160, Loss: 3.3784
  Average loss: 3.3784 (policy=3.2550, value=0.1019, l2=0.0979)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=33 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0058.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=221.2s, selfplay=164.8s, train=56.4s

============================================================
Iteration 59/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 39519 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 39773 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 39959 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 40160 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 40377 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 990 positions
  Results: Red=23, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 40377
  Balance: black=8.0%, red=92.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 148500, Leaf evals: 117627, NN batches: 9102
  Avg batch: 12.9, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6521, stall=1626, tail=955
  Avg plies: 39.6

  Sanity (990 positions):
    z: mean=0.010, std=1.000, [+/0/-]=500/0/490
    z by to_move: red=0.659 (+/0/-=414/0/85), black=-0.650 (+/0/-=86/0/405)
    v: mean=-0.076, std=0.909, range=[-1.00,1.00], mse=0.4846, sign_agree=84.4%
    v pretanh: range=[-4.37,4.07], p99=4.21, frac_sat=0.062, zv_corr=0.673 (n=256)
    TRIPWIRE: Value head saturating (p99=4.21, sat=0.062)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3319
  Step 40/160, Loss: 3.3555
  Step 60/160, Loss: 3.3556
  Step 80/160, Loss: 3.3779
  Step 100/160, Loss: 3.3543
  Step 120/160, Loss: 3.3472
  Step 140/160, Loss: 3.3467
  Step 160/160, Loss: 3.3571
  Average loss: 3.3571 (policy=3.2329, value=0.1074, l2=0.0974)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=34 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0059.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=277.7s, selfplay=221.2s, train=56.6s

============================================================
Iteration 60/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 40596 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 40742 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 40882 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 41087 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 41311 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 934 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 41311
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 140100, Leaf evals: 116241, NN batches: 8721
  Avg batch: 13.3, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6600, stall=1226, tail=895
  Avg plies: 37.4

  Sanity (934 positions):
    z: mean=0.015, std=1.000, [+/0/-]=474/0/460
    z by to_move: red=0.517 (+/0/-=355/0/113), black=-0.489 (+/0/-=119/0/347)
    v: mean=0.105, std=0.884, range=[-1.00,1.00], mse=0.8833, sign_agree=73.8%
    v pretanh: range=[-3.72,5.57], p99=4.50, frac_sat=0.074, zv_corr=0.455 (n=256)
    TRIPWIRE: Value head saturating (p99=4.50, sat=0.074)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3244
  Step 40/160, Loss: 3.3813
  Step 60/160, Loss: 3.3832
  Step 80/160, Loss: 3.3892
  Step 100/160, Loss: 3.3903
  Step 120/160, Loss: 3.3801
  Step 140/160, Loss: 3.3992
  Step 160/160, Loss: 3.4040
  Average loss: 3.4040 (policy=3.2764, value=0.1204, l2=0.0975)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=35 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0060.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=263.3s, selfplay=207.0s, train=56.3s

============================================================
Iteration 61/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 41709 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 41841 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 42089 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 42305 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 42673 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1362 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 42673
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 204300, Leaf evals: 132802, NN batches: 11708
  Avg batch: 11.3, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=6380, stall=4028, tail=1300
  Avg plies: 54.5

  Sanity (1362 positions):
    z: mean=0.012, std=1.000, [+/0/-]=689/0/673
    z by to_move: red=0.029 (+/0/-=352/0/332), black=-0.006 (+/0/-=337/0/341)
    v: mean=-0.172, std=0.763, range=[-1.00,1.00], mse=0.7656, sign_agree=72.7%
    v pretanh: range=[-4.24,5.15], p99=4.06, frac_sat=0.027, zv_corr=0.423 (n=256)
    TRIPWIRE: Value head saturating (p99=4.06, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5226
  Step 40/160, Loss: 3.5105
  Step 60/160, Loss: 3.4961
  Step 80/160, Loss: 3.5117
  Step 100/160, Loss: 3.4933
  Step 120/160, Loss: 3.4893
  Step 140/160, Loss: 3.4826
  Step 160/160, Loss: 3.4758
  Average loss: 3.4758 (policy=3.3473, value=0.1234, l2=0.0977)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=36 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0061.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=641.0s, selfplay=584.5s, train=56.5s

============================================================
Iteration 62/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 42833 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 43018 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 43176 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 43571 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 43961 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1288 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 43961
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 193200, Leaf evals: 126494, NN batches: 11109
  Avg batch: 11.4, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=6093, stall=3778, tail=1238
  Avg plies: 51.5

  Sanity (1288 positions):
    z: mean=0.011, std=1.000, [+/0/-]=651/0/637
    z by to_move: red=0.662 (+/0/-=540/0/110), black=-0.652 (+/0/-=111/0/527)
    v: mean=-0.156, std=0.812, range=[-1.00,1.00], mse=0.8633, sign_agree=71.5%
    v pretanh: range=[-3.96,4.72], p99=4.24, frac_sat=0.039, zv_corr=0.411 (n=256)
    TRIPWIRE: Value head saturating (p99=4.24, sat=0.039)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4611
  Step 40/160, Loss: 3.4966
  Step 60/160, Loss: 3.4885
  Step 80/160, Loss: 3.4657
  Step 100/160, Loss: 3.4538
  Step 120/160, Loss: 3.4550
  Step 140/160, Loss: 3.4639
  Step 160/160, Loss: 3.4472
  Average loss: 3.4472 (policy=3.3196, value=0.1187, l2=0.0979)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=37 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0062.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=462.6s, selfplay=405.7s, train=56.9s

============================================================
Iteration 63/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 44208 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 44400 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 44723 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 44909 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 45087 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1126 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 45087
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 168900, Leaf evals: 118982, NN batches: 10039
  Avg batch: 11.9, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=6215, stall=2769, tail=1055
  Avg plies: 45.0

  Sanity (1126 positions):
    z: mean=0.009, std=1.000, [+/0/-]=568/0/558
    z by to_move: red=0.294 (+/0/-=365/0/199), black=-0.278 (+/0/-=203/0/359)
    v: mean=-0.141, std=0.841, range=[-1.00,1.00], mse=0.7660, sign_agree=75.4%
    v pretanh: range=[-3.60,4.76], p99=4.35, frac_sat=0.059, zv_corr=0.480 (n=256)
    TRIPWIRE: Value head saturating (p99=4.35, sat=0.059)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4476
  Step 40/160, Loss: 3.4495
  Step 60/160, Loss: 3.4444
  Step 80/160, Loss: 3.4246
  Step 100/160, Loss: 3.4023
  Step 120/160, Loss: 3.4008
  Step 140/160, Loss: 3.3958
  Step 160/160, Loss: 3.3933
  Average loss: 3.3933 (policy=3.2611, value=0.1358, l2=0.0982)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=38 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0063.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=471.3s, selfplay=414.5s, train=56.8s

============================================================
Iteration 64/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 45328 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 45456 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 45604 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 45760 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 46123 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1036 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 46123
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 155400, Leaf evals: 106908, NN batches: 9129
  Avg batch: 11.7, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=5489, stall=2655, tail=985
  Avg plies: 41.4

  Sanity (1036 positions):
    z: mean=0.015, std=1.000, [+/0/-]=526/0/510
    z by to_move: red=0.523 (+/0/-=396/0/124), black=-0.496 (+/0/-=130/0/386)
    v: mean=-0.055, std=0.821, range=[-1.00,1.00], mse=0.5865, sign_agree=79.3%
    v pretanh: range=[-4.15,4.49], p99=4.36, frac_sat=0.043, zv_corr=0.545 (n=256)
    TRIPWIRE: Value head saturating (p99=4.36, sat=0.043)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3310
  Step 40/160, Loss: 3.4290
  Step 60/160, Loss: 3.4073
  Step 80/160, Loss: 3.4037
  Step 100/160, Loss: 3.4098
  Step 120/160, Loss: 3.4247
  Step 140/160, Loss: 3.4167
  Step 160/160, Loss: 3.4123
  Average loss: 3.4123 (policy=3.2820, value=0.1267, l2=0.0986)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=39 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0064.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=354.1s, selfplay=297.3s, train=56.9s

============================================================
Iteration 65/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 46263 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 46478 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 46807 positions, GPU: 21MB active, 0MB cache
  TIMEOUT: plies=420, last10=[(18, 17), (4, 21), (18, 14), (4, 7), (21, 14), (23, 6), (9, 18), (17, 9), (6, 15), (10, 3)]
  Games: 20/25, Buffer: 47334 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 47515 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1392 positions
  Results: Red=20, Black=4, Draws=1
    Draw breakdown: timeout=1, board_full=0, state_cap=0, unknown=0
  Buffer size: 47515
  Balance: black=16.7%, red=83.3%, draw=4.0% (decisive=24/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=24/25 (96%), draw=4%
  Backups: 208800, Leaf evals: 157940, NN batches: 12478
  Avg batch: 12.7, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=8344, stall=2805, tail=1329
  Avg plies: 55.7

  Sanity (1392 positions):
    z: mean=0.011, std=0.836, [+/0/-]=494/420/478
    z by to_move: red=0.481 (+/0/-=415/210/77), black=-0.467 (+/0/-=79/210/401)
    v: mean=-0.171, std=0.826, range=[-1.00,1.00], mse=0.4985, sign_agree=85.8%
    v pretanh: range=[-4.62,4.76], p99=4.56, frac_sat=0.055, zv_corr=0.655 (n=176)
    TRIPWIRE: Value head saturating (p99=4.56, sat=0.055)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4041
  Step 40/160, Loss: 3.4547
  Step 60/160, Loss: 3.4518
  Step 80/160, Loss: 3.4541
  Step 100/160, Loss: 3.4495
  Step 120/160, Loss: 3.4567
  Step 140/160, Loss: 3.4633
  Step 160/160, Loss: 3.4538
  Average loss: 3.4538 (policy=3.3217, value=0.1328, l2=0.0989)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=40 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0065.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=4110.2s, selfplay=4053.2s, train=56.9s

============================================================
Iteration 66/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 47656 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 47843 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 47972 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 48109 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 48237 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 722 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 48237
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 100% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 108300, Leaf evals: 91102, NN batches: 6779
  Avg batch: 13.4, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=5252, stall=830, tail=697
  Avg plies: 28.9

  Sanity (722 positions):
    z: mean=0.022, std=1.000, [+/0/-]=369/0/353
    z by to_move: red=0.307 (+/0/-=236/0/125), black=-0.263 (+/0/-=133/0/228)
    v: mean=-0.049, std=0.919, range=[-1.00,1.00], mse=0.4015, sign_agree=86.3%
    v pretanh: range=[-4.05,4.49], p99=4.16, frac_sat=0.055, zv_corr=0.722 (n=256)
    TRIPWIRE: Value head saturating (p99=4.16, sat=0.055)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5214
  Step 40/160, Loss: 3.4358
  Step 60/160, Loss: 3.4394
  Step 80/160, Loss: 3.4432
  Step 100/160, Loss: 3.4191
  Step 120/160, Loss: 3.4027
  Step 140/160, Loss: 3.3877
  Step 160/160, Loss: 3.3918
  Average loss: 3.3918 (policy=3.2612, value=0.1257, l2=0.0992)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=41 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0066.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=217.9s, selfplay=161.3s, train=56.6s

============================================================
Iteration 67/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 48450 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 48639 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 48908 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 49133 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 49285 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1048 positions
  Results: Red=10, Black=15, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 49285
  Balance: black=60.0%, red=40.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 157200, Leaf evals: 118580, NN batches: 9450
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6356, stall=2098, tail=996
  Avg plies: 41.9

  Sanity (1048 positions):
    z: mean=0.015, std=1.000, [+/0/-]=532/0/516
    z by to_move: red=-0.284 (+/0/-=187/0/335), black=0.312 (+/0/-=345/0/181)
    v: mean=-0.026, std=0.858, range=[-1.00,1.00], mse=0.4613, sign_agree=84.8%
    v pretanh: range=[-4.02,4.14], p99=3.97, frac_sat=0.023, zv_corr=0.638 (n=256)
    TRIPWIRE: Value head saturating (p99=3.97, sat=0.023)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3323
  Step 40/160, Loss: 3.3670
  Step 60/160, Loss: 3.4017
  Step 80/160, Loss: 3.4030
  Step 100/160, Loss: 3.4045
  Step 120/160, Loss: 3.4109
  Step 140/160, Loss: 3.4184
  Step 160/160, Loss: 3.4234
  Average loss: 3.4234 (policy=3.2889, value=0.1394, l2=0.0996)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=42 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0067.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=285.9s, selfplay=229.2s, train=56.7s

============================================================
Iteration 68/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 49537 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 49863 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 50126 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 50271 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 50734 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1449 positions
  Results: Red=16, Black=9, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 50734
  Balance: black=36.0%, red=64.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 217350, Leaf evals: 147765, NN batches: 12632
  Avg batch: 11.7, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=7310, stall=3937, tail=1385
  Avg plies: 58.0

  Sanity (1449 positions):
    z: mean=0.012, std=1.000, [+/0/-]=733/0/716
    z by to_move: red=0.076 (+/0/-=390/0/335), black=-0.052 (+/0/-=343/0/381)
    v: mean=-0.207, std=0.799, range=[-1.00,1.00], mse=1.0184, sign_agree=70.7%
    v pretanh: range=[-3.63,4.58], p99=4.09, frac_sat=0.027, zv_corr=0.331 (n=256)
    TRIPWIRE: Value head saturating (p99=4.09, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3162
  Step 40/160, Loss: 3.3380
  Step 60/160, Loss: 3.4449
  Step 80/160, Loss: 3.4381
  Step 100/160, Loss: 3.4409
  Step 120/160, Loss: 3.4427
  Step 140/160, Loss: 3.4359
  Step 160/160, Loss: 3.4279
  Average loss: 3.4279 (policy=3.2893, value=0.1548, l2=0.0998)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=43 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0068.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=548.8s, selfplay=492.0s, train=56.7s

============================================================
Iteration 69/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 50900 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 51041 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 51193 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 51400 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 51541 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 807 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 51541
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 95% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 121050, Leaf evals: 100481, NN batches: 7532
  Avg batch: 13.3, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5734, stall=1025, tail=773
  Avg plies: 32.3

  Sanity (807 positions):
    z: mean=0.019, std=1.000, [+/0/-]=411/0/396
    z by to_move: red=0.640 (+/0/-=333/0/73), black=-0.611 (+/0/-=78/0/323)
    v: mean=-0.003, std=0.929, range=[-1.00,1.00], mse=0.1378, sign_agree=94.5%
    v pretanh: range=[-3.74,4.64], p99=4.15, frac_sat=0.039, zv_corr=0.863 (n=256)
    TRIPWIRE: Value head saturating (p99=4.15, sat=0.039)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4203
  Step 40/160, Loss: 3.4043
  Step 60/160, Loss: 3.4144
  Step 80/160, Loss: 3.4257
  Step 100/160, Loss: 3.4156
  Step 120/160, Loss: 3.4281
  Step 140/160, Loss: 3.4444
  Step 160/160, Loss: 3.4390
  Average loss: 3.4390 (policy=3.3051, value=0.1354, l2=0.1001)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=44 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0069.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=241.1s, selfplay=184.3s, train=56.9s

============================================================
Iteration 70/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 51735 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 51907 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 52230 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 52566 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 52760 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1219 positions
  Results: Red=12, Black=13, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 52760
  Balance: black=52.0%, red=48.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 182850, Leaf evals: 128022, NN batches: 10828
  Avg batch: 11.8, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=6613, stall=3062, tail=1153
  Avg plies: 48.8

  Sanity (1219 positions):
    z: mean=0.014, std=1.000, [+/0/-]=618/0/601
    z by to_move: red=-0.211 (+/0/-=240/0/368), black=0.237 (+/0/-=378/0/233)
    v: mean=0.034, std=0.807, range=[-1.00,1.00], mse=0.6678, sign_agree=76.2%
    v pretanh: range=[-3.44,4.44], p99=4.12, frac_sat=0.027, zv_corr=0.492 (n=256)
    TRIPWIRE: Value head saturating (p99=4.12, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.2251
  Step 40/160, Loss: 3.2920
  Step 60/160, Loss: 3.3469
  Step 80/160, Loss: 3.3782
  Step 100/160, Loss: 3.4002
  Step 120/160, Loss: 3.4144
  Step 140/160, Loss: 3.4316
  Step 160/160, Loss: 3.4384
  Average loss: 3.4384 (policy=3.3039, value=0.1365, l2=0.1003)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=45 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0070.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=385.7s, selfplay=328.9s, train=56.8s

============================================================
Iteration 71/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 52915 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 53069 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 53343 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 53509 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 53722 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 962 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 53722
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 144300, Leaf evals: 115812, NN batches: 8863
  Avg batch: 13.1, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=6464, stall=1483, tail=916
  Avg plies: 38.5

  Sanity (962 positions):
    z: mean=0.019, std=1.000, [+/0/-]=490/0/472
    z by to_move: red=0.609 (+/0/-=391/0/95), black=-0.584 (+/0/-=99/0/377)
    v: mean=-0.039, std=0.885, range=[-1.00,1.00], mse=0.3356, sign_agree=88.3%
    v pretanh: range=[-4.04,5.10], p99=3.99, frac_sat=0.023, zv_corr=0.724 (n=256)
    TRIPWIRE: Value head saturating (p99=3.99, sat=0.023)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4710
  Step 40/160, Loss: 3.4781
  Step 60/160, Loss: 3.4693
  Step 80/160, Loss: 3.4704
  Step 100/160, Loss: 3.4681
  Step 120/160, Loss: 3.4462
  Step 140/160, Loss: 3.4390
  Step 160/160, Loss: 3.4354
  Average loss: 3.4354 (policy=3.3008, value=0.1365, l2=0.1004)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=46 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0071.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=286.9s, selfplay=230.1s, train=56.7s

============================================================
Iteration 72/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 53857 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 53997 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 54137 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 54291 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 54470 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 748 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 54470
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 112200, Leaf evals: 85895, NN batches: 6871
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=4825, stall=1340, tail=706
  Avg plies: 29.9

  Sanity (748 positions):
    z: mean=0.024, std=1.000, [+/0/-]=383/0/365
    z by to_move: red=0.500 (+/0/-=285/0/95), black=-0.467 (+/0/-=98/0/270)
    v: mean=0.025, std=0.843, range=[-1.00,1.00], mse=0.2687, sign_agree=87.9%
    v pretanh: range=[-3.85,4.08], p99=3.95, frac_sat=0.027, zv_corr=0.722 (n=256)
    TRIPWIRE: Value head saturating (p99=3.95, sat=0.027)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4140
  Step 40/160, Loss: 3.4865
  Step 60/160, Loss: 3.4476
  Step 80/160, Loss: 3.4232
  Step 100/160, Loss: 3.4313
  Step 120/160, Loss: 3.4380
  Step 140/160, Loss: 3.4448
  Step 160/160, Loss: 3.4513
  Average loss: 3.4513 (policy=3.3180, value=0.1309, l2=0.1006)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=47 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0072.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=218.1s, selfplay=161.4s, train=56.7s

============================================================
Iteration 73/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 54607 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 54904 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 55041 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 55188 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 55369 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 899 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 55369
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 134850, Leaf evals: 100128, NN batches: 8075
  Avg batch: 12.4, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=5355, stall=1864, tail=856
  Avg plies: 36.0

  Sanity (899 positions):
    z: mean=0.017, std=1.000, [+/0/-]=457/0/442
    z by to_move: red=0.227 (+/0/-=278/0/175), black=-0.197 (+/0/-=179/0/267)
    v: mean=-0.040, std=0.842, range=[-1.00,1.00], mse=0.4757, sign_agree=80.1%
    v pretanh: range=[-4.07,5.03], p99=4.81, frac_sat=0.156, zv_corr=0.617 (n=256)
    TRIPWIRE: Value head saturating (p99=4.81, sat=0.156)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5155
  Step 40/160, Loss: 3.4124
  Step 60/160, Loss: 3.4059
  Step 80/160, Loss: 3.4232
  Step 100/160, Loss: 3.4360
  Step 120/160, Loss: 3.4382
  Step 140/160, Loss: 3.4422
  Step 160/160, Loss: 3.4473
  Average loss: 3.4473 (policy=3.3116, value=0.1401, l2=0.1007)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=48 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0073.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=278.9s, selfplay=222.1s, train=56.8s

============================================================
Iteration 74/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 55650 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 56019 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 56155 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 56413 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 56818 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1449 positions
  Results: Red=15, Black=10, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 56818
  Balance: black=40.0%, red=60.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 217350, Leaf evals: 136471, NN batches: 12212
  Avg batch: 11.2, Avg waiters: 1.6, Max waiters: 17
  Flushes: full=6220, stall=4614, tail=1378
  Avg plies: 58.0

  Sanity (1449 positions):
    z: mean=0.012, std=1.000, [+/0/-]=733/0/716
    z by to_move: red=-0.054 (+/0/-=344/0/383), black=0.078 (+/0/-=389/0/333)
    v: mean=-0.231, std=0.805, range=[-1.00,1.00], mse=1.0073, sign_agree=68.0%
    v pretanh: range=[-4.13,4.94], p99=4.00, frac_sat=0.020, zv_corr=0.347 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3845
  Step 40/160, Loss: 3.3929
  Step 60/160, Loss: 3.4280
  Step 80/160, Loss: 3.4322
  Step 100/160, Loss: 3.4233
  Step 120/160, Loss: 3.4173
  Step 140/160, Loss: 3.4298
  Step 160/160, Loss: 3.4299
  Average loss: 3.4299 (policy=3.2946, value=0.1372, l2=0.1010)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=49 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0074.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=1093.6s, selfplay=1036.6s, train=57.0s

============================================================
Iteration 75/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 57022 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 57223 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 57420 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 57597 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 57822 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1004 positions
  Results: Red=17, Black=8, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 57822
  Balance: black=32.0%, red=68.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 150600, Leaf evals: 114831, NN batches: 9065
  Avg batch: 12.7, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6154, stall=1944, tail=967
  Avg plies: 40.2

  Sanity (1004 positions):
    z: mean=0.016, std=1.000, [+/0/-]=510/0/494
    z by to_move: red=0.184 (+/0/-=299/0/206), black=-0.154 (+/0/-=211/0/288)
    v: mean=-0.074, std=0.840, range=[-1.00,1.00], mse=0.4912, sign_agree=82.4%
    v pretanh: range=[-3.68,4.74], p99=4.41, frac_sat=0.039, zv_corr=0.610 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5129
  Step 40/160, Loss: 3.5291
  Step 60/160, Loss: 3.4923
  Step 80/160, Loss: 3.4953
  Step 100/160, Loss: 3.4793
  Step 120/160, Loss: 3.4592
  Step 140/160, Loss: 3.4609
  Step 160/160, Loss: 3.4673
  Average loss: 3.4673 (policy=3.3334, value=0.1306, l2=0.1013)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=50 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0075.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=279.7s, selfplay=223.0s, train=56.7s

============================================================
Iteration 76/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 58016 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 58286 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 58426 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 58715 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 58913 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1091 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 58913
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 163650, Leaf evals: 117941, NN batches: 9756
  Avg batch: 12.1, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=6215, stall=2515, tail=1026
  Avg plies: 43.6

  Sanity (1091 positions):
    z: mean=0.012, std=1.000, [+/0/-]=552/0/539
    z by to_move: red=0.438 (+/0/-=394/0/154), black=-0.418 (+/0/-=158/0/385)
    v: mean=-0.112, std=0.825, range=[-1.00,1.00], mse=0.7415, sign_agree=74.6%
    v pretanh: range=[-4.30,4.15], p99=4.05, frac_sat=0.039, zv_corr=0.476 (n=256)
    TRIPWIRE: Value head saturating (p99=4.05, sat=0.039)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3141
  Step 40/160, Loss: 3.3847
  Step 60/160, Loss: 3.3624
  Step 80/160, Loss: 3.3718
  Step 100/160, Loss: 3.4058
  Step 120/160, Loss: 3.4032
  Step 140/160, Loss: 3.3949
  Step 160/160, Loss: 3.3881
  Average loss: 3.3881 (policy=3.2548, value=0.1276, l2=0.1014)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=51 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0076.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=308.1s, selfplay=251.2s, train=56.9s

============================================================
Iteration 77/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 59071 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 59221 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 59376 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 59559 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 59690 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 777 positions
  Results: Red=14, Black=11, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 59690
  Balance: black=44.0%, red=56.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 116550, Leaf evals: 97960, NN batches: 7295
  Avg batch: 13.4, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5659, stall=908, tail=728
  Avg plies: 31.1

  Sanity (777 positions):
    z: mean=0.027, std=1.000, [+/0/-]=399/0/378
    z by to_move: red=0.013 (+/0/-=197/0/192), black=0.041 (+/0/-=202/0/186)
    v: mean=-0.041, std=0.908, range=[-1.00,1.00], mse=0.3189, sign_agree=89.8%
    v pretanh: range=[-4.27,5.85], p99=4.82, frac_sat=0.086, zv_corr=0.753 (n=256)
    TRIPWIRE: Value head saturating (p99=4.82, sat=0.086)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3917
  Step 40/160, Loss: 3.3992
  Step 60/160, Loss: 3.4749
  Step 80/160, Loss: 3.4651
  Step 100/160, Loss: 3.4666
  Step 120/160, Loss: 3.4694
  Step 140/160, Loss: 3.4784
  Step 160/160, Loss: 3.4717
  Average loss: 3.4717 (policy=3.3345, value=0.1429, l2=0.1015)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=52 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0077.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=231.1s, selfplay=174.6s, train=56.5s

============================================================
Iteration 78/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 60028 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 60243 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 60493 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 60741 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 60948 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1258 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 60948
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 188700, Leaf evals: 149844, NN batches: 11539
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=8216, stall=2126, tail=1197
  Avg plies: 50.3

  Sanity (1258 positions):
    z: mean=0.010, std=1.000, [+/0/-]=635/0/623
    z by to_move: red=0.518 (+/0/-=479/0/152), black=-0.502 (+/0/-=156/0/471)
    v: mean=-0.107, std=0.845, range=[-1.00,1.00], mse=0.6180, sign_agree=79.3%
    v pretanh: range=[-3.55,4.79], p99=4.66, frac_sat=0.074, zv_corr=0.553 (n=256)
    TRIPWIRE: Value head saturating (p99=4.66, sat=0.074)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4049
  Step 40/160, Loss: 3.3836
  Step 60/160, Loss: 3.4011
  Step 80/160, Loss: 3.4132
  Step 100/160, Loss: 3.3985
  Step 120/160, Loss: 3.4088
  Step 140/160, Loss: 3.4202
  Step 160/160, Loss: 3.4271
  Average loss: 3.4271 (policy=3.2896, value=0.1432, l2=0.1017)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=53 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0078.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=418.7s, selfplay=362.0s, train=56.7s

============================================================
Iteration 79/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 61077 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 61226 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 61426 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 61599 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 61791 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 843 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 61791
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 126450, Leaf evals: 100441, NN batches: 7758
  Avg batch: 12.9, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5620, stall=1344, tail=794
  Avg plies: 33.7

  Sanity (843 positions):
    z: mean=0.011, std=1.000, [+/0/-]=426/0/417
    z by to_move: red=0.325 (+/0/-=279/0/142), black=-0.303 (+/0/-=147/0/275)
    v: mean=0.053, std=0.871, range=[-1.00,1.00], mse=0.5018, sign_agree=84.0%
    v pretanh: range=[-3.76,3.99], p99=3.77, frac_sat=0.004, zv_corr=0.630 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4652
  Step 40/160, Loss: 3.4040
  Step 60/160, Loss: 3.3873
  Step 80/160, Loss: 3.4182
  Step 100/160, Loss: 3.4182
  Step 120/160, Loss: 3.4219
  Step 140/160, Loss: 3.4244
  Step 160/160, Loss: 3.4240
  Average loss: 3.4240 (policy=3.2901, value=0.1283, l2=0.1019)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=54 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0079.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=242.3s, selfplay=185.2s, train=57.2s

============================================================
Iteration 80/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 62107 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 62250 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 62459 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 62618 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 62812 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1021 positions
  Results: Red=11, Black=14, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 62812
  Balance: black=56.0%, red=44.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 153150, Leaf evals: 112769, NN batches: 9225
  Avg batch: 12.2, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6031, stall=2225, tail=969
  Avg plies: 40.8

  Sanity (1021 positions):
    z: mean=0.013, std=1.000, [+/0/-]=517/0/504
    z by to_move: red=-0.235 (+/0/-=195/0/315), black=0.260 (+/0/-=322/0/189)
    v: mean=-0.065, std=0.860, range=[-1.00,1.00], mse=0.7401, sign_agree=76.2%
    v pretanh: range=[-3.05,5.11], p99=4.77, frac_sat=0.059, zv_corr=0.502 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4347
  Step 40/160, Loss: 3.4940
  Step 60/160, Loss: 3.4664
  Step 80/160, Loss: 3.4759
  Step 100/160, Loss: 3.4673
  Step 120/160, Loss: 3.4582
  Step 140/160, Loss: 3.4389
  Step 160/160, Loss: 3.4326
  Average loss: 3.4326 (policy=3.2938, value=0.1472, l2=0.1020)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=55 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0080.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=286.0s, selfplay=228.9s, train=57.1s

============================================================
Iteration 81/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 62998 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 63155 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 63576 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 63713 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 64018 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1206 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 64018
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 180900, Leaf evals: 134422, NN batches: 10782
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=7046, stall=2573, tail=1163
  Avg plies: 48.2

  Sanity (1206 positions):
    z: mean=0.010, std=1.000, [+/0/-]=609/0/597
    z by to_move: red=0.539 (+/0/-=468/0/140), black=-0.528 (+/0/-=141/0/457)
    v: mean=0.044, std=0.822, range=[-1.00,1.00], mse=0.6596, sign_agree=77.7%
    v pretanh: range=[-3.28,4.44], p99=4.03, frac_sat=0.035, zv_corr=0.509 (n=256)
    TRIPWIRE: Value head saturating (p99=4.03, sat=0.035)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4425
  Step 40/160, Loss: 3.4170
  Step 60/160, Loss: 3.4364
  Step 80/160, Loss: 3.4263
  Step 100/160, Loss: 3.4477
  Step 120/160, Loss: 3.4416
  Step 140/160, Loss: 3.4550
  Step 160/160, Loss: 3.4515
  Average loss: 3.4515 (policy=3.3134, value=0.1431, l2=0.1023)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=56 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0081.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=959.3s, selfplay=902.0s, train=57.3s

============================================================
Iteration 82/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 64303 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 64473 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 64612 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 64811 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 65215 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1197 positions
  Results: Red=16, Black=9, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 65215
  Balance: black=36.0%, red=64.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 179550, Leaf evals: 119644, NN batches: 10203
  Avg batch: 11.7, Avg waiters: 1.5, Max waiters: 17
  Flushes: full=5720, stall=3348, tail=1135
  Avg plies: 47.9

  Sanity (1197 positions):
    z: mean=0.009, std=1.000, [+/0/-]=604/0/593
    z by to_move: red=0.120 (+/0/-=337/0/265), black=-0.103 (+/0/-=267/0/328)
    v: mean=-0.184, std=0.809, range=[-1.00,1.00], mse=0.7691, sign_agree=70.7%
    v pretanh: range=[-4.92,4.15], p99=4.07, frac_sat=0.020, zv_corr=0.459 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4841
  Step 40/160, Loss: 3.5371
  Step 60/160, Loss: 3.5639
  Step 80/160, Loss: 3.5343
  Step 100/160, Loss: 3.5209
  Step 120/160, Loss: 3.5117
  Step 140/160, Loss: 3.4949
  Step 160/160, Loss: 3.4928
  Average loss: 3.4928 (policy=3.3539, value=0.1451, l2=0.1027)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=57 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0082.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.0%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=711.7s, selfplay=654.0s, train=57.7s

============================================================
Iteration 83/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 65516 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 65678 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 65843 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 66091 positions, GPU: 21MB active, 0MB cache
  TIMEOUT: plies=420, last10=[(10, 13), (13, 3), (9, 19), (18, 16), (10, 19), (13, 4), (2, 12), (5, 10), (18, 7), (14, 1)]
  Games: 25/25, Buffer: 66643 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1428 positions
  Results: Red=18, Black=6, Draws=1
    Draw breakdown: timeout=1, board_full=0, state_cap=0, unknown=0
  Buffer size: 66643
  Balance: black=25.0%, red=75.0%, draw=4.0% (decisive=24/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=24/25 (96%), draw=4%
  Backups: 214200, Leaf evals: 170297, NN batches: 13117
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=9359, stall=2388, tail=1370
  Avg plies: 57.1

  Sanity (1428 positions):
    z: mean=0.010, std=0.840, [+/0/-]=511/420/497
    z by to_move: red=0.174 (+/0/-=316/210/191), black=-0.156 (+/0/-=195/210/306)
    v: mean=-0.087, std=0.843, range=[-1.00,1.00], mse=0.6964, sign_agree=77.7%
    v pretanh: range=[-3.46,4.65], p99=4.41, frac_sat=0.020, zv_corr=0.514 (n=193)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3991
  Step 40/160, Loss: 3.4442
  Step 60/160, Loss: 3.4602
  Step 80/160, Loss: 3.4640
  Step 100/160, Loss: 3.4571
  Step 120/160, Loss: 3.4685
  Step 140/160, Loss: 3.4690
  Step 160/160, Loss: 3.4694
  Average loss: 3.4694 (policy=3.3287, value=0.1519, l2=0.1028)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=58 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0083.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=3566.3s, selfplay=3508.4s, train=57.9s

============================================================
Iteration 84/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 66799 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 66964 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 67173 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 67302 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 67461 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 818 positions
  Results: Red=18, Black=7, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 67461
  Balance: black=28.0%, red=72.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 122700, Leaf evals: 98596, NN batches: 7590
  Avg batch: 13.0, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5577, stall=1235, tail=778
  Avg plies: 32.7

  Sanity (818 positions):
    z: mean=0.020, std=1.000, [+/0/-]=417/0/401
    z by to_move: red=0.322 (+/0/-=273/0/140), black=-0.289 (+/0/-=144/0/261)
    v: mean=-0.048, std=0.925, range=[-1.00,1.00], mse=0.2916, sign_agree=89.8%
    v pretanh: range=[-4.07,4.44], p99=4.35, frac_sat=0.078, zv_corr=0.784 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5988
  Step 40/160, Loss: 3.4685
  Step 60/160, Loss: 3.4727
  Step 80/160, Loss: 3.4969
  Step 100/160, Loss: 3.4787
  Step 120/160, Loss: 3.4672
  Step 140/160, Loss: 3.4830
  Step 160/160, Loss: 3.4876
  Average loss: 3.4876 (policy=3.3487, value=0.1433, l2=0.1031)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=59 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0084.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=242.1s, selfplay=185.0s, train=57.2s

============================================================
Iteration 85/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 67625 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 67753 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 67894 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 68019 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 68185 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 724 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 68185
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 108600, Leaf evals: 91464, NN batches: 6786
  Avg batch: 13.5, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=5293, stall=810, tail=683
  Avg plies: 29.0

  Sanity (724 positions):
    z: mean=0.017, std=1.000, [+/0/-]=368/0/356
    z by to_move: red=0.760 (+/0/-=323/0/44), black=-0.748 (+/0/-=45/0/312)
    v: mean=0.017, std=0.929, range=[-1.00,1.00], mse=0.1511, sign_agree=92.6%
    v pretanh: range=[-3.77,4.74], p99=4.29, frac_sat=0.066, zv_corr=0.856 (n=256)
    TRIPWIRE: Value head saturating (p99=4.29, sat=0.066)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3579
  Step 40/160, Loss: 3.3565
  Step 60/160, Loss: 3.3936
  Step 80/160, Loss: 3.4061
  Step 100/160, Loss: 3.4234
  Step 120/160, Loss: 3.4484
  Step 140/160, Loss: 3.4496
  Step 160/160, Loss: 3.4505
  Average loss: 3.4505 (policy=3.3106, value=0.1463, l2=0.1033)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=60 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0085.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=225.0s, selfplay=167.8s, train=57.1s

============================================================
Iteration 86/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 68358 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 68640 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 68780 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 68924 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 69052 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 867 positions
  Results: Red=22, Black=3, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 69052
  Balance: black=12.0%, red=88.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 85% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 130050, Leaf evals: 101443, NN batches: 7895
  Avg batch: 12.8, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5573, stall=1487, tail=835
  Avg plies: 34.7

  Sanity (867 positions):
    z: mean=0.013, std=1.000, [+/0/-]=439/0/428
    z by to_move: red=0.748 (+/0/-=381/0/55), black=-0.731 (+/0/-=58/0/373)
    v: mean=-0.134, std=0.869, range=[-1.00,1.00], mse=0.5084, sign_agree=82.0%
    v pretanh: range=[-3.27,5.00], p99=3.89, frac_sat=0.016, zv_corr=0.633 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.5069
  Step 40/160, Loss: 3.4585
  Step 60/160, Loss: 3.4691
  Step 80/160, Loss: 3.4766
  Step 100/160, Loss: 3.4726
  Step 120/160, Loss: 3.4701
  Step 140/160, Loss: 3.4775
  Step 160/160, Loss: 3.4671
  Average loss: 3.4671 (policy=3.3298, value=0.1358, l2=0.1034)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=61 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0086.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=289.6s, selfplay=232.4s, train=57.2s

============================================================
Iteration 87/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 69198 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 69331 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 69493 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 69668 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 69871 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 819 positions
  Results: Red=20, Black=5, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 69871
  Balance: black=20.0%, red=80.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: black decisive win rate 20.0% (<25%) for 3 consecutive iters | decisive=25/25 (100%), draw=0%
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 122850, Leaf evals: 103539, NN batches: 7733
  Avg batch: 13.4, Avg waiters: 1.2, Max waiters: 17
  Flushes: full=5977, stall=962, tail=794
  Avg plies: 32.8

  Sanity (819 positions):
    z: mean=0.018, std=1.000, [+/0/-]=417/0/402
    z by to_move: red=0.564 (+/0/-=323/0/90), black=-0.537 (+/0/-=94/0/312)
    v: mean=0.028, std=0.916, range=[-1.00,1.00], mse=0.2082, sign_agree=92.6%
    v pretanh: range=[-3.76,5.62], p99=5.21, frac_sat=0.160, zv_corr=0.816 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4564
  Step 40/160, Loss: 3.4866
  Step 60/160, Loss: 3.4749
  Step 80/160, Loss: 3.4907
  Step 100/160, Loss: 3.4835
  Step 120/160, Loss: 3.4616
  Step 140/160, Loss: 3.4510
  Step 160/160, Loss: 3.4430
  Average loss: 3.4430 (policy=3.3047, value=0.1393, l2=0.1035)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=62 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0087.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=244.4s, selfplay=186.8s, train=57.6s

============================================================
Iteration 88/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 70063 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 70219 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 70468 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 70639 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 70880 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1009 positions
  Results: Red=14, Black=11, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 70880
  Balance: black=44.0%, red=56.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 151350, Leaf evals: 103775, NN batches: 8988
  Avg batch: 11.5, Avg waiters: 1.4, Max waiters: 17
  Flushes: full=5437, stall=2586, tail=965
  Avg plies: 40.4

  Sanity (1009 positions):
    z: mean=0.013, std=1.000, [+/0/-]=511/0/498
    z by to_move: red=-0.138 (+/0/-=218/0/288), black=0.165 (+/0/-=293/0/210)
    v: mean=-0.214, std=0.854, range=[-1.00,1.00], mse=0.6342, sign_agree=78.9%
    v pretanh: range=[-3.72,4.39], p99=4.13, frac_sat=0.020, zv_corr=0.570 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.4342
  Step 40/160, Loss: 3.4180
  Step 60/160, Loss: 3.4340
  Step 80/160, Loss: 3.4480
  Step 100/160, Loss: 3.4371
  Step 120/160, Loss: 3.4335
  Step 140/160, Loss: 3.4460
  Step 160/160, Loss: 3.4356
  Average loss: 3.4356 (policy=3.2951, value=0.1478, l2=0.1036)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=63 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0088.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=330.5s, selfplay=273.4s, train=57.1s

============================================================
Iteration 89/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 71014 positions, GPU: 21MB active, 0MB cache
  Games: 10/25, Buffer: 71191 positions, GPU: 21MB active, 0MB cache
  Games: 15/25, Buffer: 71525 positions, GPU: 21MB active, 0MB cache
  Games: 20/25, Buffer: 71803 positions, GPU: 21MB active, 0MB cache
  Games: 25/25, Buffer: 71967 positions, GPU: 21MB active, 0MB cache
  Generated 25 games, 1087 positions
  Results: Red=19, Black=6, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 71967
  Balance: black=24.0%, red=76.0%, draw=0.0% (decisive=25/25, window=20/20)
  ⚠️  BALANCE WARNING: Red dominated 90% of last 20 eligible iters (>80%) | decisive=25/25 (100%), draw=0%
  Backups: 163050, Leaf evals: 120321, NN batches: 9663
  Avg batch: 12.5, Avg waiters: 1.3, Max waiters: 17
  Flushes: full=6293, stall=2332, tail=1038
  Avg plies: 43.5

  Sanity (1087 positions):
    z: mean=0.012, std=1.000, [+/0/-]=550/0/537
    z by to_move: red=0.400 (+/0/-=383/0/164), black=-0.381 (+/0/-=167/0/373)
    v: mean=-0.033, std=0.802, range=[-1.00,1.00], mse=0.4490, sign_agree=82.8%
    v pretanh: range=[-3.15,5.06], p99=4.56, frac_sat=0.043, zv_corr=0.598 (n=256)

Training: 160 steps... (value_weight=0.25)
  Step 20/160, Loss: 3.3610
  Step 40/160, Loss: 3.3508
  Step 60/160, Loss: 3.3967
  Step 80/160, Loss: 3.4125
  Step 100/160, Loss: 3.4330
  Step 120/160, Loss: 3.4323
  Step 140/160, Loss: 3.4470
  Step 160/160, Loss: 3.4326
  Average loss: 3.4326 (policy=3.2918, value=0.1484, l2=0.1037)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=64 demote_streak=0 full=True train=160/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0089.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=524.2s, selfplay=467.1s, train=57.1s

============================================================
Iteration 90/200
  Curriculum: active_size=24, max_moves=420
  Sims: cli=800, table=150, factor=1.00, effective=150 (table)
============================================================

Self-play: generating 25 games...
  Games: 5/25, Buffer: 72131 positions, GPU: 21MB active, 0MB cache
^C

Interrupted during self-play! Completed 5/25 games.
Saving partial checkpoint and exiting...
  Generated 5 games, 164 positions
  Results: Red=3, Black=2, Draws=0
    Draw breakdown: timeout=0, board_full=0, state_cap=0, unknown=0
  Buffer size: 72131
  Balance: skipped (decisive=5/5, draw=0.0%)
  Backups: 24600, Leaf evals: 20825, NN batches: 1546
  Avg batch: 13.5, Avg waiters: 1.1, Max waiters: 17
  Flushes: full=1201, stall=183, tail=162
  Avg plies: 32.8

  Sanity (164 positions):
    z: mean=0.012, std=1.000, [+/0/-]=83/0/81
    z by to_move: red=0.358 (+/0/-=55/0/26), black=-0.325 (+/0/-=28/0/55)
    v: mean=0.038, std=0.917, range=[-1.00,1.00], mse=0.0898, sign_agree=97.0%
    v pretanh: range=[-3.23,3.95], p99=3.88, frac_sat=0.024, zv_corr=0.876 (n=164)
    TRIPWIRE: Value head saturating (p99=3.88, sat=0.024)
              Consider: --value-lr-scale 0.05 or reducing value_grad_max_norm

Skipping training (interrupted)
  State: size=24 sims=150 factor=1.00 frozen=False promo_streak=64 demote_streak=0 full=False train=0/160

Checkpoint saved: checkpoints/alphazero-fresh/model_iter_0090_partial.safetensors
  Curriculum: active_size=24, draw_rate_true=0.0%, timeout_rate_rolling=0.5%
  Iter metrics (partial): iter_timeout_rate=0.0% (0/5 games)
  Status: sims_used=150, sims_next=150, factor=1.00, frozen=False
  Timing: iter=942.3s, selfplay=942.3s, train=0.0s
