# H-TB G3, V8a browser check: one `hard` game in the browser client (card §8 step 1)

**Run:** 2026-11-10, 08:19–08:23 local. Launch worktree `/Users/bill/projects/TwixT_Game-g3`, branch `htb-g3-launch`,
at `542bb847`. The tree was clean and `node_modules` was an APFS clone (`cp -c -R`) of the main checkout's.
`node tests/htb_integration/g3_match.mjs bind` in that worktree reproduced `tests/htb_integration/g3_binding.json`
exactly (digest `da0b86bf…`).

**Machine:** Unreal Editor closed (its editor process and `zenserver` had exited); BOINC quit (only the exempt podman
VM ran). Two Epic processes were still running at about 1% CPU: `UnrealEditorServices` (pid 4310) and the Epic Games
Launcher. 1-minute load 2.52 just before the server started.

**Server:** `npm start` in default mode, with `MODEL_MANIFEST`, `HARD_MODEL_MANIFEST` and `HARD_MODE` unset. The only
addition was `TWIXT_NO_BROWSER=1`, so the launcher didn't open its own window; the game was played in a Claude in
Chrome tab. Log: `server.log` (copied from the run).

**Client:** http://localhost:5500. Settings: 1 Player Mode, Hard, REC off (grey dot, checked before the game). Black
(the AI) starts. I played Red with real mouse clicks on the board's holes. The screen coordinates came from projecting
each hole's 3-D position through the page's camera; the moves were placed through the client's normal click path.
Red's moves were deliberately weak. The aim was a complete game, not a strength test.

## Checks

| Check | Result |
|---|---|
| The server banner names `Hard mode: protocol` and both model ids | PASS. `[AI] Hard mode:  protocol`, `[AI] Hard model id:   78c16a31c3dba37f`, default `Model id: c34b7ff3297c785a`; both ONNX models loaded |
| The AI's moves arrive | PASS. 23 of 23 AI moves (plies 0, 2, …, 44) arrived over the WebSocket |
| The MCTS bar updates after each AI move, including across ply 20 | PASS. The bar changed after the AI's moves: Red 37% → 17% → 8% → 3% → 1% … then 4% after the ply-20 move. From ply 20 the bar follows `valueRed` from the played move's mean (`valueKind: played_move_mean`, D6), while the console's `value` is the root mean. So they differ from ply 20 on (ply 20: value 0.737, valueRed −0.916; ply 22: 0.941 / −0.992), and before ply 20 they are exact negatives. This is the designed behaviour (`server/protocol_move.js` header, `displayValue`) |
| The console shows no client error | PASS. `read_console_messages` with `onlyErrors`: "No console errors or exceptions found for this tab." The server's only complaint was `File not found: /favicon.ico` (a static-file 404) |
| The game ends normally | PASS. Black connected left ↔ right with (18, 23) at ply 44, 45 moves in all; `game.gameOver = true`, `winner = 'black'`; both bars went to the terminal value (Red 0%); the "Black Wins!" modal showed (`final_board.jpg`) |

**Per-move times (client log, `elapsed`, all 200 simulations as reported):**
- plies 0–18: 1,434–1,509 ms;
- plies 20–42: 1,425–1,453 ms;
- ply 44, the winning move: 960 ms.

These figures are informational. The smoke pair measures timing for the match.

## Observations that are not failures

1. **The win `alert()` blocks automation.** At game end the client calls a blocking `alert("🎉 Black Player Wins! 🎉")`
   (`assets/js/game/gameController.js:219`) before showing the modal. That stopped Claude in Chrome
   (`Runtime.evaluate` timed out) until the user dismissed it by hand. This is existing product behaviour, unchanged
   by H-TB, and it isn't one of the checks.
2. **After the dialog was dismissed, the console reader returned the whole game a second time,** with every entry
   stamped 08:23:52. The game state had only 45 plies and no request was sent, so this is most likely the extension
   replaying its buffered log when it reattached, not new moves. This is an interpretation; it wasn't verified further.
3. **Claude in Chrome was not connected** at the first attempt that morning, before the user connected it. The server
   had been started once, showed the same banner, and was stopped cleanly; no game was played then.

## Moves (row, col; b = AI)

```
b7,13 r12,11 b7,9 r10,12 b8,15 r14,12 b7,5 r8,13 b6,11 r16,13 b8,19 r6,14 b7,17 r18,14 b20,14 r4,15 b8,7 r20,15
b21,16 r2,16 b20,10 r22,16 b20,18 r0,17 b19,8 r23,18 b19,12 r3,3 b19,4 r3,20 b19,20 r21,3 b18,2 r12,8 b8,3 r14,8
b7,21 r10,8 b18,6 r16,8 b19,0 r12,16 b20,22 r14,16 b18,23
```

**Verdict for step 1: all five checks PASS.** The sequence continues to the smoke pair (card §8 step 2). The server
was stopped after the game (ports 5500 and 3001 free) and the tab was closed.
