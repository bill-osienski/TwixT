package net.schwagereit.t1j;

/*
 * PROVENANCE. Original code written for the E4 preflight. It contains no T1j
 * source and modifies no T1j class. It declares T1j's own package deliberately,
 * because the members it needs -- the Match constructor, setlastMove,
 * getBoardY/getBoardX -- are package-visible. It is compiled against, and links
 * with, T1j (GPL-3.0), which is neither modified nor redistributed here.
 *
 * A GENERIC fixed-position query path: the position and the search depth both
 * come from argv. Nothing about the position or the depth is hard-coded, which
 * is the difference from E3bDump's single "smoke" position.
 *
 * NO GAMES. Every mode below answers ONE question about ONE frozen position:
 * what move does T1j return at a requested fixed depth, did that depth actually
 * complete, and is the position T1j searched the one we bound. No opponent
 * moves, no self-play, no seeds.
 *
 * Reflection is limited to the three fields already qualified by E2 attempt 4
 * and E3a, and is audited in a finally path.
 *
 * modes:
 *   query        <depth> <x,y ...>              one construction, one computeMove
 *   determinism  <n> <depth> <x,y ...>          n independent constructions in THIS jvm
 *
 * Exit 0 = every requirement met, 3 = one or more failed, 4 = threw.
 */

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.prefs.Preferences;
import java.util.stream.Stream;

public final class E4Preflight {

    private static final List<String> REFLECTED = new ArrayList<String>();
    private static final List<String> AUTHORIZED = Arrays.asList(
        "net.schwagereit.t1j.Match.nextPlayer(write)",
        "net.schwagereit.t1j.FindMove.usealphabeta(read)",
        "net.schwagereit.t1j.FindMove.currentMaxPly(read)");
    private static int failures = 0;
    private static String plistBefore;
    private static long countBefore;

    private static void req(boolean c, String what) {
        if (!c) { System.out.println("FAIL " + what); failures++; }
    }
    private static Field reflect(Class<?> o, String n, String m) throws Exception {
        REFLECTED.add(o.getName() + "." + n + "(" + m + ")");
        Field f = o.getDeclaredField(n); f.setAccessible(true); return f;
    }
    private static Path prefsDir() {
        return Paths.get(System.getProperty("user.home"), "Library", "Preferences");
    }
    private static String plistHash() {
        try {
            Path p = prefsDir().resolve("com.apple.java.util.prefs.plist");
            if (!Files.exists(p)) return "ABSENT";
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(Files.readAllBytes(p)));
        } catch (Exception e) { return "ERROR"; }
    }
    private static long prefsCount() {
        try (Stream<Path> s = Files.list(prefsDir())) { return s.count(); }
        catch (Exception e) { return -1L; }
    }
    private static int windows() { try { return java.awt.Window.getWindows().length; } catch (Throwable t) { return -1; } }
    private static int frames()  { try { return java.awt.Frame.getFrames().length;  } catch (Throwable t) { return -1; } }

    public static void main(String[] args) {
        Throwable thrown = null;
        try {
            plistBefore = plistHash(); countBefore = prefsCount();
            req("e2probe.ScratchPrefs".equals(Preferences.userRoot().getClass().getName()),
                "isolated Preferences active");
            req(java.awt.GraphicsEnvironment.isHeadless(), "headless at start");
            req(windows() == 0 && frames() == 0, "zero Window/Frame at start");
            CheckPattern.getInstance().loadPattern();
            Zobrist.getInstance().initialize();

            System.out.println("PROC pid=" + ProcessHandle.current().pid()
                + " java_version=" + System.getProperty("java.version")
                + " vm=" + System.getProperty("java.vm.name").replace(' ', '_')
                + " headless=" + System.getProperty("java.awt.headless")
                + " prefs_factory=" + Preferences.userRoot().getClass().getName());

            String mode = args.length > 0 ? args[0] : "";
            if ("query".equals(mode)) {
                int depth = Integer.parseInt(args[1]);
                queries(1, depth, Arrays.copyOfRange(args, 2, args.length));
            } else if ("determinism".equals(mode)) {
                int n = Integer.parseInt(args[1]);
                int depth = Integer.parseInt(args[2]);
                queries(n, depth, Arrays.copyOfRange(args, 3, args.length));
            } else {
                System.out.println("FAIL unknown mode " + mode); failures++;
            }
        } catch (Throwable t) {
            thrown = t;
            System.out.println("THREW: " + t);
            t.printStackTrace(System.out);
        } finally {
            post(thrown);
        }
        System.exit(failures == 0 ? 0 : 3);
    }

    /** A fresh Match with both boards cleared, sized and set up, Y to move. */
    private static Match freshMatch() throws Exception {
        Match m = new Match();                     // the ctor repoints FindMove's static match
        Board bY = m.getBoardY(), bX = m.getBoardX();
        bY.clearBoard(); bX.clearBoard();
        bY.setSize(24, 24); bX.setSize(24, 24);
        bY.getEval().setupForY(); bX.getEval().setupForY();
        reflect(Match.class, "nextPlayer", "write").setInt(m, Board.YPLAYER);
        return m;
    }

    /**
     * n independent constructions of the SAME frozen position, each rebuilt and
     * revalidated from scratch, each followed by one computeMove at `depth`.
     */
    private static void queries(int n, int depth, String[] xy) throws Exception {
        GeneralSettings gs = GeneralSettings.getInstance();
        for (int q = 1; q <= n; q++) {
            Match m = freshMatch();
            Board bY = m.getBoardY(), bX = m.getBoardX();
            for (String s : xy) {
                String[] p = s.split(",");
                req(m.setlastMove(Integer.parseInt(p[0]), Integer.parseInt(p[1])),
                    "q" + q + ": setlastMove " + s + " accepted");
            }
            req(m.getMoveNr() == xy.length, "q" + q + ": moveNr == submitted plies");

            // the position T1j is about to search, in E3bDump's dump vocabulary,
            // so the E3b-qualified parser binds it without a second format
            dump(m, bY, bX);

            // wall-clock mode is excluded by construction: mdFixedPly is forced true
            gs.mdFixedPly = true;
            gs.mdPly = depth;
            int toMove = m.getNextPlayer();
            FindMove fm = FindMove.getFindMove();
            long t0 = System.nanoTime();
            Move mv = fm.computeMove(toMove);
            long elapsedUs = (System.nanoTime() - t0) / 1000L;

            boolean uab = reflect(FindMove.class, "usealphabeta", "read").getBoolean(fm);
            int cmp = reflect(FindMove.class, "currentMaxPly", "read").getInt(null);
            int mx = mv == null ? -1 : mv.getX(), my = mv == null ? -1 : mv.getY();
            // under mdFixedPly the loop cannot break early, so it exits at maxPly+1
            boolean completed = uab && cmp == depth + 1;
            boolean legal = mv != null && bY.getPin(mx, my) == 0 && bY.pinAllowed(mx, my, toMove);

            System.out.println("QUERY q=" + q
                + " requested_depth=" + depth
                + " mdFixedPly=" + gs.mdFixedPly + " mdPly=" + gs.mdPly
                + " move_x=" + mx + " move_y=" + my
                + " to_move=" + (toMove == Board.YPLAYER ? "Y" : "X")
                + " usealphabeta=" + uab
                + " currentMaxPly=" + cmp
                + " completed_depth=" + (uab ? cmp - 1 : -1)
                + " completed=" + completed
                + " legal=" + legal
                + " null_sentinel=" + (mx == -1 && my == -1)
                + " moveNr=" + m.getMoveNr()
                + " eval_regime=" + (m.getMoveNr() < 8 ? "early_moveNr_lt_8" : "normal")
                + " elapsed_us=" + elapsedUs);

            req(completed, "q" + q + ": requested depth " + depth + " completed");
            req(legal, "q" + q + ": returned move is legal in T1j");
        }
    }

    /** Identical wire format to E3bDump.dump, so t1j_adapter.parse_dump binds it. */
    private static void dump(Match m, Board bY, Board bX) {
        int n = bY.getXsize();
        Set<String> pegs = new TreeSet<String>();
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++) {
                int v = bY.getPin(x, y);
                if (v != 0) pegs.add(x + "," + y + "," + (v == Board.YPLAYER ? "Y" : "X"));
            }
        Set<String> br = new TreeSet<String>();
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                for (int d = 0; d < 4; d++)
                    if (bY.isBridged(x, y, d)) {
                        int e = Board.bridgeEnd(x, y, d);
                        int tx = e / 1000, ty = e % 1000;
                        int owner = bY.getPin(x, y);
                        String a = x + "," + y, b = tx + "," + ty;
                        String lo = a.compareTo(b) <= 0 ? a : b, hi = a.compareTo(b) <= 0 ? b : a;
                        br.add(lo + "|" + hi + "|" + (owner == Board.YPLAYER ? "Y" : "X"));
                    }
        System.out.println("PLY " + m.getMoveNr() + " mover=- move=-"
            + " next=" + (m.getNextPlayer() == Board.YPLAYER ? "Y" : "X")
            + " moveNr=" + m.getMoveNr()
            + " termY=" + bY.checkGameOver() + " termX=" + bX.checkGameOver()
            + " pegs=" + pegs.size() + " bridges=" + br.size());
        System.out.println("  PEGS " + String.join(" ", pegs));
        System.out.println("  BRIDGES " + String.join(" ", br));
        StringBuilder h = new StringBuilder();
        for (int i = 1; i <= m.getMoveNr(); i++) {
            if (h.length() > 0) h.append(' ');
            h.append(m.getMoveX(i)).append(',').append(m.getMoveY(i));
        }
        System.out.println("  HIST " + h);
        int toMove = m.getNextPlayer();
        StringBuilder lg = new StringBuilder();
        for (int x = 0; x < n; x++)
            for (int y = 0; y < n; y++)
                lg.append(bY.pinAllowed(x, y, toMove) ? '1' : '0');
        System.out.println("  LEGAL " + lg);
    }

    private static void post(Throwable thrown) {
        int w = windows(), f = frames();
        boolean headless = java.awt.GraphicsEnvironment.isHeadless();
        boolean prefsOk = plistBefore != null && plistBefore.equals(plistHash())
                          && countBefore == prefsCount();
        Set<String> names = new LinkedHashSet<String>(REFLECTED);
        boolean reflOk = !REFLECTED.isEmpty()
            && new LinkedHashSet<String>(AUTHORIZED).containsAll(names);
        req(thrown == null, "no unexpected exception");
        req(w == 0 && f == 0, "zero Window/Frame at end");
        req(headless, "headless at end");
        req(prefsOk, "preference surfaces unchanged");
        req(reflOk, "only authorized reflective fields used");
        System.out.println("POSTCOND no_throw=" + (thrown == null) + " windows=" + w
            + " frames=" + f + " headless=" + headless + " prefs_ok=" + prefsOk
            + " refl_ok=" + reflOk + " refl_n=" + REFLECTED.size()
            + " failures=" + failures);
    }
}
