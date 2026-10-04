"""H5 incumbent tactics (EXPLORATORY) -- card §1-§9 with §10 A1.

SYNTHETIC ONLY: no model is loaded, the real incumbent makes no decision, and the real corpus
is not frozen. The tactical rules run on rules-engine positions with known answers (small
boards, found by random playout and pinned here -- each test re-checks its position) and on
hand-built trees of real engine states; the decision path is the real frozen search driven by
the steered stub evaluator. Messages fixed.
"""
import contextlib
import copy
import dataclasses
import json
import os
import pathlib
import random
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_defence as D
from scripts.GPU.alphazero import h5_incumbent_defence_factors as X
from scripts.GPU.alphazero import h5_incumbent_tactics as M
from scripts.GPU.alphazero import h5_incumbent_tactics_authorization as AUTH
from scripts.GPU.alphazero import h5_weakness_review as WR
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move as enc
from tests.test_h5_cap_completion import ONE, TWO
from tests.test_h5_incumbent_defence import BY_ID, CONFIG, FIX, IDENT, Steered

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: 8x8 positions found by random playout and pinned; each test re-checks the property it needs
OWN_WIN_AND_THREAT = [(1, 1), (1, 4), (7, 3), (3, 4), (7, 2), (2, 3), (3, 1), (3, 6), (5, 4),
                      (5, 7), (4, 3), (2, 7), (5, 1), (5, 5), (0, 2), (5, 2), (2, 6), (1, 7),
                      (0, 5), (6, 1), (3, 5), (4, 6), (6, 6), (1, 6), (6, 3), (3, 0)]
ONE_THREAT_FIVE_DEFENCES = [(0, 2), (5, 1), (3, 3), (6, 3), (1, 1), (3, 1), (6, 4), (4, 2),
                            (2, 1), (4, 3), (2, 6), (6, 2), (7, 5)]
TWO_THREATS_ONE_DEFENCE = [(5, 4), (3, 0), (6, 1), (2, 4), (4, 6), (2, 0), (4, 4), (1, 7),
                           (4, 3), (5, 3), (3, 4), (1, 4), (2, 6), (5, 7), (0, 5)]
THREE_THREATS_NO_DEFENCE = [(4, 5), (5, 4), (5, 1), (4, 4), (6, 1), (6, 2), (2, 1), (2, 4),
                            (6, 5), (5, 3), (1, 6), (1, 7), (6, 3), (3, 7), (1, 4), (2, 0),
                            (7, 6), (1, 2), (5, 6), (3, 5), (4, 3), (5, 7), (4, 2), (5, 2),
                            (0, 6), (5, 0), (0, 1), (4, 1), (0, 4), (1, 1), (0, 3), (2, 6)]


#: an 8x8 legal full-board DRAW (no winner, no legal move); random playout seed 2386
DRAW8 = [(6, 3), (4, 2), (3, 3), (2, 3), (6, 2), (1, 3), (2, 4), (4, 7), (6, 6), (3, 1), (6, 4), (4, 5),
         (3, 2), (1, 0), (4, 4), (2, 0), (3, 5), (1, 7), (1, 2), (3, 7), (1, 6), (6, 5), (2, 5), (4, 0),
         (0, 5), (5, 4), (1, 5), (6, 1), (4, 1), (3, 0), (2, 2), (6, 7), (5, 5), (5, 7), (0, 2), (5, 1),
         (3, 4), (5, 0), (5, 3), (6, 0), (5, 6), (3, 6), (7, 6), (1, 4), (5, 2), (1, 1), (2, 1), (2, 6),
         (0, 1), (2, 7), (4, 6), (4, 3), (7, 3)]
A24 = "abcdefghijklmnopqrstuvwx"


def game24(code):
    """A pinned 24x24 game: two letters (row, col) per move."""
    return [(A24.index(code[i]), A24.index(code[i + 1])) for i in range(0, len(code), 2)]


#: OFFICIAL-BOARD positions from random legal playouts (scratch finders, streams noted), pinned
#: here; each is re-checked in test_the_24x24_POSITIONS_still_have_their_properties
DRAW24 = game24(  # Random("draw24:0"): ply 570, no winner, red to move with NO legal move
    "wgvwshopanntwsokcslgjlgxqloqqsdodpgcpbwwfbkeidedbnrqmpkdxieupnlvonijogiqqjixqelwajqdrvkp"
    "cdlsofduxpkjnhwqitrbgjdcqhmtumirvckthuwknebomiouhjqtlqnbtumcbtknpdcamntjxocqifsenifiuipg"
    "dednskqmaigfjnpcjkfofkwewcwvpmfqckfhlkgeruqatsmjamtgmrwdmvdrxhmmjsvxdsmfhrvgmuiigkwmvbbf"
    "vrnnxwrdgnucliflstvuvvtmxrsuxvgdehoxgmbbdwghiscmswgqtbjqrwexodcrrtchxgeiooolukgrpkutpewb"
    "xqwhssjagbepjgbaklnfmkhlpqudlbwrjccohcnsvsfphvrnteqxsbbuuerrgutrldcjmwdimdrmkspojvwpsown"
    "dfiprcsgvpdlapelvmtwkqcnglvflfsnwjfnkcdaprhgkborcphbgwbkkvefsqhfcfuvlrsfrjrealkftdroqfgi"
    "mguhsdebfuuqaclpxtgshtrsostfdqloicekjtslkunoatrfkimecthawiwtfmianudgakmsegscxciwotlxfsio"
    "qrohupugetmhhwmqafvljofeahenavhdxuqbdbtncbdjomoeaguxqoilthltuunadmnrdvubiuqvovtiqgffjwbl"
    "rkkgrifcggowaobmphbvbsvthqemsivqtcsjccllqnnqpwpseembjiqcjmpjndranmojrhsahpnvrlnklunloiml"
    "kkevhmnxpiocnwimhiqkdkivmowfujwuquttnppvledtusgtaejrcvjelcbxbehxhnuavdkxurrxfjinvewaerbg"
    "fteanjjbulkhadrgxkljtlunjpptecfvkobdfdihxdqpvhvolhtabjcxlnhkaupxvnfgtvjuxbbrddcwcuibxlfw"
    "bcuwpltpbijfbwigqwhhwobpbqlaobuoxjpangpfviufsvoadhgpsphevksmqiejawsrwlhsvjjdiesxtkkreqka"
    "eofaesqqxebhfrewaqjhnctoarpplmkwabwxtqkmxmmxhocgcefxasjjxndxclgvcigoxsmapugaxfikrpva")
OWN_WIN_279 = game24(  # Random("draw24:5"): ply 279 = black, the last move the cap allows; own win (6,19)
    "riojajgvcdtndihepijpwswaxtfrmdsuxlslvffoqkcijbubjcfatfsgwfqfsjwjnilwqlbvpgecfwqsswpcakcg"
    "xmpffebkwmwuvrscvuhjhihgwpkehbclepbgqnfxorcsxwpdvpemjtkjwwvbcqnwiougaggptutarsvvglmajkdm"
    "mlmbtlvdhogcehccqmlbxphaeidhdtsredfpuqgqhdilcmlvgjfuhmcausiuudgwsbofumrvcopepqrkjjjlafnr"
    "chlagilmrdqjowbscnwltgvjdocviigkmkfhwtkadcwibikcesibjggxoifbkibcirwhkgndovhqopbhdnnxdptm"
    "ogpklulouoefrwkovwtpljkbxhhnqhuhnfiqgdfmblqefqghsesmjhdleeuagmexinlptwrqduolgohpmompvkbq"
    "fkdfohpuhtdaftgrisprigeksnmjukskjeggjfotmwkdxejvfntenhdgnojxlqcwaeqaxrfjqvcfrtfignrumvue"
    "vooqidealfwbplrpbfmqipmcocvmbd")
THREAT_279 = game24(  # Random("draw24:63"): ply 279, black to move; ONE red threat (7,3), 6 defences
    "jkdxccfjifiqpushgsdwqsfqegqxwkfttvbckgtfrkhadofnlsgjsrrqitmxmtrrxvjtpqgohkurjgoxeflcwosp"
    "ulhrsleqxkcqiohoeeffdfnopwropeuuvelganewoudkrmqocnhqcbunlttjxmmcicrxnpesjsdvqpwinvpiogdt"
    "kdcoaqeltbptdgnbqdsvvcqcoisjskoluotuqgkxdemoghbpntcrcpbixfhptthgfltxxdluvtbvwbtlidomnjvk"
    "lklrcsrfamviauojdpljosbsfrtcnhtevghtrtixciqadcjcstmktdlxmmdhbublcfhemhmfvucmqrbmnfsqeuvq"
    "dqgierbtmbjqqjnmxeplxrlebrecutudevuvjnmauwfbiwhwddvogrvxatqtmqhnbdibngoftqprvvekqejoklpd"
    "xqceagwrgpbaxstnkqvnsopgetiiuithhibomggtafkirlfvjjwxsdgevfrwwhpvbecuaiuescgqpkpofodiieup"
    "glfaepnatwtrhjkwfcgainiahhfhkf")
UNDEFENDABLE_256 = game24(  # Random("undef24:5"): ply 256, red to move; 4 black threats, no defence
    "qricgfkjsrjiacqsewdtldhwavbppcpwdpjfnivqlweaivsswciecfmwbklvdgslwgkwpnjqxnotxwgxtefguumj"
    "unoolcvjbvclairlwlembnnbhcokhinxjcqghbhuxqstrocnmpmfupconsqoutvaqekewermlmjwihoekhhsrbvb"
    "ehrrglugnjvsckjpeuobipbilpdxwhtjxinhptowitcifemshvequcvuqvlfjlgexuncshqtpkvfmdnufpkkmvrf"
    "wvlgccndphmqxbuxgspmgpbaenkowbwnagsmbgfrqnummnjretgtngdheikmdkdeedsofftsijsiuwexsjguxsqw"
    "scvnluvtpforcdkulegrdcesmeibtfjmawhxnnuvjdrvlsuartdfftwuxllrdrejkfepvvvxarntwksatofjwrog"
    "qdtmljpgktmaspcrfnghvlkpxjqmqidnvrpusgcphkpbnewpimnoinrcdjfuxoosquhohjja")


def small(moves):
    return M.replay(moves, size=8)


def other(state, avoid=()):
    """The first legal move (row, col) outside `avoid` -- a legal non-defending move."""
    avoid = {tuple(m) for m in avoid}
    return next(m for m in sorted(state.legal_moves()) if m not in avoid)


def counts_of(state, visits=None):
    visits = visits or {}
    return {m: visits.get(m, 0) for m in state.legal_moves()}


# ─────────────────────────────── the gate and the frozen names ───────────────────────────────

def test_the_gate_is_false_as_published():
    assert AUTH.H5_INCUMBENT_TACTICS_AUTHORIZED is False


@pytest.mark.parametrize("entry", ["freeze_corpus", "run", "run_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch):
    for owner, name in ((M, "_derive_manifest"), (M, "_committed_manifest"), (D, "_frozen"),
                        (D, "_identity"), (M, "_claim"), (CC, "_watch"), (os, "mkdir"),
                        (os, "makedirs"), (R4, "load_production_evaluator")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(M.TacticsUnauthorized):
        getattr(M, entry)()


def test_the_CLI_exits_5_and_takes_no_output_directory(tmp_path):
    r = subprocess.run([sys.executable, "-m", M.MODULE, "run"], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == M.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", M.MODULE, "run", "--out-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr


def test_the_FROZEN_NAMES_cap_and_excluded_games():
    assert (M.MODULE, M.PLY_CAP, R4.PLY_CAP) == \
        ("scripts.GPU.alphazero.h5_incumbent_tactics", 280, 280)
    assert M.out_dir() == pathlib.Path(R4.EVIDENCE_ROOT) / "t1j-h5-incumbent-tactics-EXPLORATORY"
    assert M.corpus_path() == pathlib.Path(R4.EVIDENCE_ROOT) / \
        "t1j-h5-incumbent-tactics-corpus-EXPLORATORY" / "corpus.json"
    members = {g for ep in CC.EPISODES.values() for g in ep["members"]}
    assert set(M.EXCLUDED_GAMES) == members | {ct["game"] for ct in CC.CONTROLS.values()}
    assert len(members) == 13 and len(M.EXCLUDED_GAMES) == 18, "two controls share E games"


# ─────────────────────────────── M1: classes and precedence (§3.1) ───────────────────────────

def test_the_24x24_POSITIONS_still_have_their_properties():
    d = M.replay(DRAW24)
    assert (d.ply, d.to_move, d.winner(), d.legal_moves(), d.is_terminal()) == \
        (570, "red", None, [], True)
    w = M.replay(OWN_WIN_279, semantics="protocol")
    assert (w.ply, w.to_move, w.winner(), CC.immediate_wins(w), CC.threats_against(w)) == \
        (279, "black", None, [(6, 19)], [])
    t = M.replay(THREAT_279, semantics="protocol")
    assert (t.ply, t.to_move, CC.immediate_wins(t), CC.threats_against(t)) == \
        (279, "black", [], [(7, 3)])
    assert len(WR.saving_moves_of(t, [(7, 3)])) == 6
    u = M.replay(UNDEFENDABLE_256, semantics="protocol")
    assert (u.ply, u.to_move, CC.immediate_wins(u), len(u.legal_moves())) == (256, "red", [], 286)
    assert CC.threats_against(u) == [(15, 16), (16, 15), (17, 16), (19, 16)]
    assert WR.saving_moves_of(u, CC.threats_against(u)) == []


def test_the_SYNTHETIC_POSITIONS_still_have_their_properties():
    s = small(OWN_WIN_AND_THREAT)
    assert (CC.immediate_wins(s), CC.threats_against(s)) == ([(2, 4)], [(2, 2)])
    s = small(ONE_THREAT_FIVE_DEFENCES)
    assert CC.threats_against(s) == [(5, 4)] and len(WR.saving_moves_of(s, [(5, 4)])) == 5
    s = small(TWO_THREATS_ONE_DEFENCE)
    assert CC.threats_against(s) == [(7, 3), (7, 5)]
    s = small(THREE_THREATS_NO_DEFENCE)
    assert len(CC.threats_against(s)) == 3 and len(s.legal_moves()) == 21


@pytest.mark.parametrize("moves,cls,D_", [
    (OWN_WIN_AND_THREAT, "OWN_WIN", None),
    (ONE_THREAT_FIVE_DEFENCES, "DEFEND_SINGLE", [[2, 4], [3, 5], [4, 4], [5, 4], [5, 5]]),
    (TWO_THREATS_ONE_DEFENCE, "DEFEND_MULTI", [[6, 5]]),
    (THREE_THREATS_NO_DEFENCE, "UNDEFENDABLE", []),
    (ONE_THREAT_FIVE_DEFENCES[:2], "QUIET", []),
])
def test_the_CLASSES_in_precedence_order(moves, cls, D_):
    c = M.classify(small(moves), semantics="protocol")
    assert (c["class"], c["D"]) == (cls, D_)


def test_M1_an_OWN_WIN_takes_precedence_over_defending():
    s = small(OWN_WIN_AND_THREAT)
    r = M.m1_choice(M.classify(s, semantics="protocol"), counts_of(s, {(2, 2): 50, (2, 4): 3}))
    assert (r["baseline"], r["choice"], r["changed"]) == ([2, 2], [2, 4], True)


def test_M1_TWO_OWN_WINS_are_both_admissible_and_the_most_visited_is_chosen():
    s = CC.state_of(TWO)
    c = M.classify(s, semantics="protocol")
    assert c["W"] == [[12, 2], [12, 4]]
    assert M.m1_choice(c, counts_of(s, {(12, 4): 9, (12, 2): 2, (3, 3): 40}))["choice"] == [12, 4]


def test_M1_ALTERNATIVE_DEFENCES_choose_the_most_visited_and_flag_no_evidence():
    s = small(ONE_THREAT_FIVE_DEFENCES)
    c = M.classify(s, semantics="protocol")
    out = other(s, c["D"])
    r = M.m1_choice(c, counts_of(s, {(3, 5): 7, (4, 4): 9, out: 100}))
    assert (r["choice"], r["no_search_evidence"]) == ([4, 4], False)
    r = M.m1_choice(c, counts_of(s, {out: 100}))
    assert (r["choice"], r["no_search_evidence"]) == ([2, 4], True), \
        "an unvisited defence is the (row, col) minimum and must be flagged"


def test_M1_MULTIPLE_THREATS_admit_only_the_common_defence():
    s = small(TWO_THREATS_ONE_DEFENCE)
    c = M.classify(s, semantics="protocol")
    r = M.m1_choice(c, counts_of(s, {other(s, c["D"]): 30}))
    assert r["choice"] == [6, 5]


@pytest.mark.parametrize("moves", [THREE_THREATS_NO_DEFENCE, ONE_THREAT_FIVE_DEFENCES[:2]])
def test_M1_UNDEFENDABLE_and_QUIET_change_nothing(moves):
    s = small(moves)
    some = sorted(s.legal_moves())[3]
    r = M.m1_choice(M.classify(s, semantics="protocol"), counts_of(s, {some: 5}))
    assert (r["choice"], r["changed"]) == (list(some), False)


# ─────────────────────────────── the cap (§10 A1.2) ───────────────────────────────

def test_CAP_a_win_at_the_boundary_counts_and_a_position_at_the_cap_is_not_classified():
    s = small(OWN_WIN_AND_THREAT)                # ply 26
    assert M.classify(s, semantics="protocol", ply_cap=27)["class"] == "OWN_WIN", \
        "the winner is checked before the cap"
    assert M.classify(s, semantics="protocol", ply_cap=26) is None
    assert M.classify(s, semantics="board", ply_cap=26)["class"] == "OWN_WIN"


def test_CAP_a_purported_reply_after_termination_is_void():
    s = small(ONE_THREAT_FIVE_DEFENCES)          # ply 13, one threat
    p = M.classify(s, semantics="protocol", ply_cap=14)
    b = M.classify(s, semantics="board", ply_cap=14)
    assert (p["class"], p["T_void"]) == ("QUIET", [[5, 4]]) and b["class"] == "DEFEND_SINGLE"


def tree_root(state, *, expanded=True):
    root = MCTSNode(state=state, priors={enc(*m): 1.0 for m in state.legal_moves()}
                    if expanded else None)
    return root


def child(parent, move, visits=1, expanded=True):
    st = parent.state.apply_move(move)
    n = MCTSNode(state=st, parent=parent, move=enc(*move), visit_count=visits,
                 priors={enc(*m): 1.0 for m in st.legal_moves()} if expanded and
                 st.winner() is None else None)
    parent.children[enc(*move)] = n
    return n


def test_CAP_a_BOARD_ONLY_proof_never_justifies_a_PROTOCOL_exclusion():
    s = small(ONE_THREAT_FIVE_DEFENCES)          # ply 13; the threat (5,4) wins at ply 15
    root = tree_root(s)
    bad = other(s, M.classify(s, semantics="board")["D"])
    m = child(root, bad, visits=9)               # does not defend
    child(m, (5, 4))                             # the terminal refutation, at ply 15
    counts = counts_of(s, {bad: 9, (2, 4): 3})
    board = M.m2_choice(root, counts, semantics="board", ply_cap=14)
    prot = M.m2_choice(root, counts, semantics="protocol", ply_cap=14)
    assert board["excluded"] == [list(bad)] and board["choice"] == [2, 4]
    assert prot["excluded"] == [] and prot["choice"] == list(bad), \
        "a proof past the cap was used under the protocol"
    assert prot["proven_root_children"] == []


def test_CAP_a_node_AT_the_cap_without_a_winner_is_UNKNOWN():
    s = small(ONE_THREAT_FIVE_DEFENCES)
    root = tree_root(s)
    m = child(root, other(s))
    st = M.proof_statuses(root, semantics="protocol", ply_cap=14)
    assert st[id(m)] == "UNKNOWN"


def test_CAP_24x24_a_win_on_the_LAST_allowed_move_counts_and_nothing_is_decided_AT_the_cap():
    s = M.replay(OWN_WIN_279, semantics="protocol")          # ply 279; the real cap, 280
    assert M.classify(s, semantics="protocol")["class"] == "OWN_WIN"
    leaf = {"kind": "TERMINAL_LOSS", "winner": "black"}
    assert M.verify_certificate(OWN_WIN_279, [(6, 19)], "LOSS", leaf,
                                semantics="protocol") == (True, "verified")
    assert M.verify_certificate(OWN_WIN_279, [], "WIN", {"kind": "WIN", "move": [6, 19], "child": leaf},
                                semantics="protocol") == (True, "verified")
    root = tree_root(s)
    child(root, (6, 19))
    assert M.m2_choice(root, counts_of(s, {(6, 19): 1}), semantics="protocol")["rule"] == "PROVEN_WIN"
    at_cap = OWN_WIN_279 + [other(s, [(6, 19)])]
    a = M.replay(at_cap, semantics="protocol")
    assert (a.ply, a.winner()) == (280, None)
    assert M.classify(a, semantics="protocol") is None and M.classify(a, semantics="board") is not None
    with pytest.raises(M._Bad, match="past the cap 280"):
        M.replay(at_cap + [other(a)], semantics="protocol")
    assert M.replay(at_cap + [other(a)], semantics="board").ply == 281


def threat279_tree(defence_visits=3):
    """THREAT_279: black's non-defending move refuted by red's (7,3) -- a reply made AT the cap."""
    s = M.replay(THREAT_279, semantics="protocol")
    D_ = M.classify(s, semantics="board")["D"]
    bad, dfn = other(s, D_), tuple(D_[0])
    root = tree_root(s)
    m = child(root, bad, visits=9)
    m.value_sum = 4.5                            # q_node +0.5: for RED, to move at the child
    child(m, (7, 3))
    if defence_visits:
        child(root, dfn, visits=defence_visits).value_sum = -0.2 * defence_visits
    return s, root, bad, dfn, counts_of(s, {bad: 9, dfn: defence_visits})


def test_CAP_24x24_a_reply_AT_the_cap_is_void_and_a_BOARD_ONLY_proof_never_excludes():
    s, root, bad, dfn, counts = threat279_tree()
    p, b = M.classify(s, semantics="protocol"), M.classify(s, semantics="board")
    assert (p["class"], p["T_void"], b["class"], b["T"]) == ("QUIET", [[7, 3]], "DEFEND_SINGLE", [[7, 3]])
    board = M.m2_choice(root, counts, semantics="board")
    prot = M.m2_choice(root, counts, semantics="protocol")
    assert (board["excluded"], board["choice"]) == ([list(bad)], list(dfn))
    assert (prot["excluded"], prot["proven_root_children"], prot["choice"]) == ([], [], list(bad)), \
        "a proof resting on a reply AT the cap was used under the protocol"
    cert = board["proven_root_children"][0]["certificate"]
    assert M.verify_certificate(THREAT_279, [bad], "WIN", cert, semantics="board") == (True, "verified")
    ok_, why = M.verify_certificate(THREAT_279, [bad], "WIN", cert, semantics="protocol")
    assert not ok_ and "the protocol cap 280" in why


# ─────────────────────────────── M2: statuses, choices, certificates (§3.2, §10 A1.4) ────────

def refuted_root():
    """ONE_THREAT_FIVE_DEFENCES: a non-defending move loses to (5,4); (2,4) defends."""
    s = small(ONE_THREAT_FIVE_DEFENCES)
    bad = other(s, M.classify(s, semantics="protocol")["D"])
    root = tree_root(s)
    m = child(root, bad, visits=9)
    child(m, (5, 4))
    child(root, (2, 4), visits=3)
    return s, root, bad


def test_M2_a_VISITED_REFUTATION_excludes_its_parent_and_the_certificate_verifies():
    s, root, bad = refuted_root()
    r = M.m2_choice(root, counts_of(s, {bad: 9, (2, 4): 3}), semantics="protocol")
    assert (r["rule"], r["excluded"], r["choice"]) == ("EXCLUSION", [list(bad)], [2, 4])
    p = r["proven_root_children"][0]
    assert (p["move"], p["status"], p["certificate"]["kind"]) == (list(bad), "WIN", "WIN")
    assert M.verify_certificate(ONE_THREAT_FIVE_DEFENCES, [bad], "WIN", p["certificate"],
                                semantics="protocol", size=8) == (True, "verified")
    assert not M.verify_certificate(ONE_THREAT_FIVE_DEFENCES, [bad], "LOSS", p["certificate"],
                                    semantics="protocol", size=8)[0], "a WIN certificate proved a LOSS"


def test_M2_a_PROVEN_WIN_is_chosen():
    s = small(OWN_WIN_AND_THREAT)
    root = tree_root(s)
    child(root, (2, 4), visits=1)                # terminal: the mover wins
    child(root, (2, 2), visits=40)
    r = M.m2_choice(root, counts_of(s, {(2, 4): 1, (2, 2): 40}), semantics="protocol")
    assert (r["rule"], r["choice"], r["root_status"]) == ("PROVEN_WIN", [2, 4], "WIN"), \
        "ONE losing child proves the parent: the rule is existential"
    assert r["proven_root_children"][0]["certificate"] == {"kind": "TERMINAL_LOSS",
                                                           "winner": s.to_move}


def full_refutation_tree(drop=None):
    """THREE_THREATS_NO_DEFENCE: every legal move answered by a winning reply."""
    s = small(THREE_THREATS_NO_DEFENCE)
    root = tree_root(s)
    for mv in s.legal_moves():
        if mv == drop:
            continue
        n = child(root, mv)
        w = CC.immediate_wins(n.state)[0]
        child(n, w)
    return s, root


def test_M2_EVERY_MOVE_LOSES_is_ROOT_PROVEN_LOSS_with_full_coverage():
    s, root = full_refutation_tree()
    st = M.proof_statuses(root, semantics="protocol")
    assert st[id(root)] == "LOSS"
    cert = M.certificate(root, st)
    assert cert["kind"] == "ALL_REPLIES_WIN" and len(cert["legal"]) == 21
    assert M.verify_certificate(THREE_THREATS_NO_DEFENCE, [], "LOSS", cert,
                                semantics="protocol", size=8) == (True, "verified")
    counts = counts_of(s, {m: 1 for m in s.legal_moves()})
    r = M.m2_choice(root, counts, semantics="protocol")
    assert (r["rule"], r["choice"], r["root_status"]) == \
        ("ROOT_PROVEN_LOSS", r["baseline"], "LOSS")


def test_M2_an_ABSENT_legal_child_keeps_the_node_UNKNOWN_and_unvisited_is_never_LOSS():
    drop = other(small(THREE_THREATS_NO_DEFENCE))
    s, root = full_refutation_tree(drop=drop)
    assert drop in s.legal_moves()
    st = M.proof_statuses(root, semantics="protocol")
    assert st[id(root)] == "UNKNOWN", "a node was proven LOSS with a legal child missing"
    counts = counts_of(s, {m: 1 for m in s.legal_moves() if m != drop})
    r = M.m2_choice(root, counts, semantics="protocol")
    assert (r["choice"], r["no_search_evidence"]) == (list(drop), True), \
        "the unvisited move is UNKNOWN, preferred over proven losses, and flagged"


def test_M2_a_NETWORK_VALUE_or_MEAN_of_plus_minus_1_is_not_proof():
    s = small(ONE_THREAT_FIVE_DEFENCES)
    root = tree_root(s)
    m = child(root, other(s, M.classify(s, semantics="protocol")["D"]), visits=50)
    m.nn_value, m.value_sum = 1.0, 50.0          # "certain" -- but not terminal
    g = child(m, other(m.state, [(5, 4)]), visits=49)
    g.nn_value, g.value_sum = -1.0, -49.0
    st = M.proof_statuses(root, semantics="protocol")
    assert (st[id(m)], st[id(g)]) == ("UNKNOWN", "UNKNOWN")


def test_TACTICS_labels_PROTOCOL_and_BOARD_apart_and_flags_cap_sensitivity():
    s, root, bad = refuted_root()                # ply 13; the refutation lands at ply 15
    counts = counts_of(s, {bad: 9, (2, 4): 3})
    t = M.tactics(s, counts, root, ply_cap=14)
    p, b = t["protocol"], t["board"]
    assert (p["class"]["class"], b["class"]["class"]) == ("QUIET", "DEFEND_SINGLE")
    assert (p["M2"]["excluded"], b["M2"]["excluded"]) == ([], [list(bad)])
    assert t["cap_sensitive"] is True
    assert M.tactics(s, counts, root)["cap_sensitive"] is False


def test_VALUES_are_the_ROOT_MOVERs_per_SEMANTICS_and_NULL_when_unvisited():
    s, root, bad, dfn, counts = threat279_tree()
    t = M.tactics(s, counts, root)
    for sem in M.SEMANTICS:
        base = t[sem]["values"]["baseline"]
        assert (base["move"], base["visits"], base["q_node"], base["q_root_mover"]) == \
            (list(bad), 9, 0.5, -0.5), "the value must be the ROOT mover's: +0.5 for red is -0.5 here"
    p, b = t["protocol"]["values"], t["board"]["values"]
    assert p["M1"]["move"] == p["M2"]["move"] == list(bad), "protocol: the threat is void, nothing proven"
    assert b["M1"]["move"] == b["M2"]["move"] == list(dfn)
    assert (b["M2"]["visits"], b["M2"]["q_root_mover"]) == (3, pytest.approx(0.2))
    s, root, bad, dfn, counts = threat279_tree(defence_visits=0)
    v = M.tactics(s, counts, root)["board"]["values"]["M1"]
    assert (v["move"], v["visits"], v["q_node"], v["q_root_mover"]) == (list(dfn), 0, None, None), \
        "an ABSENT child is null, never 0"
    child(root, dfn, visits=0)
    v = M.tactics(s, counts, root)["board"]["values"]["M1"]
    assert (v["visits"], v["q_root_mover"]) == (0, None), "a created child with 0 visits is null, never 0"


def test_VALUES_through_the_REAL_search_a_visited_own_win_is_plus_1_for_the_root_mover():
    row = M.observe("corpus", build(), CONFIG, {"id": "w", "moves": TWO})
    for sem in M.SEMANTICS:
        t = row["tactics"][sem]
        assert (t["M1"]["choice"], t["M2"]["rule"]) == ([12, 2], "PROVEN_WIN")
        v = t["values"]["M1"]
        assert v["visits"] > 0 and (v["q_node"], v["q_root_mover"]) == (-1.0, 1.0)


# ─────────────────────────────── certificate tampering (§10 A1.4) ───────────────────────────────

@pytest.fixture
def loss_cert():
    s, root = full_refutation_tree()
    return M.certificate(root, M.proof_statuses(root, semantics="protocol"))


def ok(cert, claim="LOSS", path=(), cap=280):
    return M.verify_certificate(THREE_THREATS_NO_DEFENCE, list(path), claim, cert,
                                semantics="protocol", ply_cap=cap, size=8)[0]


def test_TAMPER_a_missing_legal_move_is_rejected(loss_cert):
    c = copy.deepcopy(loss_cert)
    gone = c["legal"].pop()
    c["children"].pop(f"{gone[0]},{gone[1]}")
    assert ok(loss_cert) and not ok(c)


def test_TAMPER_a_wrong_legal_LIST_alone_is_rejected(loss_cert):
    c = copy.deepcopy(loss_cert)
    c["legal"] = c["legal"][:-1]                 # the children still cover every move
    assert not ok(c)


def test_REPLAY_refuses_a_winning_move_made_AT_the_cap():
    s = small(OWN_WIN_AND_THREAT)                # ply 26; (2,4) wins
    cert = {"kind": "TERMINAL_LOSS", "winner": s.to_move}
    args = (OWN_WIN_AND_THREAT, [(2, 4)], "LOSS", cert)
    assert M.verify_certificate(*args, semantics="board", ply_cap=26, size=8)[0]
    assert M.verify_certificate(*args, semantics="protocol", ply_cap=27, size=8)[0], \
        "a win on the last allowed move is a win"
    assert not M.verify_certificate(*args, semantics="protocol", ply_cap=26, size=8)[0], \
        "a move at the cap was replayed"
    assert not M.verify_certificate(OWN_WIN_AND_THREAT, [(2, 4)], "WIN", cert, semantics="board",
                                    size=8)[0], "a TERMINAL_LOSS certificate proved a WIN"


def test_TAMPER_an_extra_move_is_rejected(loss_cert):
    c = copy.deepcopy(loss_cert)
    c["children"]["9,9"] = c["children"][next(iter(c["children"]))]
    assert not ok(c)


def test_TAMPER_an_illegal_winning_move_is_rejected(loss_cert):
    c = copy.deepcopy(loss_cert)
    k = next(iter(c["children"]))
    c["children"][k]["move"] = list(THREE_THREATS_NO_DEFENCE[0])   # an occupied cell
    assert not ok(c)


def test_TAMPER_a_wrong_winner_is_rejected(loss_cert):
    c = copy.deepcopy(loss_cert)
    k = next(iter(c["children"]))
    leaf = c["children"][k]["child"]
    leaf["winner"] = "red" if leaf["winner"] == "black" else "black"
    assert not ok(c)


def test_TAMPER_a_path_past_the_cap_and_a_claim_mismatch_are_rejected(loss_cert):
    ply = small(THREE_THREATS_NO_DEFENCE).ply
    assert not ok(loss_cert, cap=ply + 1), "a reply at the cap was accepted as proof"
    assert not ok(loss_cert, claim="WIN")


@pytest.mark.parametrize("moves,size", [(DRAW24, 24), (DRAW8, 8)], ids=["24x24", "8x8"])
def test_DRAW_a_full_board_without_a_winner_is_never_a_proven_LOSS(moves, size):
    d = M.replay(moves, size=size)
    assert (d.winner(), d.legal_moves(), d.is_terminal()) == (None, [], True)
    empty = {"kind": "ALL_REPLIES_WIN", "legal": [], "children": {}}
    ok_, why = M.verify_certificate(moves, [], "LOSS", empty, semantics="board", size=size)
    assert not ok_ and "a draw: no winner, 0 legal moves" in why, why
    win = {"kind": "WIN", "move": list(moves[-1]), "child": empty}
    assert not M.verify_certificate(moves[:-1], [], "WIN", win, semantics="board", size=size)[0], \
        "the move INTO the draw was proven winning"
    assert not M.verify_certificate(moves, [], "LOSS", {"kind": "TERMINAL_LOSS", "winner": None},
                                    semantics="board", size=size)[0]
    ok_, why = M.verify_certificate(moves, [], "LOSS", empty, semantics="protocol", size=size)
    assert not ok_ and ("past the cap 280" if len(moves) > 280 else "a draw: no winner") in why
    parent = tree_root(M.replay(moves[:-1], size=size))
    leaf = child(parent, moves[-1])
    for sem in M.SEMANTICS:
        st = M.proof_statuses(parent, semantics=sem)
        assert (st[id(parent)], st.get(id(leaf), "UNKNOWN")) == ("UNKNOWN", "UNKNOWN")
        assert M.classify(d, semantics=sem) is None


def test_24x24_a_GENUINE_universal_loss_proof_still_verifies_and_a_missing_reply_does_not():
    s = M.replay(UNDEFENDABLE_256, semantics="protocol")    # replies at ply 257: inside the cap
    T = CC.threats_against(s)
    root = tree_root(s)
    for mv in s.legal_moves():
        n = child(root, mv)
        child(n, next(t for t in T if n.state.is_valid_placement(*t) and
                      n.state.apply_move(t).winner() == "black"))
    for sem in M.SEMANTICS:
        st = M.proof_statuses(root, semantics=sem)
        cert = M.certificate(root, st)
        assert (st[id(root)], cert["kind"], len(cert["legal"])) == ("LOSS", "ALL_REPLIES_WIN", 286)
        assert M.verify_certificate(UNDEFENDABLE_256, [], "LOSS", cert,
                                    semantics=sem) == (True, "verified")
    gone = copy.deepcopy(cert)
    gone["children"].pop(next(iter(gone["children"])))
    assert not M.verify_certificate(UNDEFENDABLE_256, [], "LOSS", gone, semantics="board")[0]


# ─────────────────────────────── the conclusion: verification and structural rules ───────────

def real_board_row(defence_visits=3):
    """A 24x24 row: ONE[:23] (black to move, red threatens (12,4)); a non-defending
    move refuted by the terminal (12,4) -- proven under both semantics; one defence visited."""
    s = CC.state_of(ONE[:23])
    D_ = M.classify(s, semantics="protocol")["D"]
    bad, dfn = other(s, D_), tuple(D_[0])
    root = tree_root(s)
    m = child(root, bad, visits=9)
    m.value_sum = 4.5
    child(m, (12, 4))
    if defence_visits:
        child(root, dfn, visits=defence_visits).value_sum = -0.2 * defence_visits
    counts = counts_of(s, {bad: 9, dfn: defence_visits})
    return {"kind": "corpus", "id": "t", "moves": [list(x) for x in ONE[:23]], "move": list(bad),
            "repeat_of": None, "tactics": M.tactics(s, counts, root)}


def test_VERIFY_ROWS_replays_every_certificate_and_STOPS_on_a_tampered_one():
    row = real_board_row()
    assert M.verify_rows([row]) == 2, "one proven root child under each semantics"
    bad = copy.deepcopy(row)
    leaf = bad["tactics"]["protocol"]["M2"]["proven_root_children"][0]["certificate"]["child"]
    leaf["winner"] = "black"
    with pytest.raises(M.TacticsStop, match="t protocol: the certificate for"):
        M.verify_rows([bad])


@pytest.mark.parametrize("breakage,problem", [
    (lambda t: t["protocol"]["M1"].update(choice=[0, 1], changed=True), "non-admissible"),
    (lambda t: t["protocol"]["M2"].update(excluded=[[3, 3]]), "excluded a move not proven"),
    (lambda t: t["protocol"]["M2"].update(changed=True, rule="UNCHANGED",
                                          baseline=[3, 3]), "changed a move without a proof"),
])
def test_INTERPRET_STOPS_on_each_structural_violation(breakage, problem):
    row = real_board_row()
    assert M.interpret([row])["verdict"] == "the structural rules hold"
    bad = copy.deepcopy(row)
    breakage(bad["tactics"])
    with pytest.raises(M.TacticsStop, match=problem):
        M.interpret([bad])


def test_INTERPRET_a_QUIET_move_changed_by_M1_is_a_violation():
    s = small(ONE_THREAT_FIVE_DEFENCES[:2])
    c = M.classify(s, semantics="protocol")
    row = {"kind": "corpus", "id": "q", "moves": [], "move": [1, 1], "tactics": {
        sem: {"class": c, "M1": {"choice": [1, 1], "changed": True, "baseline_admissible": True,
                                 "no_search_evidence": False},
              "M2": {"excluded": [], "changed": False, "rule": "UNCHANGED", "baseline": [1, 1],
                     "choice": [1, 1], "proven_root_children": []}} for sem in M.SEMANTICS}}
    row["tactics"]["cap_sensitive"] = False
    with pytest.raises(M.TacticsStop, match="M1 changed a QUIET move"):
        M.interpret([row])


def test_INTERPRET_reports_ROOT_MOVER_values_only_where_BOTH_moves_were_visited():
    seen, unseen = real_board_row(), {**real_board_row(defence_visits=0), "id": "u"}
    tally = M.interpret([seen, unseen])["protocol_tally"]["DEFEND_SINGLE"]
    assert tally["distinct_positions"] == 2
    for mech in ("M1", "M2"):
        (pair,) = tally["values_where_both_visited"][mech]
        assert (pair["id"], pair["changed"]) == ("t", True)
        assert (pair["baseline_q_root_mover"], pair["choice_q_root_mover"]) == \
            (-0.5, pytest.approx(0.2)), "baseline then choice, the ROOT mover's"


def test_the_PLAN_runs_B_REPRO_first_and_each_DISTINCT_corpus_state_once():
    manifest = {"selected": [{"key": "a", "moves": [], "repeat_of": None},
                             {"key": "b", "moves": [], "repeat_of": "a"},
                             {"key": "c", "moves": [], "repeat_of": None}]}
    plan = M.plan_of(FIX, manifest)
    assert [k for k, _ in plan] == ["B-repro"] * 5 + ["corpus"] * 2
    assert [p["id"] for k, p in plan if k == "corpus"] == ["a", "c"]


# ─────────────────────────────── the corpus (§10 A1.1) ───────────────────────────────

def synthetic_entries(spec):
    """spec: {class: {game: [plies]}} -> entries with distinct digests."""
    out = []
    for cls, games in spec.items():
        for g, plies in games.items():
            for p in plies:
                out.append({"key": f"{g}@{p}", "game": g, "ply": p, "moves": [],
                            "state_digest": f"{g}:{p}", "class_protocol": {"class": cls}})
    return out


def test_SELECTION_is_one_per_game_seeded_sorted_and_reproducible():
    spec = {"QUIET": {f"g{i:02d}": [10 + i, 20 + i, 30 + i] for i in range(30)}}
    a, b = M.select_corpus(synthetic_entries(spec)), M.select_corpus(synthetic_entries(spec))
    assert a == b, "selection is not reproducible"
    sel = [e for e in a["selected"] if e["class"] == "QUIET"]
    assert len(sel) == 20 and len({e["game"] for e in sel}) == 20
    assert [(e["game"], e["ply"]) for e in sel] == sorted((e["game"], e["ply"]) for e in sel)
    games = sorted(spec["QUIET"])
    want = sorted(random.Random(f"{M.CORPUS_SEED}:QUIET").sample(games, 20))
    assert [e["game"] for e in sel] == want
    for e in sel:
        assert e["ply"] == random.Random(f"{M.CORPUS_SEED}:QUIET:{e['game']}").choice(
            sorted(spec["QUIET"][e["game"]]))


def test_SELECTION_a_SHORTAGE_takes_every_game_once_and_is_recorded():
    spec = {"DEFEND_MULTI": {"gA": [5, 7, 9], "gB": [11]}}
    r = M.select_corpus(synthetic_entries(spec))
    sel = [e for e in r["selected"] if e["class"] == "DEFEND_MULTI"]
    assert sorted(e["game"] for e in sel) == ["gA", "gB"], "a game gave two positions"
    assert r["quota_shortfall"]["DEFEND_MULTI"] == 18 and r["quota_shortfall"]["QUIET"] == 20
    assert r["universe"]["DEFEND_MULTI"] == {"positions": 4, "games": 2}


def test_SELECTION_REPEATED_states_are_decided_once_and_listed():
    entries = synthetic_entries({"QUIET": {"gA": [5], "gB": [8]}, "OWN_WIN": {"gC": [3]}})
    for e in entries:
        e["state_digest"] = "same"
    r = M.select_corpus(entries)
    first = [e for e in r["selected"] if e["repeat_of"] is None]
    assert len(first) == 1 and first[0]["class"] == "OWN_WIN", "class order decides first"
    assert r["repeats"] == {"same": ["gC@3", "gA@5", "gB@8"]}


def game_of(task_id, moves):
    return {"task_id": task_id, "moves": [tuple(m) for m in moves],
            "actors": ["t1j" if i % 2 == 0 else "incumbent" for i in range(len(moves))]}


def test_UNIVERSE_excludes_fixture_games_positions_at_the_cap_and_fixture_equivalents():
    games = [game_of("keep", ONE + [(12, 4)]), game_of("h5s0-21-B", ONE)]
    fx = {"FAKE": M.state_digest(CC.state_of(ONE[:5]))}
    entries, feq = M.corpus_universe(games, excluded=["h5s0-21-B"], fixtures=fx, ply_cap=20)
    assert {e["game"] for e in entries} == {"keep"}
    assert all(e["ply"] < 20 and e["ply"] % 2 == 1 for e in entries), "not decision positions"
    assert [(e["key"], e["equivalent_to"]) for e in feq] == [("keep@5", "FAKE")]
    assert "keep@5" not in {e["key"] for e in entries}


def test_FREEZE_CORPUS_through_the_public_entry_writes_once(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "H5_INCUMBENT_TACTICS_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(WR, "load_and_verify", lambda: ([game_of("keep", ONE + [(12, 4)]),
                                                        game_of("h5s0-21-B", ONE)], {}))
    path = pathlib.Path(M.freeze_corpus())
    m = json.loads(path.read_text())
    assert m["selected"] and {e["game"] for e in m["selected"]} == {"keep"}, \
        "a fixture/control source game reached the corpus"
    assert path.read_bytes() == M.manifest_bytes(m) and m["seed"] == 20261003
    assert m["excluded_games"] == list(M.EXCLUDED_GAMES) and set(m["fixture_digests"]) == \
        {"E1", "E2", "E3", "CT1", "CT2", "CT3", "CI1", "CI2", "CD1", "CD2"}
    with pytest.raises(M.TacticsRefused, match="occupied"):
        M.freeze_corpus()


# ─────────────────────────────── one position observed (§10 A1.3) ───────────────────────────

def build(ev=None, config=CONFIG):
    return D.study_builder(config, ev or Steered(), D.tasks_for(IDENT))


@pytest.fixture(scope="module")
def stub_pins():
    pins = {}
    for p in FIX:
        r = D.decide(build(), CC.state_of(p["moves"]), capture=False)
        pins[p["id"]] = D.fingerprint(r["root"], r["counts"], r["root_value"])
    return pins


def test_OBSERVE_a_CORPUS_position_needs_no_in_game_move():
    row = M.observe("corpus", build(), CONFIG, {"id": "c1", "moves": BY_ID["E1"]["moves"],
                                                "recorded": (0, 0)})
    assert "B-repro move" not in row["checks"] and "tactics agree off/on" in row["checks"]
    t = row["tactics"]
    assert t["protocol"]["class"]["class"] == "DEFEND_SINGLE" and not t["cap_sensitive"]


def test_OBSERVE_B_REPRO_STOPS_on_the_real_pins_under_the_stub():
    with pytest.raises(M.TacticsStop, match="B-repro E1: check 'B-repro fingerprint'"):
        M.observe("B-repro", build(), CONFIG, BY_ID["E1"])
    with pytest.raises(M.TacticsStop, match="B-repro E1: check 'B-repro move'"):
        M.observe("B-repro", build(), CONFIG, {**BY_ID["E1"], "recorded": (0, 1)})


def test_OBSERVE_STOPS_on_a_config_change():
    other = dataclasses.replace(CONFIG, mcts_sims=400)   # equal: no diff
    assert M.observe("corpus", build(config=other), CONFIG, BY_ID["CD1"])["checks"]

    def leaky(colour, capture):
        a = build()(colour, capture)
        a.mcts.config = dataclasses.replace(a.mcts.config, c_puct=1.6)
        return a
    with pytest.raises(M.TacticsStop, match="check 'configuration \\(off\\)'"):
        M.observe("corpus", leaky, CONFIG, BY_ID["CD1"])


def test_OBSERVE_STOPS_on_a_root_budget_other_than_400_and_on_the_wrong_seed(monkeypatch):
    real = D.decide

    def short(*a, **k):
        r = real(*a, **k)
        r["root"].visit_count -= 1
        return r
    monkeypatch.setattr(D, "decide", short)
    with pytest.raises(M.TacticsStop, match="check 'budget \\(off\\)'"):
        M.observe("corpus", build(), CONFIG, BY_ID["CD1"])
    monkeypatch.setattr(D, "decide", real)

    def reseeded(colour, capture):
        a = build()(colour, capture)
        a.seed = D.SEED + 1
        return a
    with pytest.raises(M.TacticsStop, match="check 'seed and colour \\(off\\)'"):
        M.observe("corpus", reseeded, CONFIG, BY_ID["CD1"])


def test_OBSERVE_STOPS_when_the_TACTICS_change_a_tree_or_draw_a_number(monkeypatch):
    real = M.tactics

    def touching(state, counts, root, **k):
        root.nn_value += 1e-9
        return real(state, counts, root, **k)
    monkeypatch.setattr(M, "tactics", touching)
    with pytest.raises(M.TacticsStop, match="tactics left the tree unchanged \\(off\\)"):
        M.observe("corpus", build(), CONFIG, BY_ID["CD1"])
    LAST = []
    real_decide = D.decide
    monkeypatch.setattr(D, "decide", lambda *a, **k: (LAST.append(real_decide(*a, **k)),
                                                     LAST[-1])[1])

    def drawing(state, counts, root, **k):
        LAST[-1]["agent"].mcts.rng.random()
        return real(state, counts, root, **k)
    monkeypatch.setattr(M, "tactics", drawing)
    with pytest.raises(M.TacticsStop, match="tactics drew from no stream"):
        M.observe("corpus", build(), CONFIG, BY_ID["CD1"])


# ─────────────────────────────── the child's run (§8) ───────────────────────────────

@pytest.fixture
def stub_predictions(monkeypatch, stub_pins):
    """The §4 predictions are derived from the REAL trees; under the stub, the predictions
    are whatever the stub's trees give -- read once, then frozen for the run."""
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    m1, m2, none = {}, {}, set()
    for p in FIX:
        t = M.observe("B-repro", build(), CONFIG, p)["tactics"]["protocol"]
        m1[p["id"]], m2[p["id"]] = tuple(t["M1"]["choice"]), tuple(t["M2"]["choice"])
        if t["M1"]["no_search_evidence"]:
            none.add(p["id"])
    monkeypatch.setattr(M, "FIXTURE_M1", m1)
    monkeypatch.setattr(M, "FIXTURE_M2", m2)
    monkeypatch.setattr(M, "FIXTURE_M1_NO_EVIDENCE", none)


def claimed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    d = tmp_path / M.OUT_NAME
    M._claim(d, {})
    return d


def log_steps(d):
    return [json.loads(line)["step"] for line in (d / "log.jsonl").read_text().splitlines()]


PLAN = [("B-repro", p) for p in FIX] + [
    ("corpus", {"id": "c-E2", "moves": BY_ID["E2"]["moves"]}),
    ("corpus", {"id": "c-CD2", "moves": BY_ID["CD2"]["moves"]})]


def test_RUN_COMPLETE_records_verifies_and_interprets(tmp_path, stub_predictions):
    d = claimed(tmp_path)
    rec = M._run(PLAN, CONFIG, IDENT, Steered, d)
    assert (rec["status"], rec["failure"], rec["interpretable"]) == ("COMPLETE", None, True)
    assert len(rec["rows"]) == 7 and log_steps(d) == ["model_load"] + ["observation"] * 7
    assert rec["interpretation"]["verdict"] == "the structural rules hold"
    assert rec["certificates_verified"] is not None


def test_RUN_the_REAL_fixture_predictions_STOP_under_the_stub(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    rec = M._run(PLAN, CONFIG, IDENT, Steered, claimed(tmp_path))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "interpretation")
    assert "not the frozen prediction" in rec["failure"]["reason"]
    assert len(rec["rows"]) == 7 and rec["interpretation"] is None


@pytest.mark.parametrize("which", ["FIXTURE_M1", "FIXTURE_M2", "FIXTURE_M1_NO_EVIDENCE"])
def test_RUN_ONE_wrong_fixture_prediction_alone_STOPS(tmp_path, monkeypatch, stub_predictions,
                                                       which):
    if which == "FIXTURE_M1_NO_EVIDENCE":
        monkeypatch.setattr(M, which, M.FIXTURE_M1_NO_EVIDENCE ^ {"E1"})
    else:
        monkeypatch.setitem(getattr(M, which), "E1", (23, 23))
    rec = M._run(PLAN, CONFIG, IDENT, Steered, claimed(tmp_path))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "interpretation")
    assert "E1:" in rec["failure"]["reason"]


def test_RUN_an_UNVERIFIED_certificate_STOPS_keeping_every_row(tmp_path, monkeypatch,
                                                               stub_predictions):
    def tampered(rows):
        raise M.TacticsStop("x the certificate for [0, 0] does not verify: injected")
    monkeypatch.setattr(M, "verify_rows", tampered)
    d = claimed(tmp_path)
    rec = M._run(PLAN, CONFIG, IDENT, Steered, d)
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "verification")
    assert len(rec["rows"]) == 7 and log_steps(d) == ["model_load"] + ["observation"] * 7


def test_RUN_an_INTERPRETATION_exception_is_a_recorded_STOP(tmp_path, monkeypatch,
                                                           stub_predictions):
    monkeypatch.setattr(M, "interpret", lambda rows: (_ for _ in ()).throw(
        ValueError("synthetic interpretation failure")))
    rec = M._run(PLAN, CONFIG, IDENT, Steered, claimed(tmp_path))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "interpretation", "ValueError")


def test_RUN_a_FAILURE_mid_plan_keeps_the_completed_rows(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    plan = PLAN[:5] + [("corpus", {"id": "c-bad", "moves": BY_ID["E1"]["moves"] + [(0, 0)]})]
    d = claimed(tmp_path)
    rec = M._run(plan, CONFIG, IDENT, Steered, d)
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "observe:corpus:c-bad")
    assert len(rec["rows"]) == 5 and log_steps(d) == ["model_load"] + ["observation"] * 5


def test_RUN_INTERRUPTED_INCOMPLETE_and_MODEL_LOAD(tmp_path, monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", stub_pins)
    d = claimed(tmp_path / "i")
    with pytest.raises(KeyboardInterrupt):
        M._run(PLAN, CONFIG, IDENT, lambda: Steered(fail_on_call=2000), d)
    assert json.loads((d / "record.json").read_text())["status"] == "INTERRUPTED"

    def load():
        raise FileNotFoundError("checkpoint not found")
    rec = M._run(PLAN, CONFIG, IDENT, load, claimed(tmp_path / "m"))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "model_load")
    t = [0.0]

    def slow():
        t[0] += 10_000.0
        return Steered()
    rec = M._run(PLAN, CONFIG, IDENT, slow, claimed(tmp_path / "t"), clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "observe:B-repro:E1")
    assert "deadline passed" in rec["failure"]["reason"]


def test_RUN_an_INEXACT_set_of_rows_is_INCOMPLETE(tmp_path, monkeypatch, stub_predictions):
    real = M.observe
    monkeypatch.setattr(M, "observe", lambda kind, b, c, p: {**real(kind, b, c, p),
                                                             "id": "renamed"}
                        if p["id"] == "c-CD2" else real(kind, b, c, p))
    rec = M._run(PLAN, CONFIG, IDENT, Steered, claimed(tmp_path))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("INCOMPLETE", "completion_check", None)
    assert f["missing"] == [["corpus", "c-CD2"]] and f["extra"] == [["corpus", "renamed"]]


# ─────────────────────────────── the parent, the claim, the watchdog ───────────────────────────

def test_the_CHILD_takes_only_an_UNCONSUMED_claim_with_ITS_nonce(tmp_path, monkeypatch):
    d = tmp_path / M.OUT_NAME
    claim, nonce = M._claim(d, {"corpus_sha256": "x"})
    monkeypatch.setenv(M.CLAIM_ENV, "0" * 32)
    with pytest.raises(M.TacticsRefused, match="nonce does not match"):
        M._accept_claim(d)
    monkeypatch.setenv(M.CLAIM_ENV, nonce)
    assert M._accept_claim(d)["corpus_sha256"] == "x"
    (d / "log.jsonl").write_text("")
    with pytest.raises(M.TacticsRefused, match="unconsumed claim"):
        M._accept_claim(d)


@pytest.fixture
def open_gate(monkeypatch, tmp_path, stub_predictions):
    monkeypatch.setattr(AUTH, "H5_INCUMBENT_TACTICS_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: Steered())
    manifest = {"selected": [{"key": "c-E2", "moves": BY_ID["E2"]["moves"], "repeat_of": None}]}
    monkeypatch.setattr(M, "_committed_manifest", lambda rederive: (manifest, "s" * 64))
    return tmp_path


def test_PUBLIC_ENTRY_runs_in_the_real_state(open_gate, monkeypatch):
    def watch(argv, *, limit_s, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:]) == ["-m", M.MODULE, "run-claimed"]
        assert env["PYTHONUNBUFFERED"] == "1"
        monkeypatch.setenv(M.CLAIM_ENV, env[M.CLAIM_ENV])
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = M.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False,
                "cause": None, "signal": None, "rc": rc, "elapsed_s": 1.0,
                "survivors_killed": False}
    monkeypatch.setattr(CC, "_watch", watch)
    res = M.run()
    rec = json.loads((M.out_dir() / "record.json").read_text())
    assert (res["status"], res["record_status"]) == ("COMPLETE", "COMPLETE"), rec["failure"]
    assert len(rec["rows"]) == 6 and res == json.loads((M.out_dir() / M.WATCHDOG_FILE).read_text())
    assert json.loads((M.out_dir() / M.CLAIM_FILE).read_text())["corpus_sha256"] == "s" * 64


def test_PUBLIC_ENTRY_the_REAL_child_reads_the_CLOSED_on_disk_gate(open_gate):
    res = M.run()
    assert (res["status"], res["child_exit"]) == ("EXITED_WITHOUT_RECORD", "exit status 5")
    assert "UNAUTHORIZED" in (M.out_dir() / M.CHILD_STREAMS["stderr"]).read_text()


def test_the_CHILD_refuses_a_corpus_other_than_the_one_its_parent_rederived(open_gate,
                                                                           monkeypatch):
    claim, nonce = M._claim(M.out_dir(), {"corpus_sha256": "t" * 64})
    monkeypatch.setenv(M.CLAIM_ENV, nonce)
    with pytest.raises(M.TacticsRefused, match="not the one the parent re-derived"):
        M.run_claimed()


def test_PUBLIC_ENTRY_refuses_an_UNCOMMITTED_corpus_before_any_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "H5_INCUMBENT_TACTICS_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(CC, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(M.TacticsRefused, match="must be committed before any model load"):
        M.run()
    assert not M.out_dir().exists()


def test_the_PARENT_refuses_a_committed_corpus_that_does_not_REDERIVE(monkeypatch, tmp_path):
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(M.F, "check_durable", lambda paths, _fixture: None)   # durability: above
    M.corpus_path().parent.mkdir(parents=True)
    M.corpus_path().write_bytes(M.manifest_bytes({"selected": []}))
    monkeypatch.setattr(M, "_derive_manifest", lambda: {"selected": []})
    assert M._committed_manifest(rederive=True)[0] == {"selected": []}
    monkeypatch.setattr(M, "_derive_manifest", lambda: {"selected": [{"key": "x"}]})
    with pytest.raises(M.TacticsRefused, match="not the one re-derived"):
        M._committed_manifest(rederive=True)
    assert M._committed_manifest(rederive=False)[0] == {"selected": []}, "the child reads only"


HANG = r'''
import json, os, signal, sys, time
d = sys.argv[1]
open(os.path.join(d, "log.jsonl"), "x").write('{"step": "model_load"}\n')
json.dump({"status": "COMPLETE"}, open(os.path.join(d, "record.json"), "x"))
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


def test_PUBLIC_ENTRY_a_hung_child_is_TERMINATED_never_COMPLETE(open_gate, monkeypatch):
    monkeypatch.setattr(M, "_child_argv", lambda: [sys.executable, "-c", HANG, str(M.out_dir())])
    monkeypatch.setattr(M, "HARD_LIMIT_S", 3.0)
    t0 = time.monotonic()
    res = M.run()
    assert time.monotonic() - t0 < 20
    assert (res["status"], res["record_status"]) == ("TERMINATED", "COMPLETE")
