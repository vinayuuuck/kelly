#!/usr/bin/env python3
# host.py
import json
import random
import socket
import sys
import os
import re
import time
import select
from typing import Dict, Any, Optional, Tuple, List

# ---------- Optional plotting (host saves PNGs) ----------
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ---------- JSON line helpers + buffered recv ----------
def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


class LineBufferedConn:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def recv_json(self) -> Optional[Dict[str, Any]]:
        # Unchanged, blocking version (kept for general use)
        while True:
            nl = self.buf.find(b"\n")
            if nl != -1:
                line = self.buf[:nl]
                self.buf = self.buf[nl + 1 :]
                if not line:
                    continue
                try:
                    return json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            self.buf += chunk

    def recv_json_timeout(
        self, timeout_sec: float
    ) -> Tuple[Optional[Dict[str, Any]], float, bool]:
        """
        Try to read exactly one JSON line within timeout_sec seconds.
        Returns (msg_or_None, elapsed_seconds, timed_out_flag).
        - If timed_out_flag is True, msg_or_None is None and no bytes are consumed beyond what's already in buf.
        - If the peer closes, returns (None, elapsed, False).
        """
        start = time.time()
        # First, see if we already have a full line in the buffer
        nl = self.buf.find(b"\n")
        if nl != -1:
            line = self.buf[:nl]
            self.buf = self.buf[nl + 1 :]
            elapsed = time.time() - start
            if not line:
                return self.recv_json_timeout(max(0.0, timeout_sec - elapsed))
            try:
                return json.loads(line.decode("utf-8")), elapsed, False
            except json.JSONDecodeError:
                # malformed line: skip and keep waiting within remaining time
                return self.recv_json_timeout(max(0.0, timeout_sec - elapsed))

        # Otherwise wait for readability up to timeout
        remaining = timeout_sec
        while remaining > 0:
            t0 = time.time()
            r, _, _ = select.select([self.sock], [], [], remaining)
            waited = time.time() - t0
            remaining = max(0.0, remaining - waited)

            if not r:  # timeout
                elapsed = time.time() - start
                return None, elapsed, True

            chunk = self.sock.recv(4096)
            if not chunk:  # peer closed
                elapsed = time.time() - start
                return None, elapsed, False
            self.buf += chunk

            nl = self.buf.find(b"\n")
            if nl != -1:
                line = self.buf[:nl]
                self.buf = self.buf[nl + 1 :]
                elapsed = time.time() - start
                if not line:
                    # keep looping with whatever time remains
                    continue
                try:
                    return json.loads(line.decode("utf-8")), elapsed, False
                except json.JSONDecodeError:
                    # keep looping with time remaining
                    continue

        # Fell out with no line
        elapsed = time.time() - start
        return None, elapsed, True


class PlayerConn:
    def __init__(self, sock: socket.socket, addr: Tuple[str, int], name: str):
        self.sock = sock
        self.addr = addr
        self.name = name
        self.reader = LineBufferedConn(sock)

    def send(self, msg: Dict[str, Any]) -> None:
        send_json(self.sock, msg)

    def recv(self) -> Optional[Dict[str, Any]]:
        return self.reader.recv_json()

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------- Config loader ----------
def load_match_config(path: str) -> Tuple[int, int, List[float]]:
    """
    File format (4 lines):
      1: INITIAL_CAPITAL_A (int)
      2: INITIAL_CAPITAL_B (int)
      3: N (int) number of rounds
      4: probabilities for P(A wins) — comma and/or space separated
         e.g. "0.52, 0.60,0.53 0.38 0.67,0.55 0.49 0.62, 0.58 0.44"
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 4:
        raise ValueError("Config must contain at least 4 non-empty lines.")

    try:
        capA = int(lines[0])
        capB = int(lines[1])
        N = int(lines[2])
    except Exception as e:
        raise ValueError(f"Failed to parse capitals/N on lines 1–3: {e}")

    # Parse probabilities from line 4: split on commas and/or whitespace
    probs_line = lines[3]
    tokens = [t for t in re.split(r"[,\s]+", probs_line.strip()) if t]
    try:
        probs = [float(t) for t in tokens]
    except Exception as e:
        raise ValueError(f"Failed to parse probabilities on line 4: {e}")

    if len(probs) != N:
        print(
            f"[HOST] Warning: N={N} but parsed {len(probs)} probabilities. "
            f"Using the parsed length {len(probs)}."
        )

    for i, p in enumerate(probs):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability at index {i} out of [0,1]: {p}")

    return capA, capB, probs


# ---------- Accept players ----------
def accept_two_players(port: int) -> Tuple[PlayerConn, PlayerConn]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", port))
    srv.listen(2)
    print(f"[HOST] Waiting for two clients on port {port}...")

    players: List[PlayerConn] = []
    try:
        while len(players) < 2:
            sock, addr = srv.accept()
            reader = LineBufferedConn(sock)
            hello = reader.recv_json()
            if not hello or hello.get("type") != "hello":
                send_json(sock, {"type": "error", "message": "Expected hello."})
                sock.close()
                continue
            name = str(hello.get("player_name", "")).strip()
            if not name or any(p.name == name for p in players):
                send_json(
                    sock, {"type": "error", "message": "Bad or duplicate player_name."}
                )
                sock.close()
                continue
            pc = PlayerConn(sock, addr, name)
            print(f"[HOST] Connected: {name} from {addr}")
            send_json(sock, {"type": "ack", "message": f"Welcome {name}."})
            players.append(pc)
    finally:
        try:
            srv.close()
        except Exception:
            pass
    return players[0], players[1]


def run_single_game(
    probs: List[float],
    conn_A: PlayerConn,
    conn_B: PlayerConn,
    initial_capital_A: int,
    initial_capital_B: int,
    seed: int,
    match_idx: int,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    One game with conn_A as A, conn_B as B.
    Constraints:
      - Per-turn max bet = floor(10% of current capital).
      - Total think-time per player per game = 120 seconds.
        If a player's time runs out before the game ends, they forfeit immediately.
    Returns: (A_final, B_final, rounds_info)
    """
    rng = random.Random(seed)
    capA, capB = initial_capital_A, initial_capital_B
    T = len(probs)
    rounds_info: List[Dict[str, Any]] = []

    # 2-minute budgets per player (independent for each game)
    time_left_A = 120.0
    time_left_B = 120.0

    # Start messages (include total rounds and full probs)
    conn_A.send(
        {
            "type": "game_start",
            "you": "A",
            "your_name": conn_A.name,
            "opponent_name": conn_B.name,
            "initial_capital_you": capA,
            "initial_capital_opp": capB,
            "total_rounds": T,
            "probs": probs,
        }
    )
    conn_B.send(
        {
            "type": "game_start",
            "you": "B",
            "your_name": conn_B.name,
            "opponent_name": conn_A.name,
            "initial_capital_you": capB,
            "initial_capital_opp": capA,
            "total_rounds": T,
            "probs": probs,
        }
    )

    def forfeit(winner_role: str, reason: str):
        nonlocal capA, capB
        if winner_role == "A":
            capA, capB = capA + capB, 0
            winner = "A"
            loser = "B"
        else:
            capB, capA = capB + capA, 0
            winner = "B"
            loser = "A"
        notice = {
            "type": "forfeit",
            "winner": winner,
            "loser": loser,
            "reason": reason,
            "capA": capA,
            "capB": capB,
        }
        conn_A.send(notice)
        conn_B.send(notice)
        result = {
            "type": "game_over",
            "final_capital_A": capA,
            "final_capital_B": capB,
            "name_A": conn_A.name,
            "name_B": conn_B.name,
        }
        conn_A.send(result)
        conn_B.send(result)
        print(
            f"[HOST] FORFEIT: {loser} ({conn_B.name if loser=='B' else conn_A.name}) "
            f"-> {winner} wins. Reason: {reason}. Finals A=${capA} B=${capB}"
        )
        return (capA, capB, rounds_info)

    for k, pk in enumerate(probs):
        if capA <= 0 or capB <= 0:
            break

        # Per-turn 10% caps
        maxA = int(capA * 0.10)
        maxB = int(capB * 0.10)

        # Request bets, include remaining time for this game
        conn_A.send(
            {
                "type": "request_bet",
                "round_index": k,
                "pk": pk,
                "your_capital": capA,
                "opp_capital": capB,
                "role": "A",
                "total_rounds": T,
                "max_bet": maxA,
                "time_left_sec": max(0.0, time_left_A),
            }
        )
        conn_B.send(
            {
                "type": "request_bet",
                "round_index": k,
                "pk": pk,
                "your_capital": capB,
                "opp_capital": capA,
                "role": "B",
                "total_rounds": T,
                "max_bet": maxB,
                "time_left_sec": max(0.0, time_left_B),
            }
        )

        # --- Read A's bet with their remaining time budget
        if time_left_A <= 0.0:
            return forfeit("B", f"A ran out of time at round {k+1}")
        msgA, elapsedA, timedoutA = conn_A.reader.recv_json_timeout(time_left_A)
        time_left_A -= elapsedA
        if timedoutA or msgA is None or msgA.get("type") != "bet":
            return forfeit("B", f"A timed out/disconnected at round {k+1}")

        # --- Read B's bet with their remaining time budget
        if time_left_B <= 0.0:
            return forfeit("A", f"B ran out of time at round {k+1}")
        msgB, elapsedB, timedoutB = conn_B.reader.recv_json_timeout(time_left_B)
        time_left_B -= elapsedB
        if timedoutB or msgB is None or msgB.get("type") != "bet":
            return forfeit("A", f"B timed out/disconnected at round {k+1}")

        # Enforce per-turn 10% cap and usual bounds
        a_bet_req = int(msgA.get("bet", 0))
        b_bet_req = int(msgB.get("bet", 0))
        a_bet = max(0, min(a_bet_req, capA, maxA))
        b_bet = max(0, min(b_bet_req, capB, maxB))
        bet = a_bet + b_bet

        if bet == 0:
            info = {
                "type": "round_result",
                "round_index": k,
                "pk": pk,
                "a_bet": a_bet,
                "b_bet": b_bet,
                "winner": None,
                "transfer": 0,
                "capA": capA,
                "capB": capB,
                "time_left_A_sec": max(0.0, time_left_A),
                "time_left_B_sec": max(0.0, time_left_B),
            }
            conn_A.send(info)
            conn_B.send(info)
            rounds_info.append(
                {
                    "round": k,
                    "pk": pk,
                    "a_bet": a_bet,
                    "b_bet": b_bet,
                    "winner": None,
                    "transfer": 0,
                    "capA": capA,
                    "capB": capB,
                }
            )
            continue

        # Resolve outcome
        if rng.random() <= pk:
            transfer = min(bet, capB)
            capA += transfer
            capB -= transfer
            winner = "A"
        else:
            transfer = min(bet, capA)
            capB += transfer
            capA -= transfer
            winner = "B"

        print(
            f"[HOST] Round {k+1}/{T} pk={pk:.3f} "
            f"A_bet={a_bet} (req {a_bet_req}, max {maxA}, t_left {time_left_A:.2f}s) "
            f"B_bet={b_bet} (req {b_bet_req}, max {maxB}, t_left {time_left_B:.2f}s) "
            f"-> winner {winner}, transfer ${transfer} | caps A={capA} B={capB}"
        )

        info = {
            "type": "round_result",
            "round_index": k,
            "pk": pk,
            "a_bet": a_bet,
            "b_bet": b_bet,
            "winner": winner,
            "transfer": transfer,
            "capA": capA,
            "capB": capB,
            "time_left_A_sec": max(0.0, time_left_A),
            "time_left_B_sec": max(0.0, time_left_B),
        }
        conn_A.send(info)
        conn_B.send(info)

        rounds_info.append(
            {
                "round": k,
                "pk": pk,
                "a_bet": a_bet,
                "b_bet": b_bet,
                "winner": winner,
                "transfer": transfer,
                "capA": capA,
                "capB": capB,
            }
        )

    # Normal game end (no forfeit)
    result = {
        "type": "game_over",
        "final_capital_A": capA,
        "final_capital_B": capB,
        "name_A": conn_A.name,
        "name_B": conn_B.name,
    }
    conn_A.send(result)
    conn_B.send(result)

    print(f"\n=== MATCH {match_idx} FINAL ===")
    print(f"A ({conn_A.name}) = ${capA}")
    print(f"B ({conn_B.name}) = ${capB}\n")
    return capA, capB, rounds_info


# ---------- Plotting helpers ----------
PLAYER_A_COLOR = "#2a9d8f"  # teal
PLAYER_B_COLOR = "#e76f51"  # coral
NEUTRAL_COLOR = "#bdbdbd"  # grey
EDGE_NEUTRAL = "#424242"  # dark grey


def _ensure_out_dir(a_name: str, b_name: str) -> str:
    out_dir = os.path.join("results", f"{a_name}_{b_name}_match")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_match_rounds_plot(
    rounds: List[Dict[str, Any]],
    A_name: str,
    B_name: str,
    match_idx: int,
    out_dir: str,
    kind: str = "transfers",
) -> Optional[str]:
    if not HAVE_MPL:
        print("[HOST] matplotlib not available; skipping plots.")
        return None

    N = len(rounds)
    xs = [r["round"] + 1 for r in rounds]

    if kind == "transfers":
        heights = [int(r["transfer"]) for r in rounds]
        colors = []
        for r in rounds:
            if r["winner"] == "A":
                colors.append(PLAYER_A_COLOR)
            elif r["winner"] == "B":
                colors.append(PLAYER_B_COLOR)
            else:
                colors.append(NEUTRAL_COLOR)

        plt.figure(figsize=(max(6, N * 0.35), 4.5))
        plt.bar(xs, heights, color=colors, edgecolor="black", linewidth=0.6)
        plt.xlabel("Round")
        plt.ylabel("Dollars Transferred to Winner")
        plt.title(
            f"Match {match_idx}: {A_name} (A) vs {B_name} (B) — Transfers by Round"
        )
        legend_elems = [
            Patch(
                facecolor=PLAYER_A_COLOR,
                edgecolor="black",
                label=f"Winner: {A_name} (A)",
            ),
            Patch(
                facecolor=PLAYER_B_COLOR,
                edgecolor="black",
                label=f"Winner: {B_name} (B)",
            ),
            Patch(facecolor=NEUTRAL_COLOR, edgecolor="black", label="No transfer"),
        ]
        plt.legend(handles=legend_elems, loc="upper right")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"match{match_idx}_transfers.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[HOST] Saved {fname}")
        return fname

    elif kind == "wealth":
        capA_after = [int(r["capA"]) for r in rounds]
        capB_after = [int(r["capB"]) for r in rounds]
        edge_colors = []
        for r in rounds:
            if r["winner"] == "A":
                edge_colors.append(PLAYER_A_COLOR)
            elif r["winner"] == "B":
                edge_colors.append(PLAYER_B_COLOR)
            else:
                edge_colors.append(EDGE_NEUTRAL)

        plt.figure(figsize=(max(6, N * 0.35), 4.8))
        plt.bar(
            xs,
            capA_after,
            color=PLAYER_A_COLOR,
            edgecolor=edge_colors,
            linewidth=0.9,
            label=f"{A_name} (A)",
        )
        plt.bar(
            xs,
            capB_after,
            bottom=capA_after,
            color=PLAYER_B_COLOR,
            edgecolor=edge_colors,
            linewidth=0.9,
            label=f"{B_name} (B)",
        )

        total_sum = capA_after[0] + capB_after[0] if N > 0 else 0
        plt.xlabel("Round")
        plt.ylabel("Dollars (Stacked Wealth)")
        plt.title(
            f"Match {match_idx}: Wealth by Round (Total = ${total_sum} each round)\n"
            f"Edge color = round winner"
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"match{match_idx}_wealth.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[HOST] Saved {fname}")
        return fname

    else:
        print(f"[HOST] Unknown plot kind: {kind}")
        return None


def save_totals_plot(totals: Dict[str, int], out_dir: str) -> Optional[str]:
    if not HAVE_MPL:
        print("[HOST] matplotlib not available; skipping plots.")
        return None
    names = list(totals.keys())
    vals = [totals[n] for n in names]
    colors = [PLAYER_A_COLOR, PLAYER_B_COLOR] if len(names) == 2 else None

    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8)
    plt.title("Match Totals (sum across both games)")
    plt.ylabel("Dollars")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"${v}", ha="center", va="bottom")
    plt.tight_layout()
    fname = os.path.join(out_dir, "match_totals.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[HOST] Saved {fname}")
    return fname


# ---------- Main ----------
def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <port> <config.txt>")
        sys.exit(1)
    port = int(sys.argv[1])
    config_path = sys.argv[2]

    try:
        initial_cap_A, initial_cap_B, probabilities = load_match_config(config_path)
        p1, p2 = accept_two_players(port)
        print(f"[HOST] Players: {p1.name} and {p2.name}")
        print(
            f"[HOST] Loaded config: A=${initial_cap_A}, B=${initial_cap_B}, rounds={len(probabilities)}"
        )

        # Match 1: p1 is A, p2 is B
        A1, B1, rounds1 = run_single_game(
            probabilities, p1, p2, initial_cap_A, initial_cap_B, seed=42, match_idx=1
        )
        # Match 2: swap roles
        A2, B2, rounds2 = run_single_game(
            probabilities, p2, p1, initial_cap_A, initial_cap_B, seed=42, match_idx=2
        )

        # Team totals (sum of their end-of-game capitals across both roles)
        totals = {p1.name: A1 + B2, p2.name: B1 + A2}

        # Send summary to clients
        summary = {
            "type": "match_over",
            "players": [p1.name, p2.name],
            "match1": {
                "A_name": p1.name,
                "B_name": p2.name,
                "A_final": A1,
                "B_final": B1,
            },
            "match2": {
                "A_name": p2.name,
                "B_name": p1.name,
                "A_final": A2,
                "B_final": B2,
            },
            "totals": totals,
        }
        p1.send(summary)
        p2.send(summary)

        # Save plots under results/<A>_<B>_match/
        out_dir = _ensure_out_dir(p1.name, p2.name)
        save_match_rounds_plot(
            rounds1,
            A_name=p1.name,
            B_name=p2.name,
            match_idx=1,
            out_dir=out_dir,
            kind="transfers",
        )
        save_match_rounds_plot(
            rounds1,
            A_name=p1.name,
            B_name=p2.name,
            match_idx=1,
            out_dir=out_dir,
            kind="wealth",
        )
        save_match_rounds_plot(
            rounds2,
            A_name=p2.name,
            B_name=p1.name,
            match_idx=2,
            out_dir=out_dir,
            kind="transfers",
        )
        save_match_rounds_plot(
            rounds2,
            A_name=p2.name,
            B_name=p1.name,
            match_idx=2,
            out_dir=out_dir,
            kind="wealth",
        )

        print("=== TOTALS ===")
        for name, tot in totals.items():
            print(f"{name}: ${tot}")

    finally:
        try:
            p1.close()
            p2.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
