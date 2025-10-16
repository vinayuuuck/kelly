#!/usr/bin/env python3
# client.py
import json, socket, sys
from typing import Dict, Any, Optional, List


# ================== EDIT ONLY THIS FUNCTION ==================
def my_strategy(
    p: List[float],  # full probability sequence
    my_capital: int,  # your dollars (int)
    opp_capital: int,  # opponent dollars (int)
    role: str,  # you are player 'A' or 'B' for this match
    round_index: int,  # 0-based index k
    total_rounds: int,  # len(p)
) -> int:
    """
    Example: Kelly-style using p[k] for the current round.
    You can plan ahead using the full sequence `p`.
    """
    pk = p[round_index]
    q = pk if role == "A" else (1.0 - pk)
    frac = max(0.0, 2.0 * q - 1.0)  # Kelly for even-money bet
    bet = int(frac * my_capital)

    return max(
        0, min(bet, 0.1 * my_capital)
    )  # cap at 10% of current capital. The host will also enforce this.


# =============================================================


def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


class LineBufferedConn:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def recv_json(self) -> Optional[Dict[str, Any]]:
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


def main():
    if len(sys.argv) not in (3, 4):
        print(f"Usage: python {sys.argv[0]} <port> <player_name> [host]")
        sys.exit(1)

    port = int(sys.argv[1])
    player_name = sys.argv[2]
    host = sys.argv[3] if len(sys.argv) == 4 else "127.0.0.1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    reader = LineBufferedConn(sock)

    send_json(sock, {"type": "hello", "player_name": player_name})
    msg = reader.recv_json()
    if not msg or msg.get("type") != "ack":
        print("[CLIENT] Failed to join:", msg)
        sock.close()
        return
    print(f"[CLIENT] Connected as {player_name}.")

    prob_seq: Optional[List[float]] = None
    total_rounds: Optional[int] = None
    role: Optional[str] = None

    while True:
        msg = reader.recv_json()
        if msg is None:
            print("[CLIENT] Disconnected.")
            break

        mtype = msg.get("type")
        if mtype == "game_start":
            role = str(msg["you"])
            total_rounds = int(msg["total_rounds"])
            prob_seq = list(map(float, msg.get("probs", [])))
            if total_rounds != len(prob_seq):
                prob_seq = prob_seq[:total_rounds] if prob_seq else [0.5] * total_rounds

            print(
                f"[CLIENT] Match start. You({role})={msg['initial_capital_you']}, "
                f"Opp({msg['opponent_name']})={msg['initial_capital_opp']}, rounds={total_rounds}"
            )

        elif mtype == "request_bet":
            if prob_seq is None or total_rounds is None or role is None:
                print("[CLIENT] Missing game_start data; cannot bet.")
                send_json(sock, {"type": "bet", "bet": 0})
                continue

            k = int(msg["round_index"])
            my_cap = int(msg["your_capital"])
            opp_cap = int(msg["opp_capital"])
            host_cap = int(msg.get("max_bet", my_cap))  # host-announced cap (10% rule)
            time_left = msg.get("time_left_sec", None)

            bet = my_strategy(prob_seq, my_cap, opp_cap, role, k, total_rounds)
            # Enforce local safety clamps (host also enforces):
            bet = max(0, min(int(bet), my_cap, host_cap))
            send_json(sock, {"type": "bet", "bet": bet})

            if time_left is not None:
                print(
                    f"[CLIENT] Round {k+1}: sent bet={bet} (max={host_cap}), time_left≈{time_left:.2f}s"
                )

        elif mtype == "round_result":
            tlA = msg.get("time_left_A_sec")
            tlB = msg.get("time_left_B_sec")
            tl_note = ""
            if tlA is not None and tlB is not None:
                tl_note = f" | t_left A={tlA:.2f}s B={tlB:.2f}s"
            print(
                f"[CLIENT] Round {msg['round_index']+1} -> winner={msg['winner']}, "
                f"A=${msg['capA']}, B=${msg['capB']}{tl_note}"
            )

        elif mtype == "forfeit":
            print(
                f"[CLIENT] FORFEIT: winner={msg['winner']} loser={msg['loser']} reason={msg.get('reason','')}"
            )
            print(f"[CLIENT] Capitals now: A=${msg['capA']} B=${msg['capB']}")

        elif mtype == "game_over":
            print("\n[CLIENT] MATCH END (this role)")
            print(f"  A ({msg['name_A']}) = ${msg['final_capital_A']}")
            print(f"  B ({msg['name_B']}) = ${msg['final_capital_B']}")
            # reset for the next swapped-role match
            prob_seq = None
            total_rounds = None
            role = None

        elif mtype == "match_over":
            # Be compatible with both host schemas: 'match1'/'match2' or 'game1'/'game2'
            m1 = msg.get("match1") or msg.get("game1")
            m2 = msg.get("match2") or msg.get("game2")
            totals = msg.get("totals", {})

            print("\n[CLIENT] MATCH OVER (both roles)")
            if m1:
                print(
                    f"Match 1: A({m1['A_name']})=${m1['A_final']}, B({m1['B_name']})=${m1['B_final']}"
                )
            if m2:
                print(
                    f"Match 2: A({m2['A_name']})=${m2['A_final']}, B({m2['B_name']})=${m2['B_final']}"
                )

            for name, tot in totals.items():
                print(f"Total {name}: ${tot}")
            break

        else:
            # ignore unknown
            pass

    sock.close()


if __name__ == "__main__":
    main()
