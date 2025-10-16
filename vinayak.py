#!/usr/bin/env python3
import json, socket, sys, os, hashlib, time, random
from typing import Dict, Any, Optional, List

PARAMS_FILE = "tuned_params.json"
DEFAULT_TRIALS = int(os.environ.get("TUNER_TRIALS", "120"))
DEFAULT_EVAL_SIMS = int(os.environ.get("TUNER_EVAL_SIMS", "250"))
DEFAULT_FINAL_SIMS = int(os.environ.get("TUNER_FINAL_SIMS", "800"))

CURRENT_KEY = None
CURRENT_PARAMS = None


def make_seq_key(initA: int, initB: int, pseq: List[float]) -> str:
    m = hashlib.sha256()
    s = f"{initA}-{initB}-{len(pseq)}-" + ",".join(f"{x:.8f}" for x in pseq)
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def load_all_params():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_all_params(d):
    with open(PARAMS_FILE, "w") as f:
        json.dump(d, f, indent=2)


def kelly_strategy_func(pfull, mycap, oppcap, role, round_idx, total_rounds):
    pk = float(pfull[round_idx])
    q = pk if role == "A" else 1.0 - pk
    frac = max(0.0, 2.0 * q - 1.0)
    bet = int(frac * mycap)
    cap = int(0.1 * mycap)
    bet = max(0, min(bet, cap, mycap))
    return bet


def heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh):
    def strat(pfull, mycap, oppcap, role, round_index, total_rounds):
        pk = float(pfull[round_index])
        q = pk if role == "A" else 1.0 - pk
        k_frac = max(0.0, 2.0 * q - 1.0)
        bet_frac = min_frac + alpha * k_frac
        max_bet = int(0.1 * mycap)
        if oppcap <= max_bet and q > q_thresh:
            bet_frac += kill_boost
        rem = total_rounds - round_index
        if rem <= 3 and mycap < oppcap:
            bet_frac += late_boost
        bet = int(bet_frac * mycap)
        bet = max(0, min(bet, mycap, max_bet))
        return bet

    return strat


def play_match(stratA, stratB, initA, initB, pseq, seed=None):
    if seed is not None:
        random.seed(seed)
    a = initA
    b = initB
    total_rounds = len(pseq)
    for k, p in enumerate(pseq):
        if a <= 0 or b <= 0:
            break
        betA = stratA(pseq, a, b, "A", k, total_rounds)
        betB = stratB(pseq, b, a, "B", k, total_rounds)
        betA = max(0, min(int(betA), a, int(0.1 * a)))
        betB = max(0, min(int(betB), b, int(0.1 * b)))
        total = betA + betB
        r = random.random()
        if r <= p:
            transfer = min(total, b)
            a += transfer
            b -= transfer
        else:
            transfer = min(total, a)
            a -= transfer
            b += transfer
    return a, b


def evaluate_strategy(stratA, stratB, initA, initB, pseq, n_sim=500, seed_base=0):
    wins = 0
    sumA = 0
    sumB = 0
    for i in range(n_sim):
        a_final, b_final = play_match(
            stratA, stratB, initA, initB, pseq, seed=seed_base + i
        )
        sumA += a_final
        sumB += b_final
        if a_final > b_final:
            wins += 1
    return {"wins": wins, "A_sum": sumA, "B_sum": sumB, "winrate": wins / n_sim}


def random_search_best(
    initA, initB, pseq, n_trials=DEFAULT_TRIALS, eval_sims=DEFAULT_EVAL_SIMS
):
    best = None
    for t in range(n_trials):
        alpha = random.uniform(0.5, 3.0)
        min_frac = random.uniform(0.0, 0.05)
        kill_boost = random.uniform(0.0, 0.5)
        late_boost = random.uniform(0.0, 1.0)
        q_thresh = random.uniform(0.52, 0.75)
        strat = heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh)
        res = evaluate_strategy(
            strat,
            kelly_strategy_func,
            initA,
            initB,
            pseq,
            n_sim=eval_sims,
            seed_base=10000 + t * 10,
        )
        score = res["winrate"]
        if best is None or score > best[0]:
            best = (score, alpha, min_frac, kill_boost, late_boost, q_thresh, res)
    return best


def refine_around(initA, initB, pseq, best, eval_sims=300):
    if best is None:
        return None
    a0, m0, kb0, lb0, qt0 = best[1], best[2], best[3], best[4], best[5]
    alphas = [a0 * 0.8, a0, a0 * 1.2]
    mins = [max(0, m0 - 0.005), m0, m0 + 0.005]
    kills = [max(0, kb0 - 0.05), kb0, kb0 + 0.05]
    lates = [max(0, lb0 - 0.2), lb0, lb0 + 0.2]
    qts = [max(0.52, qt0 - 0.03), qt0, min(0.75, qt0 + 0.03)]
    best2 = None
    for a in alphas:
        for m in mins:
            for kb in kills:
                for lb in lates:
                    for qt in qts:
                        strat = heuristic_factory(a, m, kb, lb, qt)
                        res = evaluate_strategy(
                            strat,
                            kelly_strategy_func,
                            initA,
                            initB,
                            pseq,
                            n_sim=eval_sims,
                            seed_base=20000,
                        )
                        score = res["winrate"]
                        if best2 is None or score > best2[0]:
                            best2 = (score, a, m, kb, lb, qt, res)
    return best2


def tune_params_for_instance(
    initA,
    initB,
    pseq,
    trials=DEFAULT_TRIALS,
    eval_sims=DEFAULT_EVAL_SIMS,
    final_sims=DEFAULT_FINAL_SIMS,
):
    all_params = load_all_params()
    key = make_seq_key(initA, initB, pseq)
    if key in all_params:
        return all_params[key]
    # Decide objective: for very long games or large capitals, optimize expected final capital
    use_expected = False
    try:
        if len(pseq) >= 500 or (initA + initB) >= 5000:
            use_expected = True
    except Exception:
        use_expected = False
    print(
        f"[TUNER] No cached params found. Running tuner (objective={'expected_capital' if use_expected else 'winrate'})..."
    )
    t0 = time.time()
    # Run random search (objective depends on use_expected)
    best = None
    chosen = None
    if not use_expected:
        best = random_search_best(
            initA, initB, pseq, n_trials=trials, eval_sims=eval_sims
        )
        print(
            f"[TUNER] Random search finished in {time.time()-t0:.1f}s. Best initial winrate={best[0] if best else None}"
        )
        refined = refine_around(
            initA, initB, pseq, best, eval_sims=max(100, int(eval_sims * 1.0))
        )
        chosen = None
        if refined is not None and refined[0] >= (best[0] if best else -1):
            chosen = refined
        else:
            chosen = best
    else:
        # optimize expected final capital
        best_exp = None
        for t in range(trials):
            alpha = random.uniform(0.6, 2.5)
            min_frac = random.uniform(0.0, 0.02)
            kill_boost = random.uniform(0.0, 0.25)
            late_boost = random.uniform(0.0, 0.5)
            q_thresh = random.uniform(0.52, 0.7)
            strat = heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh)
            # evaluate average A final capital
            avgA = evaluate_strategy(
                strat,
                kelly_strategy_func,
                initA,
                initB,
                pseq,
                n_sim=max(50, eval_sims // 3),
                seed_base=100000 + t * 11,
            )
            # re-use evaluate_strategy but extract avg final A from A_sum
            avgA_val = avgA["A_sum"] / max(50, eval_sims // 3)
            if best_exp is None or avgA_val > best_exp[0]:
                best_exp = (
                    avgA_val,
                    alpha,
                    min_frac,
                    kill_boost,
                    late_boost,
                    q_thresh,
                    avgA,
                )
        if best_exp is not None:
            # create a pseudo-'best' tuple compatible with refine_around (place avg as first elem)
            best = (
                best_exp[0],
                best_exp[1],
                best_exp[2],
                best_exp[3],
                best_exp[4],
                best_exp[5],
                best_exp[6],
            )
            refined = refine_around(
                initA, initB, pseq, best, eval_sims=max(100, int(eval_sims * 1.0))
            )
            if refined is not None:
                chosen = refined
            else:
                chosen = best

    if chosen is None:
        return None
    # chosen is (score, alpha, min_frac, kill_boost, late_boost, q_thresh, sample)
    score, alpha, min_frac, kill_boost, late_boost, q_thresh, sample = chosen
    final_params = {
        "alpha": alpha,
        "min_frac": min_frac,
        "kill_boost": kill_boost,
        "late_boost": late_boost,
        "q_thresh": q_thresh,
        "sample_winrate_or_score": score,
        "sample_eval": sample,
    }
    print("[TUNER] Final tuning: running robust final evals (may take longer)...")
    final_strat = heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh)
    res_vs_kelly = evaluate_strategy(
        final_strat,
        kelly_strategy_func,
        initA,
        initB,
        pseq,
        n_sim=final_sims,
        seed_base=30000,
    )

    def always_max(pfull, mycap, oppcap, role, round_idx, total_rounds):
        return int(min(mycap, int(0.1 * mycap)))

    res_vs_max = evaluate_strategy(
        final_strat, always_max, initA, initB, pseq, n_sim=final_sims, seed_base=40000
    )
    final_params["final_vs_kelly"] = res_vs_kelly
    final_params["final_vs_always_max"] = res_vs_max
    final_params["objective"] = "expected_capital" if use_expected else "winrate"
    all_params[key] = final_params
    save_all_params(all_params)
    print(f"[TUNER] Done. Saved tuned params for key {key[:8]}...")
    return final_params


def strategy(
    p: List[float],
    my_capital: int,
    opp_capital: int,
    role: str,
    round_index: int,
    total_rounds: int,
) -> int:
    global CURRENT_PARAMS
    if my_capital <= 0:
        return 0
    if CURRENT_PARAMS is None:
        return kelly_strategy_func(
            p, my_capital, opp_capital, role, round_index, total_rounds
        )
    alpha = float(CURRENT_PARAMS.get("alpha", 1.0))
    min_frac = float(CURRENT_PARAMS.get("min_frac", 0.0))
    kill_boost = float(CURRENT_PARAMS.get("kill_boost", 0.0))
    late_boost = float(CURRENT_PARAMS.get("late_boost", 0.0))
    q_thresh = float(CURRENT_PARAMS.get("q_thresh", 0.6))

    pk = float(p[round_index])
    q = pk if role == "A" else 1.0 - pk
    kelly_frac = max(0.0, 2.0 * q - 1.0)
    base_frac = min_frac + alpha * kelly_frac
    max_bet = int(0.1 * my_capital)

    if opp_capital <= max_bet and q > q_thresh:
        base_frac += kill_boost
    rem = total_rounds - round_index

    if rem <= 3 and my_capital < opp_capital:
        base_frac += late_boost

    bet = int(base_frac * my_capital)
    bet = max(0, min(bet, my_capital, max_bet))
    return bet


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
    initA = None
    initB = None

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

            initA = int(msg.get("initial_capital_you", 0))
            initB = int(msg.get("initial_capital_opp", 0))
            print(
                f"[CLIENT] Match start. You({role})={initA}, "
                f"Opp({msg['opponent_name']})={initB}, rounds={total_rounds}"
            )

            global CURRENT_KEY, CURRENT_PARAMS
            CURRENT_KEY = make_seq_key(initA, initB, prob_seq)
            allp = load_all_params()
            if CURRENT_KEY in allp:
                CURRENT_PARAMS = allp[CURRENT_KEY]
                print(
                    f"[TUNER] Loaded cached params for this instance (key {CURRENT_KEY[:8]})"
                )
            else:
                # run tuner (blocking). This will save to tuned_params.json
                CURRENT_PARAMS = tune_params_for_instance(initA, initB, prob_seq)
                if CURRENT_PARAMS is None:
                    print("[TUNER] Tuning failed; using Kelly fallback")
                else:
                    print(
                        f"[TUNER] Tuning complete. Using params (alpha={CURRENT_PARAMS.get('alpha'):.3f}, min_frac={CURRENT_PARAMS.get('min_frac'):.4f})"
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

            bet = strategy(prob_seq, my_cap, opp_cap, role, k, total_rounds)
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
            prob_seq = None
            total_rounds = None
            role = None

        elif mtype == "match_over":
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

            try:
                # global CURRENT_KEY, CURRENT_PARAMS
                if CURRENT_KEY:
                    allp = load_all_params()
                    if CURRENT_KEY in allp:
                        del allp[CURRENT_KEY]
                        save_all_params(allp)
                    if os.path.exists(PARAMS_FILE):
                        os.remove(PARAMS_FILE)
                    CURRENT_PARAMS = None
                    CURRENT_KEY = None

            except Exception as e:
                print(f"[TUNER] Warning: failed to delete cached params: {e}")
            break

        else:
            pass

    sock.close()


if __name__ == "__main__":
    main()
