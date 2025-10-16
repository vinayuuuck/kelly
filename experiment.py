#!/usr/bin/env python3
import sys, random, math, json, time, argparse

def read_input(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 4:
        raise ValueError("input must have at least 4 non-empty lines")
    initA = int(lines[0])
    initB = int(lines[1])
    N = int(lines[2])
    raw = " ".join(lines[3:])
    raw = raw.replace(",", " ")
    parts = [x for x in raw.split() if x]
    if len(parts) < N:
        raise ValueError("not enough probabilities provided")
    probs = [float(parts[i]) for i in range(N)]
    return initA, initB, probs

def kelly_strategy(pfull, mycap, oppcap, role, round_idx, total_rounds):
    pk = float(pfull[round_idx])
    q = pk if role=="A" else 1.0-pk
    frac = max(0.0, 2.0*q - 1.0)
    bet = int(frac * mycap)
    cap = int(0.1 * mycap)
    bet = max(0, min(bet, cap, mycap))
    return bet

def heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh):
    def strat(pfull, mycap, oppcap, role, round_index, total_rounds):
        pk = float(pfull[round_index])
        q = pk if role=="A" else 1.0-pk
        k_frac = max(0.0, 2.0*q - 1.0)
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
        betA = stratA(pseq, a, b, 'A', k, total_rounds)
        betB = stratB(pseq, b, a, 'B', k, total_rounds)
        betA = max(0, min(int(betA), a, int(0.1*a)))
        betB = max(0, min(int(betB), b, int(0.1*b)))
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
        a_final, b_final = play_match(stratA, stratB, initA, initB, pseq, seed=seed_base+i)
        sumA += a_final
        sumB += b_final
        if a_final > b_final:
            wins += 1
    return {"wins": wins, "A_sum": sumA, "B_sum": sumB, "winrate": wins / n_sim}

def random_search(initA, initB, pseq, n_trials=200, eval_sims=300):
    best = None
    for t in range(n_trials):
        alpha = random.uniform(0.5, 3.0)
        min_frac = random.uniform(0.0, 0.05)
        kill_boost = random.uniform(0.0, 0.5)
        late_boost = random.uniform(0.0, 1.0)
        q_thresh = random.uniform(0.52, 0.75)
        strat = heuristic_factory(alpha, min_frac, kill_boost, late_boost, q_thresh)
        res = evaluate_strategy(strat, kelly_strategy, initA, initB, pseq, n_sim=eval_sims, seed_base=10000+t*10)
        score = res["winrate"]
        if best is None or score > best[0]:
            best = (score, alpha, min_frac, kill_boost, late_boost, q_thresh, res)
    return best

def grid_search(initA, initB, pseq):
    alphas = [1.0, 1.5, 2.0]
    mins = [0.0, 0.001, 0.01]
    kills = [0.0, 0.1, 0.2]
    lates = [0.0, 0.2, 0.5]
    qts = [0.58, 0.62, 0.66]
    best = None
    for a in alphas:
        for m in mins:
            for kb in kills:
                for lb in lates:
                    for qt in qts:
                        strat = heuristic_factory(a, m, kb, lb, qt)
                        res = evaluate_strategy(strat, kelly_strategy, initA, initB, pseq, n_sim=200, seed_base=5000)
                        score = res["winrate"]
                        if best is None or score > best[0]:
                            best = (score, a,m,kb,lb,qt,res)
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="input.txt")
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--eval_sims", type=int, default=400)
    parser.add_argument("--final_sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    initA, initB, pseq = read_input(args.input)
    print("INPUT: A=", initA, "B=", initB, "rounds=", len(pseq))
    print("Running baseline Kelly vs Kelly (sanity)...")
    base = evaluate_strategy(kelly_strategy, kelly_strategy, initA, initB, pseq, n_sim=500, seed_base=1)
    print("Kelly vs Kelly:", base)
    print("Searching parameters with random search (trials=", args.trials, ")")
    t0 = time.time()
    best = random_search(initA, initB, pseq, n_trials=args.trials, eval_sims=args.eval_sims)
    t1 = time.time()
    print("Random search time:", t1-t0, "sec")
    if best is not None:
        score, alpha, min_frac, kill_boost, late_boost, q_thresh, res = best
        print("BEST RANDOM:", "winrate", score)
        print("params: alpha=", alpha, "min_frac=", min_frac, "kill_boost=", kill_boost, "late_boost=", late_boost, "q_thresh=", q_thresh)
        print("sample evaluation:", res)
    print("Refining with small grid around best (if any)")
    if best is not None:
        a0 = best[1]
        m0 = best[2]
        kb0 = best[3]
        lb0 = best[4]
        qt0 = best[5]
        alphas = [a0*0.8, a0, a0*1.2]
        mins = [max(0, m0-0.005), m0, m0+0.005]
        kills = [max(0, kb0-0.05), kb0, kb0+0.05]
        lates = [max(0, lb0-0.2), lb0, lb0+0.2]
        qts = [max(0.52, qt0-0.03), qt0, min(0.75, qt0+0.03)]
        best2 = None
        for a in alphas:
            for m in mins:
                for kb in kills:
                    for lb in lates:
                        for qt in qts:
                            strat = heuristic_factory(a, m, kb, lb, qt)
                            res = evaluate_strategy(strat, kelly_strategy, initA, initB, pseq, n_sim=300, seed_base=20000)
                            score = res["winrate"]
                            if best2 is None or score > best2[0]:
                                best2 = (score, a,m,kb,lb,qt,res)
        if best2 is not None:
            score, a,m,kb,lb,qt,res = best2
            print("BEST REFINED:", score)
            print("params:", a,m,kb,lb,qt)
            print("sample eval:", res)
            print("Final evaluation vs Kelly and Always-Max using", args.final_sims, "sims each:")
            final_strat = heuristic_factory(a,m,kb,lb,qt)
            res_vs_kelly = evaluate_strategy(final_strat, kelly_strategy, initA, initB, pseq, n_sim=args.final_sims, seed_base=30000)
            def always_max(pfull, mycap, oppcap, role, round_idx, total_rounds):
                return int(min(mycap, int(0.1*mycap)))
            res_vs_max = evaluate_strategy(final_strat, always_max, initA, initB, pseq, n_sim=args.final_sims, seed_base=40000)
            print("final vs kelly:", res_vs_kelly)
            print("final vs always-max:", res_vs_max)
            out = {
                "best_params": {"alpha": a, "min_frac": m, "kill_boost": kb, "late_boost": lb, "q_thresh": qt},
                "best_sample": res,
                "final_vs_kelly": res_vs_kelly,
                "final_vs_always_max": res_vs_max
            }
            with open("best_params.json", "w") as f:
                json.dump(out, f, indent=2)
            print("Wrote best_params.json")
    print("Done")

if __name__ == '__main__':
    main()
