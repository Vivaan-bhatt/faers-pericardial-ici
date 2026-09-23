#!/usr/bin/env python3
"""
analyze_final.py  -  every statistic in the paper, computed from ONE frozen dataset.

Run from the project root (after build_final_dataset.py):
    python3 scripts/analyze_final.py

READS   data/final/case_level_final.csv
WRITES  data/final/results/*.csv      (one file per analysis)
        data/final/results/final_numbers.csv   (long format: table | row | metric | value)
        data/final/results/gps_target_input_final.csv  (input for score_ebgm_final.R)

DEFINITIONS (all documented in study_config.py / DATASET_DICTIONARY.csv)
  analysis universe U = all cases  minus FDA-deleted cases  minus cases exposed ONLY to a
                        non-study checkpoint inhibitor (e.g. tislelizumab)
  exposed   (primary) = any of the 9 agents named in any drug row (any role) by generic, brand,
                        development code name, or prod_ai
  reference (primary) = every case in U with none of the 9 agents
  event     (primary) = any of the 7 primary pericardial PTs
  2x2: a = exposed+event  b = exposed+no event  c = reference+event  d = reference+no event
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import study_config as cfg

D = Path("data/final")
R = D / "results"
AG = cfg.AGENT_LIST
PRIM_COLS = [cfg.pt_slug(p) for p in cfg.PRIMARY_PTS]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def bcpnn(a, b, c, d):
    """Jeffreys-type BCPNN, identical to the formula used in calculate_bayesian_signals.py."""
    n11, n10, n01, n00 = map(float, (a, b, c, d))
    n1p, np1, npp = n11 + n10, n11 + n01, n11 + n10 + n01 + n00
    g11, a1, b1, al, be = 1.0, 1.0, 1.0, 2.0, 2.0
    g = g11 * ((npp + al) * (npp + be)) / ((n1p + a1) * (np1 + b1))
    ic = math.log2(((n11 + g11) * (npp + al) * (npp + be)) / ((npp + g) * (n1p + a1) * (np1 + b1)))
    var = (1 / math.log(2) ** 2) * (
        (npp - n11 + g - g11) / ((n11 + g11) * (1 + npp + g))
        + (npp - n1p + al - a1) / ((n1p + a1) * (1 + npp + al))
        + (npp - np1 + be - b1) / ((np1 + b1) * (1 + npp + be)))
    sd = math.sqrt(max(var, 0.0))
    return ic, ic - 1.96 * sd, ic + 1.96 * sd


def signals(a, b, c, d):
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    out = dict(a=a, b=b, c=c, d=d, N=n)
    nan = float("nan")
    if min(a, b, c, d) > 0:
        ror = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out.update(ROR=ror, ROR_lo=ror * math.exp(-1.96 * se), ROR_hi=ror * math.exp(1.96 * se))
    else:
        out.update(ROR=nan, ROR_lo=nan, ROR_hi=nan)
    if a > 0 and c > 0 and (a + b) > 0 and (c + d) > 0:
        prr = (a / (a + b)) / (c / (c + d))
        se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        out.update(PRR=prr, PRR_lo=prr * math.exp(-1.96 * se), PRR_hi=prr * math.exp(1.96 * se))
    else:
        out.update(PRR=nan, PRR_lo=nan, PRR_hi=nan)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = n * (a * d - b * c) ** 2 / den if den > 0 else nan
    out["chi2"] = chi2
    out["Evans"] = bool(a >= 3 and out["PRR"] == out["PRR"] and out["PRR"] >= 2 and chi2 == chi2 and chi2 >= 4)
    if (a + b) > 0 and (a + c) > 0 and n > 0:
        ic, lo, hi = bcpnn(a, b, c, d)
        out.update(IC=ic, IC025=lo, IC975=hi)
    else:
        out.update(IC=nan, IC025=nan, IC975=nan)
    return out


def cell(label, exposed, reference, outcome, **extra):
    a = int((exposed & outcome).sum())
    b = int((exposed & ~outcome).sum())
    c = int((reference & outcome).sum())
    d = int((reference & ~outcome).sum())
    return {"analysis": label, **extra, **signals(a, b, c, d)}


def logit_irls(X, succ, n):
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        eta = X @ beta
        mu = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        w = n * mu * (1 - mu)
        z = eta + (succ - n * mu) / w
        xtw = X.T * w
        new = np.linalg.solve(xtw @ X, xtw @ z)
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break
    eta = X @ beta
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    w = n * mu * (1 - mu)
    cov = np.linalg.inv((X.T * w) @ X)
    return beta, np.sqrt(np.diag(cov))


def temporal_model(q, start_t=0):
    """logit(P(event)) ~ 1 + t + ici + ici:t on grouped quarter x exposure cells."""
    rows = []
    for _, r in q.iterrows():
        if r["t"] < start_t:
            continue
        for ici, succ, n in ((0, r["c"], r["c"] + r["d"]), (1, r["a"], r["a"] + r["b"])):
            if n > 0:
                rows.append((r["t"], ici, succ, n))
    g = pd.DataFrame(rows, columns=["t", "ici", "succ", "n"])
    t = g["t"].to_numpy(float) - start_t
    X = np.column_stack([np.ones(len(g)), t, g["ici"], g["ici"] * t])
    beta, se = logit_irls(X, g["succ"].to_numpy(float), g["n"].to_numpy(float))
    out = []
    for name, b, s in zip(["intercept", "quarter", "ici", "ici_x_quarter"], beta, se):
        z = b / s
        out.append(dict(term=name, coef=b, se=s, OR=math.exp(b), OR_lo=math.exp(b - 1.96 * s),
                        OR_hi=math.exp(b + 1.96 * s), z=z, p=math.erfc(abs(z) / math.sqrt(2))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def main():
    R.mkdir(parents=True, exist_ok=True)
    i8 = "int8"
    dtypes = {c: i8 for c in ["deleted", "in_window", "n_agents", "ici9_any", "ici9_generic", "ici9_suspect",
                              "ici_nonblinded", "newer_ici", "outcome_primary", "outcome_immune", "outcome_broad",
                              "malig_inf_any", "myocarditis", "other_l01", "first_quarter"]}
    dtypes.update({f"ag_{a}": i8 for a in AG})
    dtypes.update({c: i8 for c in PRIM_COLS})
    dtypes.update({"first_year": "int16", "agents_generic_mask": "int32", "agents_suspect_mask": "int32",
                   "source_group": "category"})
    df = pd.read_csv(D / "case_level_final.csv", dtype=dtypes,
                     usecols=lambda c: c not in ("caseid", "pt_bits"))
    N0 = len(df)
    B = lambda c: df[c].to_numpy() == 1

    deleted, ici, newer = B("deleted"), B("ici9_any"), B("newer_ici")
    newer_only = newer & ~ici
    U = ~deleted & ~newer_only
    O1 = B("outcome_primary")
    myo, mal, l01 = B("myocarditis"), B("malig_inf_any"), B("other_l01")
    src = df["source_group"].astype(str).to_numpy()
    res = {}

    # ---- flow --------------------------------------------------------------
    flow = [("universe (legacy U modern, unique CASEID)", N0),
            ("excluded: FDA-deleted cases", int(deleted.sum())),
            ("excluded: exposed only to a non-study checkpoint inhibitor", int((newer_only & ~deleted).sum())),
            ("analysis universe U", int(U.sum())),
            ("exposed to >=1 of the 9 agents (any role)", int((ici & U).sum())),
            ("reference (none of the 9 agents)", int((~ici & U).sum())),
            ("cases with primary pericardial PT", int((O1 & U).sum())),
            ("exposed + primary pericardial PT", int((ici & O1 & U).sum()))]
    res["cohort_flow"] = pd.DataFrame(flow, columns=["analysis", "n"])

    # ---- primary ------------------------------------------------------------
    res["primary"] = pd.DataFrame([cell("primary: any of 9 agents vs none", ici & U, ~ici & U, O1)])

    # ---- agents -------------------------------------------------------------
    rows = []
    for a in AG:
        bit = B(f"ag_{a}")
        rows.append(cell(a, bit & U, ~ici & U, O1, definition="any case naming the agent (may include other agents)"))
        rows.append(cell(a, bit & (df["n_agents"].to_numpy() == 1) & U, ~ici & U, O1,
                         definition="cases naming ONLY this agent"))
    res["agent_level"] = pd.DataFrame(rows)

    n_ag = df["n_agents"].to_numpy()
    ov = [dict(analysis=f"cases with {k} agent(s)", n=int(((n_ag == k) & U & ici).sum())) for k in range(1, len(AG) + 1)]
    pair = []
    for i, a in enumerate(AG):
        for b in AG[i + 1:]:
            pair.append(dict(analysis=f"{a}+{b}", n=int((B(f"ag_{a}") & B(f"ag_{b}") & U).sum())))
    res["agent_overlap"] = pd.DataFrame(ov + pair)

    # ---- PT level -------------------------------------------------------------
    rows = [cell("ANY primary PT", ici & U, ~ici & U, O1)]
    for p, col in zip(cfg.PRIMARY_PTS, PRIM_COLS):
        rows.append(cell(p, ici & U, ~ici & U, B(col)))
    res["pt_level"] = pd.DataFrame(rows)

    # ---- sensitivity analyses -------------------------------------------------
    S = []
    g = B("ici9_generic")
    S.append(cell("S0 primary", ici & U, ~ici & U, O1))
    S.append(cell("S1a exposure = generic names only (old-style; brand-only ICI cases left in reference)",
                  g & U, ~g & U, O1))
    S.append(cell("S1b exposure = generic names only; brand-only ICI cases removed from reference",
                  g & U, ~ici & U, O1))
    S.append(cell("S2 exposure = suspect role only (PS/SS); ICI-as-concomitant-only cases excluded",
                  B("ici9_suspect") & U, ~ici & U, O1))
    S.append(cell("S3 blinded-only ICI cases excluded", ici & B("ici_nonblinded") & U, ~ici & U, O1))
    Uc = ~deleted
    cls = ici | newer
    S.append(cell("S4 class-wide: 9 agents + other checkpoint inhibitors", cls & Uc, ~cls & Uc, O1))
    S.append(cell("S5 single-agent cases only; multi-agent cases excluded", ici & (n_ag == 1) & U, ~ici & U, O1))
    for label, m in (("S6a modern-only cases", src == "modern_only"), ("S6b legacy-only cases", src == "legacy_only")):
        S.append(cell(label, ici & U & m, ~ici & U & m, O1))
    Un = ~newer_only
    S.append(cell("S7 FDA-deleted cases NOT excluded", ici & Un, ~ici & Un, O1))
    S.append(cell("S8 outcome = primary + myopericarditis + pleuropericarditis", ici & U, ~ici & U, B("outcome_immune")))
    S.append(cell("S9 outcome = broad (all non-procedural pericardial PTs)", ici & U, ~ici & U, B("outcome_broad")))
    U10 = U & ~(O1 & mal)
    S.append(cell("S10 primary outcome; cases with malignant/infective pericardial PTs removed", ici & U10, ~ici & U10, O1))
    U11 = U & ~(O1 & myo)
    S.append(cell("S11 primary outcome; myocarditis co-reported cases removed", ici & U11, ~ici & U11, O1))
    six = np.zeros(N0, bool)
    for p, col in zip(cfg.PRIMARY_PTS, PRIM_COLS):
        if p != "immune-mediated pericarditis":
            six |= B(col)
    S.append(cell("S12 outcome = original six PTs (corrected spelling); immune-mediated pericarditis not counted",
                  ici & U, ~ici & U, six))
    res["sensitivity"] = pd.DataFrame(S)

    # ---- deleted-case diagnostic (are FDA-deleted cases unusual?) -------------------
    yr_all = df["first_year"].to_numpy()
    rows = []
    for y in [None] + sorted(int(v) for v in np.unique(yr_all[yr_all > 0])):
        m = np.ones(N0, bool) if y is None else (yr_all == y)
        dm, nm = m & deleted, m & ~deleted
        if dm.sum() == 0:
            continue
        rows.append({"analysis": "ALL years" if y is None else str(y), "n_cases": int(m.sum()),
                     "n_deleted": int(dm.sum()), "pct_deleted": 100 * dm.sum() / m.sum(),
                     "pericardial_pct_deleted": 100 * (O1 & dm).sum() / dm.sum(),
                     "pericardial_pct_not_deleted": 100 * (O1 & nm).sum() / max(nm.sum(), 1),
                     "ici_pct_deleted": 100 * (ici & dm).sum() / dm.sum(),
                     "ici_pct_not_deleted": 100 * (ici & nm).sum() / max(nm.sum(), 1)})
    res["deleted_diagnostic"] = pd.DataFrame(rows)

    # ---- myocarditis split -----------------------------------------------------
    iso, wm = O1 & ~myo, O1 & myo
    M = [dict(analysis="counts: exposed+primary PT", n=int((ici & O1 & U).sum())),
         dict(analysis="counts: exposed+primary PT+myocarditis", n=int((ici & wm & U).sum())),
         dict(analysis="counts: reference+primary PT+myocarditis", n=int((~ici & wm & U).sum())),
         cell("isolated pericardial (co-reported cases kept in denominator as non-events)", ici & U, ~ici & U, iso),
         cell("with myocarditis (same construction)", ici & U, ~ici & U, wm),
         cell("isolated pericardial (co-reported cases EXCLUDED from both arms)", ici & U11, ~ici & U11, iso)]
    res["myocarditis_split"] = pd.DataFrame(M)

    # ---- L01 comparator ----------------------------------------------------------
    ref = ~ici & l01 & U
    C = [dict(analysis="counts: ICI only (no L01)", n=int((ici & ~l01 & U).sum())),
         dict(analysis="counts: ICI + other L01", n=int((ici & l01 & U).sum())),
         dict(analysis="counts: L01 only (reference)", n=int(ref.sum())),
         cell("ICI-only vs L01-only", ici & ~l01 & U, ref, O1),
         cell("ICI+L01 vs L01-only", ici & l01 & U, ref, O1),
         cell("any ICI vs L01-only (current-design estimand)", ici & U, ref, O1)]
    res["l01_comparator"] = pd.DataFrame(C)

    # ---- temporal -----------------------------------------------------------------
    W = U & B("in_window")
    yr, qt = df["first_year"].to_numpy(), df["first_quarter"].to_numpy()
    rows = []
    for (y, q_), idx in pd.Series(np.arange(N0)[W]).groupby([yr[W], qt[W]]):
        ix = idx.to_numpy()
        i_, o_ = ici[ix], O1[ix]
        r = signals((i_ & o_).sum(), (i_ & ~o_).sum(), (~i_ & o_).sum(), (~i_ & ~o_).sum())
        rows.append({"analysis": f"{y}Q{q_}", "year": int(y), "quarter": int(q_),
                     "t": (int(y) - 2011) * 4 + int(q_) - 1, **r})
    q = pd.DataFrame(rows)
    res["temporal_quarterly"] = q
    parts = []
    for label, st in (("all quarters 2011Q1-2026Q2", 0), ("2015Q1-2026Q2", (2015 - 2011) * 4)):
        try:
            m = temporal_model(q, start_t=st)
            m.insert(0, "analysis", label)
            parts.append(m)
        except Exception as e:                      # e.g. too few quarters / singular design
            print(f"WARNING: temporal model '{label}' failed: {type(e).__name__}: {e}")
    res["temporal_model"] = (pd.concat(parts, ignore_index=True) if parts
                             else pd.DataFrame(columns=["analysis", "term", "coef", "se", "OR", "OR_lo", "OR_hi", "z", "p"]))
    res["temporal_summary"] = pd.DataFrame([
        dict(analysis="quarters analysed", n=len(q)),
        dict(analysis="quarters with >=1 exposed+event case", n=int((q["a"] > 0).sum())),
        dict(analysis="quarters meeting Evans criteria", n=int(q["Evans"].sum())),
        dict(analysis="cases outside 2011Q1-2026Q2 (excluded from temporal only)", n=int((U & ~B("in_window")).sum()))])

    # ---- GPS target input ------------------------------------------------------------
    nU = int(U.sum())
    rows = []
    for a in AG:
        bit = B(f"ag_{a}") & U
        pm = int(bit.sum())
        for p, col in zip(cfg.PRIMARY_PTS, PRIM_COLS):
            pt = B(col) & U
            em = int(pt.sum())
            rows.append(dict(var1=a, var2=p, N=int((bit & pt).sum()), E=pm * em / nU if em > 0 else 0.0,
                             product_margin=pm, event_margin=em, total=nU))
    gps = pd.DataFrame(rows)
    gps.to_csv(R / "gps_target_input_final.csv", index=False)

    # ---- write + long-format numbers ----------------------------------------------------
    long = []
    for name, t in res.items():
        t.to_csv(R / f"{name}.csv", index=False)
        lab = "analysis"
        for i, r in t.iterrows():
            for c in t.columns:
                if c == lab or c == "definition":
                    continue
                v = r[c]
                if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)):
                    long.append((name, f"{r[lab]}" + (f" [{r['definition']}]" if "definition" in t.columns else ""), c, v))
    pd.DataFrame(long, columns=["table", "row", "metric", "value"]).to_csv(R / "final_numbers.csv", index=False)

    # ---- console summary ------------------------------------------------------------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n== COHORT FLOW ==")
    print(res["cohort_flow"].to_string(index=False))
    show = ["analysis", "a", "b", "c", "d", "ROR", "ROR_lo", "ROR_hi", "PRR", "IC025", "Evans"]
    for k in ("primary", "agent_level", "pt_level", "sensitivity", "l01_comparator"):
        print(f"\n== {k.upper()} ==")
        t = res[k]
        cols = [c for c in show + ["definition"] if c in t.columns]
        print(t[t["a"].notna()][cols].round(3).to_string(index=False))
    print("\n== TEMPORAL MODEL ==")
    print(res["temporal_model"].round(4).to_string(index=False))
    print(f"\nResults written to {R}/")


if __name__ == "__main__":
    main()
