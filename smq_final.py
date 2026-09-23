#!/usr/bin/env python3
"""
smq_final.py  -  SMQ sensitivity analysis on the frozen dataset (same universe, exposure and
statistics as analyze_final.py).

Run from the project root, after build_final_dataset.py:
    python3 scripts/smq_final.py

Outcome definitions compared (all with exposure = any of the 9 agents, reference = none):
  primary outcome            the 7 primary PTs (study_config.PRIMARY_PTS)
  full narrow SMQ            MedDRA v25.0 narrow "Noninfectious myocarditis/pericarditis" SMQ (14 PTs)
  SMQ pericardial subset     the 6 pericardial PTs of that SMQ (NOT itself an SMQ)

WRITES data/final/results/smq_sensitivity.csv and smq_term_counts.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import study_config as cfg
from analyze_final import signals

P = Path("data/processed")
D = Path("data/final")
R = D / "results"
CHUNK = 4_000_000

SMQ_TERMS = [
    "autoimmune myocarditis", "autoimmune pericarditis", "carditis", "chronic myocarditis",
    "eosinophilic myocarditis", "giant cell myocarditis", "hypersensitivity myocarditis",
    "immune-mediated myocarditis", "myocarditis", "myopericarditis", "pericarditis",
    "pericarditis adhesive", "pericarditis constrictive", "pleuropericarditis",
]
SMQ_PERI = ["autoimmune pericarditis", "myopericarditis", "pericarditis", "pericarditis adhesive",
            "pericarditis constrictive", "pleuropericarditis"]
BIT = {t: i for i, t in enumerate(SMQ_TERMS)}


def main():
    R.mkdir(parents=True, exist_ok=True)
    cols = ["caseid", "deleted", "ici9_any", "newer_ici", "outcome_primary"]
    df = pd.read_csv(D / "case_level_final.csv", usecols=cols,
                     dtype={"caseid": str, "deleted": "int8", "ici9_any": "int8",
                            "newer_ici": "int8", "outcome_primary": "int8"})
    idx = pd.Index(df["caseid"])
    assert idx.is_unique
    N = len(df)
    bits = np.zeros(N, np.int32)

    def add(pos, pts):
        ok = pos >= 0
        pos, pts = pos[ok], pts[ok]
        for t, b in BIT.items():
            sel = pts == t
            if sel.any():
                bits[pos[sel]] |= np.int32(1 << b)

    # legacy (unambiguous ISR -> CASE only, identical rule to the builder)
    lm = pd.read_csv(P / "legacy_case_isr_map.csv", dtype=str)
    lm["CASE"], lm["ISR"] = lm["CASE"].str.strip(), lm["ISR"].str.strip()
    lm = lm.drop_duplicates()
    per = lm.groupby("ISR")["CASE"].nunique()
    lm = lm[~lm["ISR"].isin(set(per[per > 1].index))].drop_duplicates("ISR")
    isr_pos = pd.Series(idx.get_indexer(lm["CASE"]), index=lm["ISR"].to_numpy())
    lr = pd.read_csv(P / "legacy_reac_selected.csv", dtype=str, usecols=["ISR", "PT"])
    pt = lr["PT"].fillna("").str.strip().str.lower()
    m = pt.isin(BIT)
    pos = isr_pos.reindex(lr.loc[m, "ISR"].str.strip().to_numpy()).fillna(-1).to_numpy(dtype=np.int64)
    add(pos, pt[m].to_numpy())

    # modern
    for ch in pd.read_csv(P / "modern_reac_selected.csv", dtype=str, chunksize=CHUNK,
                          usecols=lambda c: c.lower() in {"caseid", "pt"}):
        ch.columns = [c.lower() for c in ch.columns]
        pt = ch["pt"].fillna("").str.strip().str.lower()
        m = pt.isin(BIT)
        if m.any():
            add(idx.get_indexer(ch.loc[m, "caseid"].fillna("").str.strip()), pt[m].to_numpy())

    def has(terms):
        mk = np.int32(0)
        for t in terms:
            mk |= np.int32(1 << BIT[t])
        return (bits & mk) != 0

    ici = df["ici9_any"].to_numpy() == 1
    U = (df["deleted"].to_numpy() == 0) & ~((df["newer_ici"].to_numpy() == 1) & ~ici)
    outcomes = [("primary outcome (7 PTs)", df["outcome_primary"].to_numpy() == 1),
                ("full narrow SMQ: noninfectious myocarditis/pericarditis (14 PTs)", has(SMQ_TERMS)),
                ("SMQ pericardial subset (6 PTs)", has(SMQ_PERI))]
    rows = []
    for label, o in outcomes:
        a, b = int((ici & U & o).sum()), int((ici & U & ~o).sum())
        c, d = int((~ici & U & o).sum()), int((~ici & U & ~o).sum())
        rows.append({"analysis": label, "outcome_cases_in_U": int((U & o).sum()), **signals(a, b, c, d)})
    res = pd.DataFrame(rows)
    res.to_csv(R / "smq_sensitivity.csv", index=False)

    tc = [{"term": t, "cases_in_U": int((has([t]) & U).sum()), "exposed_cases_in_U": int((has([t]) & U & ici).sum())}
          for t in SMQ_TERMS]
    pd.DataFrame(tc).sort_values("cases_in_U", ascending=False).to_csv(R / "smq_term_counts.csv", index=False)

    pd.set_option("display.width", 200)
    print(res[["analysis", "outcome_cases_in_U", "a", "b", "c", "d", "ROR", "ROR_lo", "ROR_hi", "PRR", "Evans"]]
          .round(3).to_string(index=False))
    print("\n", pd.DataFrame(tc).sort_values("cases_in_U", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
