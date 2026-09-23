#!/usr/bin/env python3
"""
build_final_dataset.py  -  build ONE case-level analysis dataset.

Run from the project root:      python3 scripts/build_final_dataset.py

READS (never modifies):
  data/processed/legacy_aers_case_deduplicated.csv   legacy case universe + FDA_DT
  data/processed/legacy_case_isr_map.csv             CASE -> ISR
  data/processed/legacy_drug_selected.csv            legacy DRUG rows   (ISR, ROLE_COD, DRUGNAME)
  data/processed/legacy_reac_selected.csv            legacy REAC rows   (ISR, PT)
  data/processed/modern_case_primaryid_map.csv       modern case universe
  data/processed/modern_demo_case_deduplicated.csv   modern INIT_FDA_DT / FDA_DT
  data/processed/modern_drug_selected.csv            modern DRUG rows   (caseid, role_cod, drugname, prod_ai)
  data/processed/modern_reac_selected.csv            modern REAC rows   (caseid, pt)
  data/processed/who_l01_drug_list.csv               WHO ATC L01 names (already downloaded)
  data/raw/*.zip                                     FDA 'Deleted' case lists

WRITES (new folder only):
  data/final/case_level_final.csv        one row per unique case (the frozen dataset)
  data/final/DATASET_DICTIONARY.csv      what every column means
  data/final/build_summary.csv           headline counts + identifier hierarchy
  data/final/pt_bit_dictionary.csv       bit position of each tracked PT in `pt_bits`

IDENTIFIER HIERARCHY (documented):
  case identity   : FAERS CASEID (modern) == legacy CASE.  One row per CASEID.
  modern linkage  : DRUG/REAC rows were pre-filtered to the retained PRIMARYID of each case;
                    here they are joined by CASEID (safe: one retained version per case).
  legacy linkage  : DRUG/REAC rows join by ISR; ISR->CASE is used ONLY where the ISR maps to
                    exactly one CASE (146 ISRs shared by 292 cases are left unattributed).
  ISR / PRIMARYID are never compared with CASEID.
"""
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import study_config as cfg

P = Path("data/processed")
RAW = Path("data/raw")
OUT = Path("data/final")
CHUNK = 4_000_000

NA = len(cfg.AGENT_LIST)
BIT_NEWER, BIT_BLIND, BIT_L01 = 2 * NA, 2 * NA + 1, 2 * NA + 2     # 18, 19, 20
MASK_GENERIC = (1 << NA) - 1                                       # bits 0..8
MASK_ALL = ((1 << NA) - 1) << NA                                   # bits 9..17
RELEVANT = MASK_GENERIC | MASK_ALL | (1 << BIT_NEWER) | (1 << BIT_L01)

RX_GENERIC = [cfg.phrase_regex(cfg.AGENTS[a]["generic"]) for a in cfg.AGENT_LIST]
RX_ALL = [cfg.phrase_regex(cfg.AGENTS[a]["generic"] + cfg.AGENTS[a]["extra"]) for a in cfg.AGENT_LIST]
RX_NEWER = cfg.phrase_regex(cfg.NEWER_ICI_TERMS)
RX_BLIND = cfg.phrase_regex(cfg.BLINDED_TERMS)
RX_ICI_ANY = cfg.phrase_regex(
    [t for a in cfg.AGENT_LIST for t in cfg.AGENTS[a]["generic"] + cfg.AGENTS[a]["extra"]]
    + cfg.NEWER_ICI_TERMS)
RX_L01 = None                                                      # set in load_l01()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# drug-name classification (cached per distinct raw string)
# ---------------------------------------------------------------------------
def classify(raw, is_ai):
    s = cfg.normalize_name(raw)
    if not s:
        return 0
    code = 0
    if RX_ICI_ANY.search(s):
        for i in range(NA):
            if (not is_ai) and RX_GENERIC[i].search(s):      # generic flag = DRUGNAME only (prod_ai excluded)
                code |= 1 << i
            if RX_ALL[i].search(s):
                code |= 1 << (NA + i)
        if RX_NEWER.search(s):
            code |= 1 << BIT_NEWER
        if not is_ai and RX_BLIND.search(s):
            code |= 1 << BIT_BLIND
    if RX_L01 is not None and RX_L01.search(s):
        code |= 1 << BIT_L01
    return code


def code_series(series, cache, is_ai):
    s = series.fillna("").astype(str)
    todo = [v for v in s.unique() if v not in cache]
    for v in todo:
        cache[v] = classify(v, is_ai)
    return s.map(cache).to_numpy(dtype=np.int64)


def load_l01():
    global RX_L01
    who = pd.read_csv(P / "who_l01_drug_list.csv")
    ici_terms = {cfg.normalize_name(t) for a in cfg.AGENT_LIST
                 for t in cfg.AGENTS[a]["generic"] + cfg.AGENTS[a]["extra"]}
    ici_terms |= {cfg.normalize_name(t) for t in cfg.NEWER_ICI_TERMS}
    names = {cfg.normalize_name(n) for n in who["drug_name"]} - {""} - ici_terms
    RX_L01 = cfg.phrase_regex(sorted(names))
    log(f"WHO L01 names used for 'other antineoplastic': {len(names)} "
        f"(ICI generics removed: {len({cfg.normalize_name(n) for n in who['drug_name']} & ici_terms)})")


# ---------------------------------------------------------------------------
class Acc:
    """Per-case accumulators, indexed by position in the universe."""
    def __init__(self, n):
        self.agent_all = np.zeros(n, np.int32)
        self.agent_generic = np.zeros(n, np.int32)
        self.agent_suspect = np.zeros(n, np.int32)
        self.newer = np.zeros(n, np.int8)
        self.l01 = np.zeros(n, np.int8)
        self.nonblind = np.zeros(n, np.int8)
        self.pt_bits = np.zeros(n, np.int64)

    def add_drugs(self, pos, comb, suspect):
        keep = (comb & RELEVANT) != 0
        ok = keep & (pos >= 0)
        if not ok.any():
            return
        pos, comb, suspect = pos[ok], comb[ok], suspect[ok]
        for i in range(NA):
            g = ((comb >> i) & 1).astype(bool)
            a = ((comb >> (NA + i)) & 1).astype(bool)
            bit = np.int32(1 << i)
            self.agent_generic[pos[g]] |= bit
            self.agent_all[pos[a]] |= bit
            self.agent_suspect[pos[a & suspect]] |= bit
        self.newer[pos[((comb >> BIT_NEWER) & 1).astype(bool)]] = 1
        self.l01[pos[((comb >> BIT_L01) & 1).astype(bool)]] = 1
        any9 = (comb & MASK_ALL) != 0
        nb = any9 & (((comb >> BIT_BLIND) & 1) == 0)
        self.nonblind[pos[nb]] = 1

    def add_pts(self, pos, pts):
        ok = pos >= 0
        pos, pts = pos[ok], pts[ok]
        for pt, bit in PT_BIT.items():
            sel = pts == pt
            if sel.any():
                self.pt_bits[pos[sel]] |= np.int64(1 << bit)


PT_BIT = {pt: i for i, pt in enumerate(cfg.ALL_TRACKED_PTS)}


# ---------------------------------------------------------------------------
def read_ids(path, col):
    return pd.read_csv(path, dtype=str, usecols=[col])[col].fillna("").str.strip()


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    load_l01()

    # ---------------- 1. case universe ----------------
    log("Building case universe (legacy U modern)")
    leg_ids = read_ids(P / "legacy_aers_case_deduplicated.csv", "CASE")
    leg_ids = leg_ids[leg_ids != ""].drop_duplicates()
    mod_ids = read_ids(P / "modern_case_primaryid_map.csv", "caseid")
    mod_ids = mod_ids[mod_ids != ""].drop_duplicates()
    leg_set, mod_set = set(leg_ids), set(mod_ids)
    universe = sorted(leg_set | mod_set)
    idx = pd.Index(universe)
    assert idx.is_unique
    N = len(idx)
    log(f"legacy {len(leg_set):,} | modern {len(mod_set):,} | overlap {len(leg_set & mod_set):,} | universe {N:,}")
    if N != cfg.EXPECTED_UNIVERSE:
        log(f"WARNING: universe {N:,} != expected {cfg.EXPECTED_UNIVERSE:,}")
    in_leg, in_mod = idx.isin(leg_set), idx.isin(mod_set)
    src = np.where(in_leg & in_mod, "legacy_modern_overlap", np.where(in_leg, "legacy_only", "modern_only"))
    del in_leg, in_mod
    del leg_ids, mod_ids
    acc = Acc(N)

    # ---------------- 2. legacy ISR -> CASE (unambiguous only) ----------------
    lm = pd.read_csv(P / "legacy_case_isr_map.csv", dtype=str)
    lm["CASE"], lm["ISR"] = lm["CASE"].str.strip(), lm["ISR"].str.strip()
    lm = lm.drop_duplicates()
    per_isr = lm.groupby("ISR")["CASE"].nunique()
    ambiguous = set(per_isr[per_isr > 1].index)
    lm_ok = lm[~lm["ISR"].isin(ambiguous)].drop_duplicates("ISR")
    isr_pos = pd.Series(idx.get_indexer(lm_ok["CASE"]), index=lm_ok["ISR"].to_numpy())
    log(f"legacy ISR map: {len(lm):,} rows | ambiguous ISRs left unattributed: {len(ambiguous):,}")

    # ---------------- 3. legacy DRUG ----------------
    log("Legacy DRUG")
    ld = pd.read_csv(P / "legacy_drug_selected.csv", dtype=str, usecols=["ISR", "ROLE_COD", "DRUGNAME"])
    pos = isr_pos.reindex(ld["ISR"].str.strip().to_numpy()).fillna(-1).to_numpy(dtype=np.int64)
    comb = code_series(ld["DRUGNAME"], {}, False)
    role = ld["ROLE_COD"].fillna("").str.upper().str.strip().isin(cfg.SUSPECT_ROLES).to_numpy()
    acc.add_drugs(pos, comb, role)
    del ld

    # ---------------- 4. modern DRUG (chunked) ----------------
    log("Modern DRUG (this is the slow step)")
    want = {"caseid", "role_cod", "drugname", "prod_ai"}
    cache_n, cache_a = {}, {}
    n_rows = 0
    for ch in pd.read_csv(P / "modern_drug_selected.csv", dtype=str, chunksize=CHUNK,
                          usecols=lambda c: c.lower() in want):
        ch.columns = [c.lower() for c in ch.columns]
        pos = idx.get_indexer(ch["caseid"].fillna("").str.strip())
        comb = code_series(ch["drugname"], cache_n, False)
        if "prod_ai" in ch:
            comb = comb | (code_series(ch["prod_ai"], cache_a, True) & ~np.int64(1 << BIT_BLIND))
        role = ch["role_cod"].fillna("").str.upper().str.strip().isin(cfg.SUSPECT_ROLES).to_numpy()
        acc.add_drugs(pos, comb, role)
        n_rows += len(ch)
        log(f"   drug rows scanned: {n_rows:,}")

    # ---------------- 5. legacy REAC ----------------
    log("Legacy REAC")
    lr = pd.read_csv(P / "legacy_reac_selected.csv", dtype=str, usecols=["ISR", "PT"])
    pt = lr["PT"].fillna("").str.strip().str.lower()
    m = pt.isin(PT_BIT)
    pos = isr_pos.reindex(lr.loc[m, "ISR"].str.strip().to_numpy()).fillna(-1).to_numpy(dtype=np.int64)
    acc.add_pts(pos, pt[m].to_numpy())
    del lr

    # ---------------- 6. modern REAC (chunked) ----------------
    log("Modern REAC")
    n_rows = 0
    for ch in pd.read_csv(P / "modern_reac_selected.csv", dtype=str, chunksize=CHUNK,
                          usecols=lambda c: c.lower() in {"caseid", "pt"}):
        ch.columns = [c.lower() for c in ch.columns]
        pt = ch["pt"].fillna("").str.strip().str.lower()
        m = pt.isin(PT_BIT)
        if m.any():
            pos = idx.get_indexer(ch.loc[m, "caseid"].fillna("").str.strip())
            acc.add_pts(pos, pt[m].to_numpy())
        n_rows += len(ch)
        log(f"   reaction rows scanned: {n_rows:,}")

    # ---------------- 7. first FDA date ----------------
    log("Dates")
    date = np.zeros(N, np.int32)

    def put_dates(pos, vals):
        v = pd.to_numeric(vals, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        ok = (pos >= 0) & (v > 19000000)
        pos, v = pos[ok], v[ok].astype(np.int32)
        cur = date[pos]
        date[pos] = np.where(cur == 0, v, np.minimum(cur, v))

    ld_ = pd.read_csv(P / "legacy_aers_case_deduplicated.csv", dtype=str, usecols=["CASE", "FDA_DT"])
    put_dates(idx.get_indexer(ld_["CASE"].fillna("").str.strip()), ld_["FDA_DT"])
    del ld_
    for ch in pd.read_csv(P / "modern_demo_case_deduplicated.csv", dtype=str, chunksize=CHUNK,
                          usecols=lambda c: c.lower() in {"caseid", "init_fda_dt", "fda_dt"}):
        ch.columns = [c.lower() for c in ch.columns]
        pos = idx.get_indexer(ch["caseid"].fillna("").str.strip())
        last = pd.to_numeric(ch["fda_dt"], errors="coerce")
        first = pd.to_numeric(ch["init_fda_dt"], errors="coerce") if "init_fda_dt" in ch else last
        put_dates(pos, first.where(first.notna(), last))
    year = (date // 10000).astype(np.int16)
    quarter = (((date // 100) % 100 - 1) // 3 + 1).astype(np.int8)
    quarter[date == 0] = 0
    in_window = ((date >= cfg.STUDY_START) & (date <= cfg.STUDY_END)).astype(np.int8)

    # ---------------- 8. deleted cases ----------------
    log("FDA 'Deleted' lists")
    deleted_ids = set()
    n_lists = 0
    for z in sorted(RAW.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if "delete" in name.lower() and name.lower().endswith(".txt"):
                    n_lists += 1
                    txt = zf.read(name).decode("latin-1", errors="replace")
                    for ln in txt.splitlines():
                        tok = ln.strip().split("$")[0].strip()
                        if tok.isdigit():
                            deleted_ids.add(tok)
    deleted = np.zeros(N, np.int8)
    dpos = idx.get_indexer(list(deleted_ids))
    deleted[dpos[dpos >= 0]] = 1
    log(f"deleted lists: {n_lists} | distinct deleted IDs {len(deleted_ids):,} | in universe {int(deleted.sum()):,}")

    # ---------------- 9. derive columns ----------------
    log("Deriving flags")
    cols = {"caseid": idx.to_numpy(), "source_group": src, "deleted": deleted,
            "first_year": year, "first_quarter": quarter, "in_window": in_window}
    for i, a in enumerate(cfg.AGENT_LIST):
        cols[f"ag_{a}"] = ((acc.agent_all >> i) & 1).astype(np.int8)
    cols["agents_generic_mask"] = acc.agent_generic
    cols["agents_suspect_mask"] = acc.agent_suspect
    n_agents = np.zeros(N, np.int8)
    for i in range(NA):
        n_agents += ((acc.agent_all >> i) & 1).astype(np.int8)
    cols["n_agents"] = n_agents
    cols["ici9_any"] = (acc.agent_all > 0).astype(np.int8)
    cols["ici9_generic"] = (acc.agent_generic > 0).astype(np.int8)
    cols["ici9_suspect"] = (acc.agent_suspect > 0).astype(np.int8)
    cols["ici_nonblinded"] = acc.nonblind
    cols["newer_ici"] = acc.newer

    def has(pts):
        m = np.int64(0)
        for p in pts:
            m |= np.int64(1 << PT_BIT[p])
        return (acc.pt_bits & m) != 0

    for p in cfg.PRIMARY_PTS:
        cols[cfg.pt_slug(p)] = has([p]).astype(np.int8)
    cols["outcome_primary"] = has(cfg.PRIMARY_PTS).astype(np.int8)
    cols["outcome_immune"] = has(cfg.PRIMARY_PTS + cfg.IMMUNE_ADD).astype(np.int8)
    cols["outcome_broad"] = has(cfg.PRIMARY_PTS + cfg.IMMUNE_ADD + cfg.BROAD_ADD).astype(np.int8)
    cols["malig_inf_any"] = has(cfg.MALIGNANT_INFECTIVE).astype(np.int8)
    cols["myocarditis"] = has(cfg.MYOCARDITIS_PTS).astype(np.int8)
    cols["pt_bits"] = acc.pt_bits
    cols["other_l01"] = acc.l01

    # ---------------- 10. write ----------------
    log("Writing data/final/case_level_final.csv")
    df = pd.DataFrame(cols)
    path = OUT / "case_level_final.csv"
    if path.exists():
        path.unlink()
    step = 2_000_000
    for s in range(0, N, step):
        df.iloc[s:s + step].to_csv(path, mode="a", header=(s == 0), index=False)

    pd.DataFrame({"bit": range(len(cfg.ALL_TRACKED_PTS)), "pt": cfg.ALL_TRACKED_PTS}).to_csv(
        OUT / "pt_bit_dictionary.csv", index=False)
    write_dictionary()
    ici = df["ici9_any"] == 1
    out = df["outcome_primary"] == 1
    summ = [
        ("universe_cases", N), ("legacy_only", int((src == "legacy_only").sum())),
        ("modern_only", int((src == "modern_only").sum())),
        ("legacy_modern_overlap", int((src == "legacy_modern_overlap").sum())),
        ("deleted_cases_in_universe", int(deleted.sum())),
        ("ici9_any_role_cases", int(ici.sum())), ("ici9_generic_only_def_cases", int(df["ici9_generic"].sum())),
        ("ici9_suspect_cases", int(df["ici9_suspect"].sum())),
        ("newer_ici_cases", int(df["newer_ici"].sum())),
        ("newer_only_cases", int(((df["newer_ici"] == 1) & ~ici).sum())),
        ("outcome_primary_cases", int(out.sum())), ("ici9_and_outcome_primary", int((ici & out).sum())),
        ("ambiguous_legacy_ISRs_unattributed", len(ambiguous)),
    ]
    pd.DataFrame(summ, columns=["metric", "value"]).to_csv(OUT / "build_summary.csv", index=False)
    for k, v in summ:
        print(f"   {k:40s} {v:>12,}")
    log(f"DONE in {(time.time() - t0) / 60:.1f} min -> {path}")


def write_dictionary():
    rows = [
        ("caseid", "FAERS CASEID (modern) / legacy CASE. One row per unique case."),
        ("source_group", "legacy_only | modern_only | legacy_modern_overlap"),
        ("deleted", "1 = case appears in an FDA 'Deleted' list; excluded from the primary analysis"),
        ("first_year / first_quarter", "calendar quarter of earliest FDA date (modern: INIT_FDA_DT; legacy: FDA_DT of retained record)"),
        ("in_window", "1 = first FDA date within 2011-01-01..2026-06-30"),
        ("ag_<agent>", "1 = agent named in any drug row, ANY role, by generic OR brand OR development code name OR prod_ai"),
        ("agents_generic_mask", "bit i (AGENT_LIST order) = generic name in DRUGNAME only (old-style definition; prod_ai NOT used)"),
        ("agents_suspect_mask", "bit i = agent named on a PS/SS row"),
        ("n_agents", "number of distinct study agents (0-9)"),
        ("ici9_any / ici9_generic / ici9_suspect", "any study agent: any role / generic-only / suspect-only"),
        ("ici_nonblinded", "1 = at least one study-agent row NOT containing 'blinded'"),
        ("newer_ici", "1 = a non-study checkpoint inhibitor named (see study_config.NEWER_ICI_TERMS)"),
        ("pt_<term>", "1 = that primary-outcome PT present"),
        ("outcome_primary", "1 = any of the 7 primary PTs"),
        ("outcome_immune", "primary + myopericarditis + pleuropericarditis"),
        ("outcome_broad", "immune tier + all non-procedural pericardial PTs (incl. malignant/infective)"),
        ("malig_inf_any", "1 = any malignant/infective/other-etiology pericardial PT"),
        ("myocarditis", "1 = myocarditis / autoimmune myocarditis / immune-mediated myocarditis"),
        ("pt_bits", "bit mask of every tracked PT (see pt_bit_dictionary.csv)"),
        ("other_l01", "1 = a WHO ATC L01 antineoplastic (ICI names removed) named in drugname or prod_ai"),
    ]
    pd.DataFrame(rows, columns=["column", "definition"]).to_csv(OUT / "DATASET_DICTIONARY.csv", index=False)


if __name__ == "__main__":
    main()
