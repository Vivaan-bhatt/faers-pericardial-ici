"""Sex and age analysis for the FAERS pericardial / ICI study.

Run from the project root (the folder that contains data/):
    python3 sex_age_analysis.py
Optional: python3 sex_age_analysis.py /path/to/project_root

Reads   data/final/case_level_final.csv
        data/processed/modern_demo_case_deduplicated.csv
        data/processed/legacy_aers_case_deduplicated.csv
Writes  data/final/results/sex_age_*.csv   and prints everything to the screen.
"""
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FINAL = ROOT / "data/final/case_level_final.csv"
MOD = ROOT / "data/processed/modern_demo_case_deduplicated.csv"
LEG = ROOT / "data/processed/legacy_aers_case_deduplicated.csv"
OUT = ROOT / "data/final/results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- study population (same as the primary analysis) ----------------
cols = ["caseid", "source_group", "deleted", "ici9_any", "newer_ici", "outcome_primary"]
df = pd.read_csv(FINAL, usecols=cols, dtype={"caseid": str})
df["caseid"] = df["caseid"].str.strip()
df = df[df["deleted"] == 0]
df = df[~((df["newer_ici"] == 1) & (df["ici9_any"] == 0))].copy()
E = df["ici9_any"] == 1
print("Analysis population:", len(df), "| exposed:", int(E.sum()),
      "| exposed events:", int(df.loc[E, "outcome_primary"].sum()),
      "| reference:", int((~E).sum()), "| reference events:", int(df.loc[~E, "outcome_primary"].sum()))
print("(expected: 18,514,045 | 277,434 | 1,680 | 18,236,611 | 33,112)\n")

# ---------------- demographics ----------------
def to_years(age, cod):
    a = pd.to_numeric(age, errors="coerce")
    c = cod.fillna("").astype(str).str.upper().str.strip()
    f = c.map({"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52.1775,
               "DY": 1 / 365.25, "HR": 1 / 8766.0})
    yrs = a * f
    yrs = yrs.where((yrs >= 0) & (yrs <= 120))
    return yrs

def sex_clean(x):
    x = x.fillna("").astype(str).str.upper().str.strip()
    x = x.replace({"FEMALE": "F", "MALE": "M"})
    return x.where(x.isin(["F", "M"]), "Unknown")

m = pd.read_csv(MOD, dtype=str, usecols=lambda c: c.lower() in
                {"caseid", "age", "age_cod", "gndr_cod", "sex", "caseversion_num"})
m.columns = [c.lower() for c in m.columns]
if "caseversion_num" in m:
    m["_v"] = pd.to_numeric(m["caseversion_num"], errors="coerce")
    m = m.sort_values("_v")
m["caseid"] = m["caseid"].str.strip()
m = m.drop_duplicates("caseid", keep="last")
print("Modern gndr_cod values:", m["gndr_cod"].fillna("(blank)").str.upper().value_counts().head(8).to_dict() if "gndr_cod" in m else "column missing")
print("Modern sex values:     ", m["sex"].fillna("(blank)").str.upper().value_counts().head(8).to_dict() if "sex" in m else "column missing")
s_old = sex_clean(m["gndr_cod"]) if "gndr_cod" in m else pd.Series("Unknown", index=m.index)
s_new = sex_clean(m["sex"]) if "sex" in m else pd.Series("Unknown", index=m.index)
m = pd.DataFrame({"caseid": m["caseid"], "sex": s_new.where(s_new != "Unknown", s_old),
                  "age": to_years(m["age"], m["age_cod"])})
print("Modern file: rows", len(m), "| sex known %.1f%% | age known %.1f%%" % (100 * (m.sex != "Unknown").mean(), 100 * m.age.notna().mean()))

l = pd.read_csv(LEG, dtype=str, usecols=lambda c: c.upper() in
                {"CASE", "AGE", "AGE_COD", "GNDR_COD", "FOLL_SEQ_NUM"})
l.columns = [c.upper() for c in l.columns]
if "FOLL_SEQ_NUM" in l:
    l["_v"] = pd.to_numeric(l["FOLL_SEQ_NUM"], errors="coerce").fillna(0)
    l = l.sort_values("_v")
l["CASE"] = l["CASE"].str.strip()
l = l.drop_duplicates("CASE", keep="last")
l = pd.DataFrame({"caseid": l["CASE"], "sex": sex_clean(l["GNDR_COD"]),
                  "age": to_years(l["AGE"], l["AGE_COD"])})

print("Legacy file: rows", len(l), "| sex known %.1f%% | age known %.1f%%" % (100 * (l.sex != "Unknown").mean(), 100 * l.age.notna().mean()))
# modern record wins for cases present in both systems
demo = pd.concat([m, l[~l["caseid"].isin(set(m["caseid"]))]], ignore_index=True)
df = df.merge(demo, on="caseid", how="left")
df["sex"] = df["sex"].fillna("Unknown")
bins = [-0.001, 17.999, 44.999, 64.999, 74.999, 200]
labels = ["<18", "18-44", "45-64", "65-74", ">=75"]
df["age_grp"] = pd.cut(df["age"], bins=bins, labels=labels).astype(object)
df["age_grp"] = df["age_grp"].fillna("Unknown")
df["exposed"] = (df["ici9_any"] == 1).astype(int)
df["event"] = df["outcome_primary"].astype(int)

# ---------------- completeness ----------------
rows = []
for name, mask in [("ICI-exposed", df.exposed == 1), ("Reference", df.exposed == 0),
                   ("All", df.exposed >= 0)]:
    s = df[mask]
    rows.append({"group": name, "cases": len(s),
                 "sex_known_pct": round(100 * (s.sex != "Unknown").mean(), 1),
                 "age_known_pct": round(100 * (s.age_grp != "Unknown").mean(), 1),
                 "female_pct_of_known_sex": round(100 * (s.sex == "F").sum() / max((s.sex != "Unknown").sum(), 1), 1),
                 "median_age_known": round(float(s["age"].median()), 1)})
comp = pd.DataFrame(rows)
print("COMPLETENESS\n", comp.to_string(index=False), "\n")
comp.to_csv(OUT / "sex_age_completeness.csv", index=False)

# ---------------- ROR within strata ----------------
def ror(a, b, c, d):
    """a exposed events, b exposed non-events, c ref events, d ref non-events"""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    lr = np.log(a * d / (b * c)); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return np.exp(lr), np.exp(lr - 1.96 * se), np.exp(lr + 1.96 * se), lr, se

def stratum(label, s):
    ex, rf = s[s.exposed == 1], s[s.exposed == 0]
    a, c = int(ex.event.sum()), int(rf.event.sum())
    b, d = len(ex) - a, len(rf) - c
    r, lo, hi, lr, se = ror(a, b, c, d)
    return {"stratum": label, "exposed_cases": len(ex), "exposed_events": a,
            "reference_cases": len(rf), "reference_events": c,
            "pct_event_exposed": round(100 * a / max(len(ex), 1), 2),
            "pct_event_reference": round(100 * c / max(len(rf), 1), 2),
            "ROR": round(r, 2), "CI_low": round(lo, 2), "CI_high": round(hi, 2),
            "_lr": lr, "_se": se}

res = []
for sx in ["F", "M", "Unknown"]:
    res.append(stratum({"F": "Female", "M": "Male", "Unknown": "Sex unknown"}[sx], df[df.sex == sx]))
for g in labels + ["Unknown"]:
    res.append(stratum("Age " + g if g != "Unknown" else "Age unknown", df[df.age_grp == g]))
R = pd.DataFrame(res)
print("ROR WITHIN STRATA (ICI vs reference cases of the same sex / age group)\n",
      R.drop(columns=["_lr", "_se"]).to_string(index=False), "\n")
R.drop(columns=["_lr", "_se"]).to_csv(OUT / "sex_age_ror_by_stratum.csv", index=False)

# ---------------- difference between women and men ----------------
f = R[R.stratum == "Female"].iloc[0]; mm = R[R.stratum == "Male"].iloc[0]
d_lr = f._lr - mm._lr; d_se = np.sqrt(f._se ** 2 + mm._se ** 2)
print("Ratio of RORs, female vs male: %.2f (95%% CI %.2f to %.2f); z-test p = %.3f" % (
    np.exp(d_lr), np.exp(d_lr - 1.96 * d_se), np.exp(d_lr + 1.96 * d_se),
    math.erfc(abs(d_lr / d_se) / math.sqrt(2))))

# sex-adjusted (Mantel-Haenszel) ROR across known-sex strata, as a check
num = den = 0.0
for sx in ["F", "M"]:
    s = df[df.sex == sx]; ex, rf = s[s.exposed == 1], s[s.exposed == 0]
    a, c = ex.event.sum(), rf.event.sum(); b, d = len(ex) - a, len(rf) - c; n = len(s)
    num += a * d / n; den += b * c / n
print("Mantel-Haenszel ROR across female and male strata: %.2f" % (num / den))

# pericardial events per sex among exposed (share of events that are in women)
ev = df[(df.exposed == 1) & (df.event == 1)]
print("Exposed cases with a pericardial event: female %d, male %d, unknown %d" % (
    (ev.sex == "F").sum(), (ev.sex == "M").sum(), (ev.sex == "Unknown").sum()))
print("\nSaved CSVs in", OUT)
