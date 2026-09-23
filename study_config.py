"""
study_config.py  -  SINGLE SOURCE OF TRUTH for every definition in the final analysis.

Nothing in build_final_dataset.py / analyze_final.py hard-codes a drug or PT list;
everything comes from here.  Edit here (and only here) if a definition changes,
then re-run the pipeline.

DERIVATION NOTES (for the Methods section)
  * PT lists were derived by listing every distinct PT containing "pericard" or
    "tamponade" in the harmonized FAERS data (audit section 2) and classifying each one.
  * Drug-name matching is done on NORMALISED strings (upper-case, every run of
    non-alphanumeric characters -> one space) using whole-word / whole-phrase matching.
  * Development code names below are from public INN/code-name tables; the authors
    should spot-check them before submission.
"""
import re

# ----------------------------------------------------------------------------
# ICI agents.  generic = INN;  extra = brand names + development code names.
# ----------------------------------------------------------------------------
AGENTS = {
    "ipilimumab":    {"generic": ["ipilimumab"],
                      "extra": ["yervoy", "mdx 010", "mdx010", "bms 734016", "bms734016"]},
    "tremelimumab":  {"generic": ["tremelimumab"],
                      "extra": ["imjudo", "ticilimumab", "cp 675 206", "cp 675206", "cp675206"]},
    "nivolumab":     {"generic": ["nivolumab"],
                      "extra": ["opdivo", "opdualag", "bms 936558", "bms936558",
                                "ono 4538", "ono4538", "mdx 1106", "mdx1106"]},
    "pembrolizumab": {"generic": ["pembrolizumab"],
                      "extra": ["keytruda", "mk 3475", "mk3475", "lambrolizumab", "sch 900475"]},
    "cemiplimab":    {"generic": ["cemiplimab"],
                      "extra": ["libtayo", "regn2810", "regn 2810"]},
    "dostarlimab":   {"generic": ["dostarlimab"],
                      "extra": ["jemperli", "tsr 042", "tsr042"]},
    "atezolizumab":  {"generic": ["atezolizumab"],
                      "extra": ["tecentriq", "mpdl3280a", "rg7446", "rg 7446"]},
    "durvalumab":    {"generic": ["durvalumab"],
                      "extra": ["imfinzi", "medi4736", "medi 4736"]},
    "avelumab":      {"generic": ["avelumab"],
                      "extra": ["bavencio", "msb0010718c", "msb 0010718c"]},
}
AGENT_LIST = list(AGENTS)          # bit i of every agent mask = AGENT_LIST[i]

# Other checkpoint inhibitors NOT among the nine study agents (non-exhaustive).
# Cases exposed ONLY to these are excluded from the primary analysis (neither exposed
# nor reference) and treated as ICI in the class-wide sensitivity analysis.
NEWER_ICI_TERMS = [
    "retifanlimab", "zynyz", "toripalimab", "loqtorzi", "tislelizumab", "tevimbra",
    "cosibelimab", "unloxcyt", "penpulimab", "relatlimab", "sintilimab", "tyvyt",
    "camrelizumab", "sugemalimab", "serplulimab", "envafolimab", "zimberelimab",
    "cadonilimab", "spartalizumab", "balstilimab", "prolgolimab",
]

BLINDED_TERMS = ["blinded", "blind"]      # 'blinded X' = study drug possibly placebo
SUSPECT_ROLES = {"PS", "SS"}              # primary / secondary suspect
# Primary exposure counts ANY drug role (PS, SS, C, I, DN); suspect-only is a sensitivity analysis.

# ----------------------------------------------------------------------------
# Pericardial outcome (MedDRA Preferred Terms, lower-cased exact strings)
# ----------------------------------------------------------------------------
PRIMARY_PTS = [
    "pericarditis",
    "pericardial effusion",
    "cardiac tamponade",
    "pericarditis constrictive",       # correct MedDRA PT (NOT "constrictive pericarditis")
    "autoimmune pericarditis",
    "pericardial disease",
    "immune-mediated pericarditis",
]
IMMUNE_ADD = ["myopericarditis", "pleuropericarditis"]

BROAD_ADD = [
    "pericardial haemorrhage", "pneumopericardium", "pericardial fibrosis",
    "pericarditis adhesive", "pericardial calcification", "intrapericardial thrombosis",
    "pericardial mass", "pericardial cyst", "pericardial rub", "postpericardiotomy syndrome",
    "pericardial rupture", "pericardial injury", "pericardial perforation",
    "pericardiocutaneous fistula", "pericardial lipoma", "benign pericardium neoplasm",
    "camptodactyly-arthropathy-coxa vara-pericarditis syndrome",
    # malignant / infective / other-etiology (also used for the exclusion sensitivity):
    "pericardial effusion malignant", "pericarditis malignant", "pericardial neoplasm",
    "malignant pericardial neoplasm", "pericardial mesothelioma malignant",
    "pericardial mesothelioma malignant recurrent",
    "pericarditis infective", "infective pericardial effusion", "viral pericarditis",
    "purulent pericarditis", "bacterial pericarditis", "pericarditis tuberculous",
    "pericarditis fungal", "coxsackie pericarditis", "pericarditis mycoplasmal",
    "cytomegalovirus pericarditis", "pericarditis meningococcal", "pericarditis histoplasma",
    "atypical mycobacterium pericarditis", "radiation pericarditis", "pericarditis uraemic",
    "pericarditis rheumatic", "pericarditis lupus",
]
MALIGNANT_INFECTIVE = [
    "pericardial effusion malignant", "pericarditis malignant", "pericardial neoplasm",
    "malignant pericardial neoplasm", "pericardial mesothelioma malignant",
    "pericardial mesothelioma malignant recurrent",
    "pericarditis infective", "infective pericardial effusion", "viral pericarditis",
    "purulent pericarditis", "bacterial pericarditis", "pericarditis tuberculous",
    "pericarditis fungal", "coxsackie pericarditis", "pericarditis mycoplasmal",
    "cytomegalovirus pericarditis", "pericarditis meningococcal", "pericarditis histoplasma",
    "atypical mycobacterium pericarditis", "radiation pericarditis", "pericarditis uraemic",
    "pericarditis rheumatic", "pericarditis lupus",
]
# Deliberately NOT outcomes (documented exclusions): procedures and non-cardiac tamponade.
EXCLUDED_PTS = [
    "pericardial drainage", "pericardial excision", "pericardiotomy", "pericardial operation",
    "pericardial cyst resection", "biopsy pericardium abnormal",
    "pericardial decompression syndrome", "bladder tamponade",
]
MYOCARDITIS_PTS = ["myocarditis", "autoimmune myocarditis", "immune-mediated myocarditis"]

ALL_TRACKED_PTS = []
for _p in PRIMARY_PTS + IMMUNE_ADD + BROAD_ADD + MYOCARDITIS_PTS:
    if _p not in ALL_TRACKED_PTS:
        ALL_TRACKED_PTS.append(_p)
assert len(ALL_TRACKED_PTS) <= 62, "PT bit-mask needs <= 62 terms"
assert set(MALIGNANT_INFECTIVE) <= set(BROAD_ADD)
assert not (set(EXCLUDED_PTS) & set(ALL_TRACKED_PTS))

STUDY_START, STUDY_END = 20110101, 20260630     # temporal window (yyyymmdd)
EXPECTED_UNIVERSE = 18_629_240                  # harmonized case universe (legacy U modern)


# ----------------------------------------------------------------------------
# helpers shared by the builder
# ----------------------------------------------------------------------------
def normalize_name(s):
    """UPPER, non-alphanumeric runs -> single space, strip, lower."""
    return re.sub(r"[^A-Z0-9]+", " ", str(s).upper()).strip().lower()


def phrase_regex(terms):
    ts = sorted({normalize_name(t) for t in terms if normalize_name(t)}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, ts)) + r")(?![a-z0-9])")


def pt_slug(pt):
    return "pt_" + re.sub(r"[^a-z0-9]+", "_", pt.lower()).strip("_")
