# FAERS pericardial adverse events with immune checkpoint inhibitors: analysis code

Code for "Reporting of Pericardial Adverse Events with Immune Checkpoint Inhibitors: A Disproportionality Analysis of the FDA Adverse Event Reporting System" (preprint: https://doi.org/10.64898/2026.10.19.26365946).

## Files

| File | Purpose |
|---|---|
| `study_config.py` | Every drug name, MedDRA term list and definition used in the analysis (single source of truth) |
| `build_final_dataset.py` | Builds one case-level dataset from the processed AERS and FAERS tables |
| `analyze_final.py` | Computes every statistic in the paper from that dataset |
| `smq_final.py` | Standardized MedDRA query sensitivity analysis |
| `score_ebgm_final.R` | Empirical Bayes (gamma-Poisson shrinker) scoring with openEBGM 0.10.0 |
| `sex_age_analysis.py` | Sex and age analysis (reporting odds ratios by sex and age group) |

## Order of use

Run each script from the project root, in this order:

1. `python3 build_final_dataset.py`
2. `python3 analyze_final.py`
3. `python3 smq_final.py`
4. `Rscript score_ebgm_final.R`
5. `python3 sex_age_analysis.py`

The scripts read tables in `data/processed/` and FDA "Deleted" case lists in `data/raw/`. The FDA quarterly AERS and FAERS files are public: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

## Requirements

Python 3 with pandas and NumPy; R with openEBGM 0.10.0 and data.table.

## Citation

See `CITATION.cff`.

## Licence

MIT (see `LICENSE`).
