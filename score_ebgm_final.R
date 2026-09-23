#!/usr/bin/env Rscript
# score_ebgm_final.R
#
# Scores every agent x pericardial-PT cell with the Gamma-Poisson Shrinker (openEBGM 0.10.0).
# Adapted from your score_final_ebgm05.R (which already uses the correct API:
#   Qn() -> per-cell qn -> ebgm(qn = qn) and quantBisect(qn = qn)).
# Differences: no hard-coded cell counts; reads the new input; writes to data/final/results/.
#
# Run from the project root (after analyze_final.py):
#     Rscript scripts/score_ebgm_final.R
#
# HYPERPARAMETERS: re-uses the prior fitted on the full FAERS product-event table
# (data/processed/final_ebgm05_hyperparameters.csv). The empirical-Bayes prior is estimated
# from the whole database (~24M cells), which the recoding of ~150k ICI cases barely changes.
# State this in the Methods. To refit, rebuild the full universe and use the fit section of
# run_ebgm05.R only (do NOT use its scoring section).

suppressPackageStartupMessages({
  library(openEBGM)
  library(data.table)
})

TARGET_INPUT <- "data/final/results/gps_target_input_final.csv"
HYPER_INPUT  <- "data/processed/final_ebgm05_hyperparameters.csv"
OUT          <- "data/final/results/ebgm_final.csv"

h <- fread(HYPER_INPUT)
theta <- h$estimate
names(theta) <- h$parameter
cat("Hyperparameters (order must be alpha1, beta1, alpha2, beta2, P):\n")
print(theta)
if (length(theta) != 5 || any(!is.finite(theta))) stop("Bad hyperparameter file.")
if (!identical(tolower(names(theta)), c("alpha1", "beta1", "alpha2", "beta2", "p")))
  warning("Hyperparameter labels are ", paste(names(theta), collapse = ", "),
          " - confirm they are in the order alpha1, beta1, alpha2, beta2, P (Qn() uses position).")

target <- fread(TARGET_INPUT)
target[, `:=`(var1 = as.character(var1), var2 = as.character(var2),
              N = as.numeric(N), E = as.numeric(E))]
cat("\nTarget cells:", nrow(target),
    "| N>0 & E>0:", sum(target$N > 0 & target$E > 0),
    "| N=0 & E>0:", sum(target$N == 0 & target$E > 0),
    "| E=0 (event absent from universe; not estimable):", sum(target$E == 0), "\n")

est <- target[E > 0]
qn <- Qn(theta_hat = theta, N = est$N, E = est$E)
if (any(!is.finite(qn)) || any(qn < 0 | qn > 1)) stop("Invalid Qn.")

est[, Qn     := as.numeric(qn)]
est[, EBGM   := as.numeric(ebgm(theta_hat = theta, N = N, E = E, qn = qn, digits = 10))]
est[, EBGM05 := as.numeric(quantBisect(5,  theta_hat = theta, N = N, E = E, qn = qn, digits = 10))]
est[, EBGM95 := as.numeric(quantBisect(95, theta_hat = theta, N = N, E = E, qn = qn, digits = 10))]

stopifnot(all(is.finite(est$EBGM), is.finite(est$EBGM05), is.finite(est$EBGM95)))
if (any(est$EBGM05 > est$EBGM + 1e-8) || any(est$EBGM > est$EBGM95 + 1e-8))
  stop("Ordering violated: need EBGM05 <= EBGM <= EBGM95.")
cat("Validation passed: Qn in [0,1]; EBGM05 <= EBGM <= EBGM95 for all", nrow(est), "cells.\n")

est[, RR := N / E]   # raw observed/expected, for sanity: shrunken EBGM should sit below RR when RR > 1

non <- target[E == 0][, `:=`(Qn = NA_real_, EBGM = NA_real_, EBGM05 = NA_real_, EBGM95 = NA_real_, RR = NA_real_)]
out <- rbind(est, non, fill = TRUE)
out[, signal_EBGM05_gt_2 := !is.na(EBGM05) & EBGM05 > 2 & N >= 3]
setorder(out, -EBGM05, var1, var2)
fwrite(out, OUT)

cat("\nSaved:", OUT, "\n\nCells with N>=3 and EBGM05 > 2:\n")
print(out[signal_EBGM05_gt_2 == TRUE, .(var1, var2, N, E = round(E, 3), RR = round(RR, 2),
                                         EBGM = round(EBGM, 2), EBGM05 = round(EBGM05, 2),
                                         EBGM95 = round(EBGM95, 2))], row.names = FALSE)
