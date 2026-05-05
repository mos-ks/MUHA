# =============================================================================
# Supplementary Code — Statistical Analysis
# Localization Uncertainty Visualization Study
#
# Manuscript: "From Model Uncertainty to Human Attention: Localization-Aware
#             Visual Cues for Scalable Annotation Review"
#             Under review.
#
# Description:
#   Reproduces all statistical results reported in the main text and
#   supplementary materials. The section numbering below maps onto the
#   manuscript as follows:
#
#   Code Section   Manuscript Location
#   ------------   -----------------------------------------------------------
#   1              Methods > Statistical Analysis (data preparation)
#   2              Results > "Localization Uncertainty Improves Annotation
#                  Quality and Efficiency" (primary t-tests; reported values:
#                  quality t(117.4) = 2.19, p = 0.015, d = 0.4;
#                  efficiency t(117.4) = -1.714, p = 0.045, d = -0.313;
#                  7.2% time reduction)
#   3              Results > same subsection (LME robustness;
#                  quality beta = 0.70, SE = 0.32, t = 2.20, p = 0.030;
#                  efficiency beta = -1.93, SE = 1.12, p = 0.086)
#   4              Results > "The Attention Guidance Effect: Benefits Scale
#                  with Image Difficulty" (per-bin t-tests and LME;
#                  mid+high: quality d = 0.49, p = 0.004;
#                  efficiency d = -0.37, p = 0.022)
#   5              Results > "Cognitive Load and Treatment Effects"
#                  (moderation LME; Wald tests; predicted advantages at ±1 SD)
#   6              Results > "Cognitive Load and Treatment Effects"
#                  (task satisfaction Mann-Whitney U; U = 1985, p = 0.331)
#   7              Methods > Robustness Checks; Supplementary Tables S1–S2
#   8              Results > "Cognitive Load and Treatment Effects"
#                  (exploratory ±1 SD predictions; ~1.05 vs ~0.28 mIoU pp)
#   9              Figures 1–3 in the main text
#
# Input:
#   data/trial_level.csv   — 1,800 trials (120 participants × 15 images)
#   data/boxes_change.csv  — bounding-box-level change data (Figure 3)
#
# Note on hypothesis directionality:
#   Primary t-tests (Sections 2 and 4) are one-tailed, consistent with the
#   pre-specified directional hypotheses: H1 (annotation quality is higher in
#   the uncertainty_vis condition) and H2 (editing time is lower). LME
#   p-values from lmerTest are two-tailed by Satterthwaite default. Where
#   both test types appear together this distinction is noted explicitly.
#
# Dependencies: tidyverse, lme4, lmerTest, coin, cowplot, geepack, emmeans
# R version: 4.4.1
# =============================================================================

library(tidyverse)
library(lme4)
library(lmerTest)
library(coin)
library(cowplot)
library(geepack)
library(emmeans)
library(here)


# =============================================================================
# Helper functions
# =============================================================================

# Print a top-level section banner to console output.
section <- function(title, subtitle = NULL) {
  cat("\n", strrep("=", 70), "\n")
  cat(" ", title, "\n")
  if (!is.null(subtitle)) cat("  ", subtitle, "\n")
  cat(strrep("=", 70), "\n\n")
}

# Print a sub-section banner.
subsection <- function(title) {
  cat("\n  ---", title, "---\n\n")
}

# Compute pooled Cohen's d for two independent groups.
cohens_d <- function(a, b) {
  n_a <- length(a)
  n_b <- length(b)
  pooled_sd <- sqrt(
    ((n_a - 1) * var(a) + (n_b - 1) * var(b)) / (n_a + n_b - 2)
  )
  (mean(a) - mean(b)) / pooled_sd
}

# Run a one-tailed Welch t-test and print a formatted summary with Cohen's d.
# `alternative` follows the convention of stats::t.test ("greater" or "less").
one_tailed_t <- function(a, b, alternative = "greater", label = "") {
  res <- t.test(a, b, alternative = alternative, var.equal = FALSE)
  d   <- cohens_d(a, b)
  cat(sprintf("  %s\n", label))
  cat(sprintf("    Uncertainty Vis. : M = %.4f, SD = %.4f  (n = %d)\n",
              mean(a), sd(a), length(a)))
  cat(sprintf("    Baseline         : M = %.4f, SD = %.4f  (n = %d)\n",
              mean(b), sd(b), length(b)))
  cat(sprintf("    t(%.1f) = %.3f,  p (one-tailed) = %.3f,  d = %.3f\n\n",
              res$parameter, res$statistic, res$p.value, d))
  invisible(list(t = res$statistic, df = res$parameter, p = res$p.value, d = d))
}

# Extract and print the treatment_dummy fixed-effect row from an LME summary.
# `tails` is a descriptive string appended to the output for clarity.
print_lme_treatment <- function(model, label = "", tails = "two-tailed") {
  coef_tbl <- as.data.frame(coef(summary(model)))
  row      <- coef_tbl["treatment_dummy", ]
  cat(sprintf("  %s\n", label))
  cat(sprintf("    beta = %.3f,  SE = %.3f,  t = %.3f,  p = %.3f (%s)\n\n",
              row$Estimate, row$`Std. Error`, row$`t value`, row$`Pr(>|t|)`,
              tails))
}


# =============================================================================
# 1. Data loading and preparation
# =============================================================================
# Manuscript reference: Methods > Dataset and Annotations; Methods >
# Statistical Analysis. The participant-level aggregation produced here is
# the unit of analysis for all primary t-tests (Section 2). Trial-level data
# are passed directly to LME models in Sections 3–7.
# =============================================================================

section(
  "1. DATA PREPARATION",
  "Source: data/trial_level.csv (1,800 trials; 120 participants × 15 images)"
)

trials <- read_csv(here("analysis", "00_data", "trial_level.csv"), show_col_types = FALSE)

condition_levels <- c("baseline", "uncertainty_vis")
bin_levels       <- c("low", "mid", "high")

# Encode condition as an ordered factor and create a numeric dummy (1 =
# uncertainty_vis) for use in LME formulas, which require numeric predictors
# when the interaction term is of interest.
trials <- trials |>
  mutate(
    treatment       = factor(treatment, levels = condition_levels),
    bin             = factor(bin,       levels = bin_levels),
    treatment_dummy = as.integer(treatment == "uncertainty_vis")
  )

# Aggregate to participant-level means for primary t-tests (Section 2).
# LME models in later sections operate on the full trial-level data.
participants <- trials |>
  group_by(participant_id, treatment, treatment_dummy) |>
  summarise(
    mean_mIoU_user    = mean(mIoU_user,        na.rm = TRUE),
    mean_editing_time = mean(editing_time,      na.rm = TRUE),
    cognitive_load    = mean(cognitive_load,    na.rm = TRUE),
    task_satisfaction = mean(task_satisfaction, na.rm = TRUE),
    self_efficacy     = mean(self_efficacy,     na.rm = TRUE),
    task_familiarity  = mean(task_familiarity,  na.rm = TRUE),
    .groups = "drop"
  )

treat_p <- participants |> filter(treatment == "uncertainty_vis")
base_p  <- participants |> filter(treatment == "baseline")
treat_t <- trials       |> filter(treatment == "uncertainty_vis")
base_t  <- trials       |> filter(treatment == "baseline")

cat(sprintf("  Participants loaded : %d  (%d uncertainty_vis, %d baseline)\n",
            nrow(participants), nrow(treat_p), nrow(base_p)))
cat(sprintf("  Trials loaded       : %d  (%d per condition × 15 images)\n",
            nrow(trials), nrow(treat_t)))
cat(sprintf("  Difficulty bins     : %s\n", paste(bin_levels, collapse = ", ")))


# =============================================================================
# 2. Primary analysis — annotation quality and efficiency
# =============================================================================
# Manuscript reference: Results > "Localization Uncertainty Improves
# Annotation Quality and Efficiency" (Annotation Quality and Efficiency
# paragraphs). Reported values from this section:
#   Quality:    t(117.4) = 2.19, p = 0.015, d = 0.4
#   Efficiency: t(117.4) = -1.714, p = 0.045, d = -0.313
#   Time reduction: 7.2% (26.69 s to 24.76 s)
#
# H1: mean mIoU_user is higher in the uncertainty_vis condition (one-tailed).
# H2: mean editing_time is lower in the uncertainty_vis condition (one-tailed).
#
# The unit of analysis is the participant-level mean, making each observation
# independent. Alpha = 0.05.
# =============================================================================

section(
  "2. PRIMARY ANALYSIS: Annotation Quality and Efficiency",
  "Unit: participant-level means | Test: one-tailed Welch t-test | alpha = 0.05"
)

subsection("2a. Annotation quality (mean mIoU per participant)")
cat("  Hypothesis: uncertainty_vis > baseline\n\n")
one_tailed_t(
  treat_p$mean_mIoU_user, base_p$mean_mIoU_user,
  alternative = "greater",
  label       = "Uncertainty Vis. vs. Baseline"
)

subsection("2b. Annotation efficiency (mean editing time per participant)")
cat("  Hypothesis: uncertainty_vis < baseline (faster = better)\n\n")
one_tailed_t(
  treat_p$mean_editing_time, base_p$mean_editing_time,
  alternative = "less",
  label       = "Uncertainty Vis. vs. Baseline"
)

pct_reduction <- (
  mean(base_p$mean_editing_time) - mean(treat_p$mean_editing_time)
) / mean(base_p$mean_editing_time) * 100
cat(sprintf("  Mean time reduction: %.1f%%\n", pct_reduction))


# =============================================================================
# 3. LME robustness models — quality and efficiency
# =============================================================================
# Manuscript reference: Results > "Localization Uncertainty Improves
# Annotation Quality and Efficiency" (paragraphs beginning "To verify the
# robustness of this result"). Reported values from this section:
#   Quality:    beta = 0.70, SE = 0.32, t = 2.20, p = 0.030 (two-tailed)
#   Efficiency: beta = -1.93, SE = 1.12, t = -1.73, p = 0.086 (two-tailed)
#
# Linear mixed-effects models with a random intercept per participant account
# for the repeated-measures structure. Fitted by ML (REML = FALSE) to allow
# likelihood-ratio comparisons in later sections. p-values are two-tailed
# (Satterthwaite approximation via lmerTest). These models complement rather
# than replace the one-tailed t-tests in Section 2.
# =============================================================================

section(
  "3. LME ROBUSTNESS MODELS",
  "Formula: outcome ~ treatment_dummy + (1 | participant_id) | ML | p two-tailed"
)

subsection("3a. Annotation quality (mIoU_user)")
lme_quality <- lmer(
  mIoU_user ~ treatment_dummy + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_quality))
print_lme_treatment(lme_quality, "Treatment effect on quality (two-tailed)", "two-tailed")

subsection("3b. Annotation efficiency (editing_time)")
# p-value is two-tailed; directionally consistent with the Section 2b t-test.
lme_time <- lmer(
  editing_time ~ treatment_dummy + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_time))
print_lme_treatment(lme_time, "Treatment effect on efficiency (two-tailed)", "two-tailed")


# =============================================================================
# 4. Guidance effect — box-level attention and performance by difficulty
# =============================================================================
# Manuscript reference: Results > "The Attention Guidance Effect: Benefits
# Scale with Image Difficulty". This section mirrors the two-part structure
# of that subsection in the manuscript:
#   4a: Box-level GEE (Figure 3) — establishes that the uncertainty signal
#       successfully redirects attention toward high-uncertainty predictions.
#   4b/4c: Per-bin t-tests (Figure 4) — show that image-level gains scale
#       with difficulty, consistent with the attentional guidance account.
#   4d: Mid+high secondary analysis (Supplementary Table S2) — confirms
#       effects sharpen when restricted to theoretically informative images.
#
# Per-bin reported values:
#   Low:  quality p = 0.227, d = 0.14;  efficiency p = 0.20,  d = -0.15
#   Mid:  quality p = 0.002, d = 0.53;  efficiency p = 0.064, d = -0.28
#   High: quality p = 0.055, d = 0.29;  efficiency p = 0.018, d = -0.386
# Mid+high secondary analysis (Supplementary Table S2):
#   Quality:    t(117.9) = 2.69, p = 0.004, d = 0.49; LME beta = 0.87, p = 0.008
#   Efficiency: t(117.7) = -2.04, p = 0.022, d = -0.37; LME beta = -2.37, p = 0.042
# =============================================================================

section("4. GUIDANCE EFFECT: Box-Level Attention and Performance by Difficulty")

# -----------------------------------------------------------------------------
# 4a. Box-level GEE — probability of bounding box change
# -----------------------------------------------------------------------------
# Manuscript reference: Results > "The Attention Guidance Effect: Benefits
# Scale with Image Difficulty" (first part: "Treatment successfully guides
# attention to uncertain boxes"; Figure 3). Reported values:
#   Condition main effect:    beta = -0.8192, p < .001
#   Uncertainty main effect:  beta = -0.4321, p < .001
#   Interaction:              beta =  0.695, p < .001  (net: 0.695 - 0.432  = 0.263)
#
# GEE models the marginal probability that an annotator modifies a given
# bounding box as a function of log-transformed aleatoric uncertainty.
# Exchangeable working correlation assumes equal within-annotator correlation
# across boxes; sandwich-corrected SEs account for within-cluster dependence.
# Predicted probabilities for Figure 3 are computed via emmeans.
# -----------------------------------------------------------------------------

subsection("4a. Box-level GEE: probability of bounding box change")

boxes <- read_csv(
  here("analysis","00_data", "boxes_change.csv"),
  show_col_types = FALSE
)

boxes <- boxes |>
  select(-entropy) |>
  mutate(
    treatment   = factor(treatment, levels = c("baseline", "uncertainty")),
    log_albox   = log1p(albox_mean),
    participant_id = as.factor(participant_id)
  )

cat(sprintf("  Boxes loaded: %d observations, %d annotators\n\n",
            nrow(boxes), n_distinct(boxes$participant_id)))

gee_mod <- geeglm(
  changed ~ log_albox * treatment,
  data   = boxes,
  id     = participant_id,
  family = binomial,
  corstr = "exchangeable"
)
cat("  GEE model summary (exchangeable correlation, sandwich SEs):\n")
print(summary(gee_mod))

# Marginal predicted probabilities across the uncertainty range, used in
# Figure 3. length.out = 100 gives smooth curves without excess computation.
predictions <- emmeans(
  gee_mod, ~ log_albox * treatment,
  at   = list(log_albox = seq(
    min(boxes$log_albox, na.rm = TRUE),
    max(boxes$log_albox, na.rm = TRUE),
    length.out = 100
  )),
  type = "response"
) |>
  as_tibble() |>
  mutate(
    Condition = ifelse(treatment == "uncertainty", "Uncertainty Visualization", "Baseline"),
    Condition = factor(Condition, levels = c("Baseline", "Uncertainty Visualization"))
  )

subsection("4b. Annotation quality — one-tailed t-tests per bin")
cat("  Unit: participant-level mean mIoU within each bin\n\n")

for (b in bin_levels) {
  bin_means <- trials |>
    filter(bin == b) |>
    group_by(participant_id, treatment) |>
    summarise(mean_mIoU = mean(mIoU_user, na.rm = TRUE), .groups = "drop")
  a_vals <- bin_means |> filter(treatment == "uncertainty_vis") |> pull(mean_mIoU)
  b_vals <- bin_means |> filter(treatment == "baseline")        |> pull(mean_mIoU)
  one_tailed_t(
    a_vals, b_vals,
    alternative = "greater",
    label       = sprintf("Bin = %-4s  (5 images per participant)", b)
  )
}

subsection("4c. Annotation efficiency — one-tailed t-tests per bin")
cat("  Unit: participant-level mean editing time within each bin\n\n")

for (b in bin_levels) {
  bin_means <- trials |>
    filter(bin == b) |>
    group_by(participant_id, treatment) |>
    summarise(mean_time = mean(editing_time, na.rm = TRUE), .groups = "drop")
  a_vals <- bin_means |> filter(treatment == "uncertainty_vis") |> pull(mean_time)
  b_vals <- bin_means |> filter(treatment == "baseline")        |> pull(mean_time)
  one_tailed_t(
    a_vals, b_vals,
    alternative = "less",
    label       = sprintf("Bin = %-4s", b)
  )
}

subsection("4d. Secondary analysis: mid + high difficulty images only")
cat("  Rationale: uncertainty cues are theoretically informative only on non-trivial images.\n\n")

trials_mh <- trials |> filter(bin %in% c("mid", "high"))

cat("  --- t-tests (one-tailed) ---\n\n")

# Quality — mid + high
one_tailed_t(
  trials_mh |>
    group_by(participant_id, treatment) |>
    summarise(m = mean(mIoU_user), .groups = "drop") |>
    filter(treatment == "uncertainty_vis") |>
    pull(m),
  trials_mh |>
    group_by(participant_id, treatment) |>
    summarise(m = mean(mIoU_user), .groups = "drop") |>
    filter(treatment == "baseline") |>
    pull(m),
  alternative = "greater",
  label       = "Quality  (mid + high)"
)

# Efficiency — mid + high
one_tailed_t(
  trials_mh |>
    group_by(participant_id, treatment) |>
    summarise(m = mean(editing_time), .groups = "drop") |>
    filter(treatment == "uncertainty_vis") |>
    pull(m),
  trials_mh |>
    group_by(participant_id, treatment) |>
    summarise(m = mean(editing_time), .groups = "drop") |>
    filter(treatment == "baseline") |>
    pull(m),
  alternative = "less",
  label       = "Efficiency  (mid + high)"
)

# LME models on the mid+high subset (p-values two-tailed).
cat("  --- LME models on mid+high subset (p-values two-tailed) ---\n\n")

lme_mh_quality <- lmer(
  mIoU_user ~ treatment_dummy + (1 | participant_id),
  data = trials_mh, REML = FALSE
)
cat("  LME: quality (mid + high)\n")
print(summary(lme_mh_quality))
print_lme_treatment(
  lme_mh_quality,
  "Treatment effect on quality, mid+high (two-tailed)",
  "two-tailed"
)

lme_mh_time <- lmer(
  editing_time ~ treatment_dummy + (1 | participant_id),
  data = trials_mh, REML = FALSE
)
cat("  LME: efficiency (mid + high)\n")
print(summary(lme_mh_time))
print_lme_treatment(
  lme_mh_time,
  "Treatment effect on efficiency, mid+high (two-tailed)",
  "two-tailed"
)


# =============================================================================
# 5. Cognitive load moderation
# =============================================================================
# Manuscript reference: Results > "Cognitive Load and Treatment Effects"
# (paragraphs covering the LME moderation models). Reported values:
#   Cognitive load descriptives: UV M = 3.02 SD = 1.33; BL M = 3.23 SD = 1.12
#   Quality interaction:    beta = -0.32, SE = 0.26, Wald chi2(1) = 1.48, p = 0.224
#   Efficiency interaction: beta =  0.58, SE = 0.93, Wald chi2(1) = 0.39, p = 0.533
#
# Tests whether self-reported cognitive load moderates the effect of the
# uncertainty visualisation on both outcomes. The interaction term
# treatment_dummy:cognitive_load is of primary interest.
#
# Models are compared against an intercept-only null via likelihood-ratio test
# to assess overall explanatory value. Interaction significance is further
# evaluated with Wald chi-square tests (Section 5d) as a cross-check.
# =============================================================================

section(
  "5. COGNITIVE LOAD MODERATION",
  "Formula: outcome ~ treatment_dummy * cognitive_load + (1 | participant_id)"
)

subsection("5a. Cognitive load descriptives by condition")
cat(sprintf("  Uncertainty Vis. : M = %.2f, SD = %.2f\n",
            mean(treat_p$cognitive_load), sd(treat_p$cognitive_load)))
cat(sprintf("  Baseline         : M = %.2f, SD = %.2f\n",
            mean(base_p$cognitive_load), sd(base_p$cognitive_load)))
cog_test <- t.test(treat_p$cognitive_load, base_p$cognitive_load)
cat(sprintf("  Two-tailed t-test: t = %.3f,  p = %.3f\n",
            cog_test$statistic, cog_test$p.value))

subsection("5b. LME: mIoU_user ~ treatment × cognitive_load")
lme_cog_quality <- lmer(
  mIoU_user ~ treatment_dummy * cognitive_load + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_cog_quality))

lme_cog_quality_null <- lmer(
  mIoU_user ~ 1 + (1 | participant_id),
  data = trials, REML = FALSE
)
cat("  Likelihood-ratio test vs. intercept-only null:\n")
print(anova(lme_cog_quality_null, lme_cog_quality))

subsection("5c. LME: editing_time ~ treatment × cognitive_load")
lme_cog_time <- lmer(
  editing_time ~ treatment_dummy * cognitive_load + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_cog_time))

lme_cog_time_null <- lmer(
  editing_time ~ 1 + (1 | participant_id),
  data = trials, REML = FALSE
)
cat("  Likelihood-ratio test vs. intercept-only null:\n")
print(anova(lme_cog_time_null, lme_cog_time))

subsection("5d. Wald chi-square tests for the interaction term")
for (mod_label in c("Quality", "Editing Time")) {
  mod      <- if (mod_label == "Quality") lme_cog_quality else lme_cog_time
  coef_tbl <- as.data.frame(coef(summary(mod)))
  beta     <- coef_tbl["treatment_dummy:cognitive_load", "Estimate"]
  se       <- coef_tbl["treatment_dummy:cognitive_load", "Std. Error"]
  wald     <- (beta / se)^2
  wald_p   <- pchisq(wald, df = 1, lower.tail = FALSE)
  cat(sprintf(
    "  %-14s  beta = %+.3f,  SE = %.3f,  Wald chi2(1) = %.2f,  p = %.3f\n",
    mod_label, beta, se, wald, wald_p
  ))
}


# =============================================================================
# 6. Task satisfaction — Mann-Whitney U
# =============================================================================
# Manuscript reference: Results > "Cognitive Load and Treatment Effects"
# (final paragraph of that subsection). Reported values:
#   UV M = 4.05 SD = 1.25; BL M = 3.88 SD = 1.13
#   Mann-Whitney U = 1985, p = 0.331, r_b = -0.10
#
# Task satisfaction was measured on an ordinal Likert scale; the
# Mann-Whitney U test is therefore preferred over a t-test. Effect size is
# reported as the rank-biserial correlation (r_b), which ranges from -1 to 1.
# The non-significant result rules out the possibility that performance gains
# are driven by general positive affect toward the interface.
# =============================================================================

section(
  "6. TASK SATISFACTION: Mann-Whitney U",
  "Test: two-tailed | Effect size: rank-biserial correlation (r_b)"
)

cat(sprintf("  Uncertainty Vis. : M = %.2f, SD = %.2f\n",
            mean(treat_p$task_satisfaction), sd(treat_p$task_satisfaction)))
cat(sprintf("  Baseline         : M = %.2f, SD = %.2f\n",
            mean(base_p$task_satisfaction), sd(base_p$task_satisfaction)))

mw  <- wilcox.test(
  treat_p$task_satisfaction, base_p$task_satisfaction,
  alternative = "two.sided", exact = FALSE
)
r_b <- 1 - (2 * mw$statistic) / (nrow(treat_p) * nrow(base_p))
cat(sprintf("  Mann-Whitney U = %.0f,  p = %.3f,  r_b = %.3f\n",
            mw$statistic, mw$p.value, r_b))


# =============================================================================
# 7. Robustness checks
# =============================================================================
# Manuscript reference: Methods > Robustness Checks; Supplementary Tables S1
# (full sample) and S2 (mid+high subset). The three checks correspond
# directly to checks (a), (b), and (c) as labelled in the manuscript text
# and table notes.
#
# Three checks assess the sensitivity of the quality effect to modelling
# choices. (a) Winsorization addresses potential outlier inflation. (b)
# Adding self-efficacy and task familiarity as covariates tests whether
# pre-existing individual differences explain the treatment effect. (c)
# Controlling for initial model mIoU addresses the possibility that the
# uncertainty_vis group started from a lower baseline and had more room
# to improve; attenuation is expected in the full sample (p = 0.136, Table S1)
# but the effect should survive in the mid+high subset (p = 0.040, Table S2)
# where initial performance is lower and uncertainty cues are theoretically
# informative.
# =============================================================================

section(
  "7. ROBUSTNESS CHECKS",
  "Checks: (a) winsorization  (b) participant covariates  (c) initial model mIoU"
)

subsection("7a. Winsorized mIoU (5th–95th percentile)")
wins_limits <- quantile(trials$mIoU_user, probs = c(0.05, 0.95), na.rm = TRUE)
cat(sprintf("  Winsorization bounds: [%.2f, %.2f]\n\n", wins_limits[1], wins_limits[2]))

trials <- trials |>
  mutate(
    mIoU_user_wins = pmin(pmax(mIoU_user, wins_limits[1]), wins_limits[2])
  )

lme_wins <- lmer(
  mIoU_user_wins ~ treatment_dummy + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_wins))
print_lme_treatment(lme_wins, "Winsorized quality (two-tailed)", "two-tailed")

subsection("7b. Self-efficacy and task familiarity as covariates")
lme_covariates <- lmer(
  mIoU_user ~ treatment_dummy + self_efficacy + task_familiarity +
    (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_covariates))
print_lme_treatment(lme_covariates, "Quality + covariates (two-tailed)", "two-tailed")

subsection("7c. Controlling for initial model mIoU")
# initial_mIoU = mIoU_user - delta_mIoU recovers the model's prediction
# before any user editing. Including it isolates improvement attributable
# to the intervention rather than image-level variation in starting quality.
cat("  initial_mIoU = mIoU_user - delta_mIoU\n\n")

trials <- trials |> mutate(initial_mIoU = mIoU_user - delta_mIoU)
cat(sprintf("  initial_mIoU: M = %.2f,  SD = %.2f,  range = [%.2f, %.2f]\n\n",
            mean(trials$initial_mIoU), sd(trials$initial_mIoU),
            min(trials$initial_mIoU),  max(trials$initial_mIoU)))

cat("  Full sample:\n")
lme_initial_full <- lmer(
  mIoU_user ~ treatment_dummy + initial_mIoU + (1 | participant_id),
  data = trials, REML = FALSE
)
print(summary(lme_initial_full))
print_lme_treatment(
  lme_initial_full,
  "Quality + initial mIoU, full sample (two-tailed)",
  "two-tailed"
)

cat("  Mid + high subset:\n")
trials_mh <- trials |>
  filter(bin %in% c("mid", "high")) |>
  mutate(initial_mIoU = mIoU_user - delta_mIoU)

lme_initial_mh <- lmer(
  mIoU_user ~ treatment_dummy + initial_mIoU + (1 | participant_id),
  data = trials_mh, REML = FALSE
)
print(summary(lme_initial_mh))
print_lme_treatment(
  lme_initial_mh,
  "Quality + initial mIoU, mid+high subset (two-tailed)",
  "two-tailed"
)


# =============================================================================
# 8. Exploratory: predicted treatment effect across cognitive load levels
# =============================================================================
# Manuscript reference: Results > "Cognitive Load and Treatment Effects"
# (paragraph beginning "As an exploratory analysis"). Reported values:
#   Low CL (mean - 1SD): predicted advantage ~1.05 mIoU pp
#   High CL (mean + 1SD): predicted advantage ~0.28 mIoU pp
#
# Uses fixed-effect predictions from the Section 5b LME to characterise how
# the estimated mIoU advantage for the uncertainty_vis condition varies across
# the cognitive load range. Predictions are evaluated at ±1 SD from the mean.
# Because the interaction was not significant (Section 5d), these values are
# descriptive only and should not be interpreted as confirmatory evidence.
# =============================================================================

section(
  "8. EXPLORATORY: Predicted Treatment Effect at ±1 SD Cognitive Load",
  "Source: fixed-effect predictions from Section 5b LME | Descriptive only"
)

cog_mean <- mean(trials$cognitive_load, na.rm = TRUE)
cog_sd   <- sd(trials$cognitive_load,   na.rm = TRUE)
cat(sprintf("  Cognitive load: M = %.2f,  SD = %.2f\n",   cog_mean, cog_sd))
cat(sprintf("  Low  anchor  : %.2f  (mean - 1SD)\n",      cog_mean - cog_sd))
cat(sprintf("  High anchor  : %.2f  (mean + 1SD)\n\n",    cog_mean + cog_sd))

fe <- fixef(lme_cog_quality)

# Returns the model-predicted mIoU for a given treatment and cognitive load.
predict_miou <- function(trt, cl) {
  fe["(Intercept)"]                    +
  fe["treatment_dummy"]                * trt      +
  fe["cognitive_load"]                 * cl       +
  fe["treatment_dummy:cognitive_load"] * trt * cl
}

cat("  Predicted mIoU advantage (uncertainty_vis - baseline):\n\n")
for (label in c("Low (mean - 1SD)", "High (mean + 1SD)")) {
  cl  <- if (grepl("Low", label)) cog_mean - cog_sd else cog_mean + cog_sd
  adv <- predict_miou(1, cl) - predict_miou(0, cl)
  cat(sprintf("    %-22s  CL = %.2f  ->  advantage = %.3f mIoU pp\n",
              label, cl, adv))
}
cat("\n  Note: interaction not significant (Section 5d); descriptive only.\n")


# =============================================================================
# 9. Manuscript figures
# =============================================================================
# Produces Figures 1–3 as referenced in the main text. Output files are
# written to Figures/ as PDF at 300 dpi (11 × 5 or 11 × 5.5 inches).
#   Figure 1 (Overall_Performance.pdf)  → main text Figure 2
#   Figure 2 (Guidance_Effect.pdf)      → main text Figure 4
#   Figure 3 (Guidance_Effect_Boxes.pdf)→ main text Figure 3
# =============================================================================

section("9. FIGURES", "Generating manuscript figures")

dir.create(here("analysis", "01_figures"), showWarnings = FALSE, recursive = TRUE)

# Shared colour palette: baseline (grey) and uncertainty vis (red).
palette_vals <- c("Baseline" = "#95a5a6", "Uncertainty Visualization" = "#e74c3c")

# Minimal theme consistent with manuscript figure standards.
theme_manuscript <- function() {
  theme_classic(base_size = 13) +
    theme(
      axis.title       = element_text(size = 12),
      axis.text        = element_text(size = 11),
      plot.title       = element_text(face = "bold", size = 13, hjust = 0.5),
      legend.title     = element_text(size = 11),
      legend.text      = element_text(size = 11),
      panel.grid       = element_blank(),
      strip.background = element_blank()
    )
}


# --- Figure 1: Overall Performance ------------------------------------------
# Panel A: mean mIoU by condition, with KITTI dataset and detector baselines
#          shown in grey for context.
# Panel B: mean editing time per image by condition.
# Error bars are ±1 standard error of the mean.

exp_summary <- participants |>
  mutate(Condition = recode(treatment,
    "uncertainty_vis" = "Uncertainty\nVisualization",
    "baseline"        = "Baseline"
  )) |>
  group_by(Condition) |>
  summarise(
    mean       = mean(mean_mIoU_user),
    se         = sd(mean_mIoU_user) / sqrt(n()),
    fill_col   = ifelse(unique(Condition) == "Uncertainty\nVisualization", "#e74c3c", "#95a5a6"),
    alpha_val  = 1.0,
    has_errbar = TRUE,
    .groups    = "drop"
  )

# Reference performance levels from the KITTI dataset labels and detector
# predictions (reported in the manuscript; reproduced here for the figure).
ref_summary <- tibble(
  Condition  = c("KITTI Dataset\nLabels", "Detector\nPredictions"),
  mean       = c(84.67, 86.59),
  se         = c(4.74,  4.88) / sqrt(95),
  fill_col   = "#95a5a6",
  alpha_val  = 0.45,
  has_errbar = FALSE
)

summary_df <- bind_rows(ref_summary, exp_summary) |>
  mutate(label = factor(Condition, levels = c(
    "KITTI Dataset\nLabels", "Detector\nPredictions", "Baseline", "Uncertainty\nVisualization"
  )))

bracket_y <- max(summary_df$mean[3:4] + summary_df$se[3:4]) + 0.15
cap_h     <- 0.12

p1a <- ggplot(summary_df, aes(x = label, y = mean)) +
  geom_col(aes(fill = label, alpha = label), width = 0.55, colour = NA) +
  geom_errorbar(
    data  = filter(summary_df, has_errbar),
    aes(ymin = mean - se, ymax = mean + se),
    width = 0.15, linewidth = 0.9
  ) +
  # Significance bracket between Baseline and Uncertainty Vis.
  annotate("segment", x = 3, xend = 3,
           y = bracket_y, yend = bracket_y + cap_h, linewidth = 0.7) +
  annotate("segment", x = 4, xend = 4,
           y = bracket_y, yend = bracket_y + cap_h, linewidth = 0.7) +
  annotate("segment", x = 3, xend = 4,
           y = bracket_y + cap_h, yend = bracket_y + cap_h, linewidth = 0.7) +
  annotate("text", x = 3.5, y = bracket_y + cap_h + 0.2,
           label = "p = 0.015 *", hjust = 0.5, vjust = 0, size = 3.5) +
  scale_fill_manual(
    values = setNames(summary_df$fill_col, levels(summary_df$label)),
    guide  = "none"
  ) +
  scale_alpha_manual(
    values = setNames(summary_df$alpha_val, levels(summary_df$label)),
    guide  = "none"
  ) +
  scale_y_continuous(
    limits = c(80, bracket_y + cap_h + 1.0),
    breaks = seq(80, 95, by = 5),
    expand = expansion(mult = c(0, 0)),
    oob    = scales::rescale_none
  ) +
  labs(
    title = "Annotation Quality",
    x     = NULL,
    y     = "Mean IoU (%) vs. Relabeled Ground Truth"
  ) +
  theme_manuscript() +
  theme(panel.grid.major.y = element_line(colour = "#e0e0e0", linewidth = 0.4))

fig1b_data <- participants |>
  mutate(Condition = factor(
    recode(treatment,
      "uncertainty_vis" = "Uncertainty\nVisualization",
      "baseline"        = "Baseline"
    ),
    levels = c("Baseline", "Uncertainty\nVisualization")
  ))

fig1b_stats <- fig1b_data |>
  group_by(Condition) |>
  summarise(m = mean(mean_editing_time), se = sd(mean_editing_time) / sqrt(n()),
            .groups = "drop")
bracket_y_b <- max(fig1b_stats$m + fig1b_stats$se) + 0.5
cap_h_b     <- 0.5

p1b <- ggplot(fig1b_data,
              aes(x = Condition, y = mean_editing_time, fill = Condition)) +
  stat_summary(fun = mean, geom = "bar", width = 0.6) +
  stat_summary(fun.data = mean_se, geom = "errorbar",
               width = 0.15, linewidth = 0.7) +
  annotate("segment", x = 1, xend = 1,
           y = bracket_y_b, yend = bracket_y_b + cap_h_b, linewidth = 0.7) +
  annotate("segment", x = 2, xend = 2,
           y = bracket_y_b, yend = bracket_y_b + cap_h_b, linewidth = 0.7) +
  annotate("segment", x = 1, xend = 2,
           y = bracket_y_b + cap_h_b, yend = bracket_y_b + cap_h_b,
           linewidth = 0.7) +
  annotate("text", x = 1.5, y = bracket_y_b + cap_h_b + 0.5,
           label = "p = 0.045 *", hjust = 0.5, vjust = 0, size = 3.5) +
  scale_fill_manual(values = palette_vals, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Annotation Time", x = NULL, y = "Time per Image (s)") +
  theme_manuscript() +
  theme(panel.grid.major.y = element_line(colour = "#e0e0e0", linewidth = 0.4))

fig1 <- plot_grid(
  p1a, p1b,
  ncol = 2, labels = c("A", "B"), label_size = 13,
  align = "h", axis = "bt", rel_widths = c(1.4, 1)
)

ggsave(here("analysis", "01_figures", "Overall_Performance.pdf"),
       plot = fig1, width = 11, height = 5, dpi = 300)
cat("  Figure 1 saved.\n")


# --- Figure 2: Guidance Effect by Difficulty --------------------------------
# Panel A: mean delta-mIoU (user improvement) by difficulty bin and condition.
# Panel B: mean editing time by difficulty bin and condition.
# Points show condition means ± 1 SE; lines connect bins within each condition.
# p-values are from the per-bin one-tailed t-tests in Section 4.

fig2_data <- trials |>
  mutate(Condition = factor(
    ifelse(treatment == "uncertainty_vis", "Uncertainty Visualization", "Baseline"),
    levels = c("Baseline", "Uncertainty Visualization")
  )) |>
  group_by(participant_id, Condition, bin) |>
  summarise(
    mean_delta = mean(delta_mIoU,   na.rm = TRUE),
    mean_time  = mean(editing_time, na.rm = TRUE),
    .groups = "drop"
  )

p2a <- ggplot(fig2_data,
              aes(x = bin, y = mean_delta, colour = Condition,
                  group = Condition, shape = Condition, linetype = Condition)) +
  stat_summary(fun = mean, geom = "point", size = 3,
               position = position_dodge(0.15)) +
  stat_summary(fun = mean, geom = "line",
               position = position_dodge(0.15), linewidth = 0.8) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.08,
               linewidth = 0.6, position = position_dodge(0.15)) +
  annotate("text", x = 2, y = 2.3,  label = "p = 0.002**",
           hjust = 0.5, size = 3.5, fontface = "bold",   colour = "black") +
  annotate("text", x = 3, y = 3.15, label = "p = 0.055",
           hjust = 0.5, size = 3.2, fontface = "italic", colour = "grey40") +
  scale_colour_manual(values = palette_vals) +
  scale_shape_manual(values = c("Baseline" = 16, "Uncertainty Visualization" = 15)) +
  scale_linetype_manual(
    values = c("Baseline" = "solid", "Uncertainty Visualization" = "dashed")
  ) +
  scale_x_discrete(
    labels = c("low" = "Easy", "mid" = "Medium", "high" = "Difficult")
  ) +
  labs(
    title = "Quality Improvement",
    x     = "Image Difficulty",
    y     = expression("User Improvement (" * Delta * " mIoU %-pts)")
  ) +
  theme_manuscript() +
  theme(legend.position = "none")

p2b <- ggplot(fig2_data,
              aes(x = bin, y = mean_time, colour = Condition,
                  group = Condition, shape = Condition, linetype = Condition)) +
  stat_summary(fun = mean, geom = "point", size = 3,
               position = position_dodge(0.15)) +
  stat_summary(fun = mean, geom = "line",
               position = position_dodge(0.15), linewidth = 0.8) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.08,
               linewidth = 0.6, position = position_dodge(0.15)) +
  annotate("text", x = 2, y = 28, label = "p = 0.064",
           hjust = 0.5, size = 3.2, fontface = "italic", colour = "grey40") +
  annotate("text", x = 3, y = 29, label = "p = 0.018*",
           hjust = 0.5, size = 3.5, fontface = "bold",   colour = "black") +
  scale_colour_manual(values = palette_vals) +
  scale_shape_manual(values = c("Baseline" = 16, "Uncertainty Visualization" = 15)) +
  scale_linetype_manual(
    values = c("Baseline" = "solid", "Uncertainty Visualization" = "dashed")
  ) +
  scale_x_discrete(
    labels = c("low" = "Easy", "mid" = "Medium", "high" = "Difficult")
  ) +
  labs(
    title    = "Annotation Efficiency",
    x        = "Image Difficulty",
    y        = "Time per Image (s)",
    colour   = "Condition",
    shape    = "Condition",
    linetype = "Condition"
  ) +
  theme_manuscript() +
  theme(
    legend.position   = c(0.25, 0.88),
    legend.background = element_rect(fill = "transparent", colour = NA)
  )

fig2 <- plot_grid(
  p2a, p2b,
  ncol = 2, labels = c("A", "B"), label_size = 13
)

ggsave(here("analysis", "01_figures", "Guidance_Effect.pdf"),
       plot = fig2, width = 11, height = 5.5, dpi = 300)
cat("  Figure 2 saved.\n")


# --- Figure 3: GEE — Probability of Bounding Box Change ---------------------
# Analysis reference: Section 4a. `gee_mod` and `predictions` are fitted
# there and reused here. The figure plots the marginal probability of box
# modification against log-transformed uncertainty, with 95% Wald CIs.

fig_gee <- ggplot(predictions,
                  aes(x = log_albox, y = prob,
                      colour = Condition, fill = Condition)) +
  geom_ribbon(aes(ymin = asymp.LCL, ymax = asymp.UCL),
              alpha = 0.15, colour = NA) +
  geom_line(aes(linetype = Condition), linewidth = 1.2) +
  scale_colour_manual(values = palette_vals) +
  scale_fill_manual(values = palette_vals) +
  scale_linetype_manual(
    values = c("Baseline" = "solid", "Uncertainty Visualization" = "dashed")
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    limits = c(0.25, 1), expand = c(0, 0)
  ) +
  labs(
    x = "Log-Transformed Uncertainty",
    y = "Predicted Probability of Change"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position        = "inside",
    legend.position.inside = c(0.2, 0.85),
    legend.background      = element_rect(fill = "transparent"),
    panel.grid.major.y     = element_line(color = "grey95"),
    axis.line              = element_line(linewidth = 0.5),
    axis.ticks             = element_line(linewidth = 0.5)
  )

ggsave(here("analysis", "01_figures", "Guidance_Effect_Boxes.pdf"),
       plot = fig_gee, width = 11, height = 5.5, dpi = 300)
cat("  Figure 3 saved.\n")


cat("\n", strrep("=", 70), "\n")
cat("  Analysis complete.\n")
cat(strrep("=", 70), "\n")