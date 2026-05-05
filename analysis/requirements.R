# =============================================================================
# Package Requirements
# Localization Uncertainty Visualization Study — Supplementary Analysis
#
# Run this script once before sourcing statistical_analysis_supplementary.R.
# All packages are installed from CRAN. Tested under R 4.4.1.
#
# Minimum R version: 4.1.0  (native pipe |> operator required)
# =============================================================================

required_packages <- c(
  "tidyverse",    # Data wrangling and ggplot2 (includes readr, dplyr, forcats, etc.)
  "lme4",         # Linear mixed-effects models
  "lmerTest",     # Satterthwaite p-values for lme4 models
  "coin",         # Permutation-based inference (loaded but not directly called)
  "cowplot",      # Multi-panel figure composition (plot_grid)
  "geepack",      # Generalised estimating equations (geeglm)
  "emmeans",      # Marginal means and predicted probabilities from GEE
  "here"          # Relative file paths anchored to the project root
)

# Install any packages not already present.
missing <- setdiff(required_packages, rownames(installed.packages()))

if (length(missing) == 0) {
  message("All required packages are already installed.")
} else {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing)
}

# Verify all packages load without error.
invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' failed to load after installation.", pkg))
  }
}))

message("All packages verified. You can now source statistical_analysis_supplementary.R")

# =============================================================================
# Session info at time of original analysis (R 4.4.1)
# =============================================================================
#
# tidyverse   2.0.0   (ggplot2 3.5.2, dplyr 1.1.4, readr 2.1.5,
#                      forcats 1.0.0, stringr 1.5.1, tibble 3.2.1,
#                      lubridate 1.9.3, tidyr 1.3.1, purrr 1.0.2)
# lme4        1.1-35
# lmerTest    3.1-3
# coin        1.4-3
# cowplot     1.1.3
# geepack     1.3-5
# emmeans     1.10.1
# here        1.0.1