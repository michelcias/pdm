################################################################################
# MCMC Analysis of Changepoint Data using Normal Mixture Models
################################################################################
# This script applies different state-space models (local level, local trend,
# and local acceleration) to the GBM29 dataset from the changepoint package.
# Each model is tested with both logit and probit link functions, and under two
# prior families for every precision: the conjugate Gamma prior on the precision
# (Sections 1-3) and the weakly-informative Half-Cauchy prior on the standard
# deviation (Section 4; Gelman, 2006).
#
# Author: michelcias
# Date: 2026-01-08
################################################################################
# Load necessary libraries
devtools::load_all()

# Load required data
y <- changepoint::Lai2005fig4$GBM29

################################################################################
# 1. LOCAL LEVEL MODEL
################################################################################
# The local level model assumes the underlying state follows a random walk
# without trend or acceleration components.
################################################################################

## 1.1 Local Level with Logit Link
cat("\n=== Running Local Level Model with Logit Link ===\n")
out_logit_level <- mcmc_normal_mixture_locallevel(
  y,
  link                    = "logit",
  burnin                  = 50000,        # Burn-in iterations to discard
  thinning                = 250,          # Keep every 250th sample
  n_draws                 = 1000,         # Number of samples to keep
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile of y
  prior_mu01_prec         = 1 / (1 * var(y)),
  # Prior for precision of first mixture component
  prior_prec01_shape      = 0.1,
  prior_prec01_rate       = 0.1,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile of y
  prior_mu02_prec         = 1 / (1 * var(y)),
  # Prior for precision of second mixture component
  prior_prec02_shape      = 0.1,
  prior_prec02_rate       = 0.1,
  # Prior for state equation parameters
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Prior for state precision
  prior_prec1_shape       = 0.1,
  prior_prec1_rate        = 0.1,
  # Adaptive MALA parameters
  lag_update              = 50,           # Update adaptation every 50 iterations
  max_step_size           = 0.1,          # Maximum step size for MALA
  base_adaptation_rate    = 1,
  decay_exponent          = 0.5,
  target_acceptance       = 0.44,         # Target acceptance rate
  min_deviation_threshold = NULL,         # Default: 1/lag_update
  # Output options
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 456
); plot(out_logit_level, ask = FALSE)

## 1.2 Local Level with Probit Link
cat("\n=== Running Local Level Model with Probit Link ===\n")
out_probit_level <- mcmc_normal_mixture_locallevel(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 250,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_shape = 0.1,
  prior_prec01_rate  = 0.1,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (10 * var(y)),
  prior_prec02_shape = 0.1,
  prior_prec02_rate  = 0.1,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_prec1_shape  = 0.1,
  prior_prec1_rate   = 0.1,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 789
);plot(out_probit_level, ask = FALSE)

################################################################################
# 2. LOCAL TREND MODEL
################################################################################
# The local trend model includes both level and slope components, allowing
# the underlying trend to change over time.
################################################################################

## 2.1 Local Trend with Logit Link
cat("\n=== Running Local Trend Model with Logit Link ===\n")
out_logit_trend <- mcmc_normal_mixture_localtrend(
  y,
  link                    = "logit",
  burnin                  = 20000,       # Increased burn-in for more complex model
  thinning                = 600,
  n_draws                 = 1000,
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile
  prior_mu01_prec         = 1 / (1 * var(y)),
  prior_prec01_shape      = 0.01,
  prior_prec01_rate       = 0.01,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile
  prior_mu02_prec         = 1 / (1 * var(y)),
  prior_prec02_shape      = 0.01,
  prior_prec02_rate       = 0.01,
  # Prior for state equation - level
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Prior for state equation - slope
  prior_theta02_mean      = 0,
  prior_theta02_prec      = 1,
  # Prior for state precision - level
  prior_prec1_shape       = 0.1,
  prior_prec1_rate        = 0.1,
  # Prior for state precision - slope
  prior_prec2_shape       = 0.1,
  prior_prec2_rate        = 0.1,
  # Adaptive MALA parameters
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 1,
  decay_exponent          = 0.6,
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,         # Default: 1/lag_update
  # Output options
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 456
);plot(out_logit_trend, ask = FALSE);mcmc_convergence(out_logit_trend)

## 2.2 Local Trend with Probit Link
cat("\n=== Running Local Trend Model with Probit Link ===\n")
out_probit_trend <- mcmc_normal_mixture_localtrend(
  y,
  link               = "probit",
  burnin             = 20000,
  thinning           = 600,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_shape = 0.01,
  prior_prec01_rate  = 0.01,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (1 * var(y)),
  prior_prec02_shape = 0.01,
  prior_prec02_rate  = 0.01,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_prec1_shape  = 0.1,
  prior_prec1_rate   = 0.1,
  prior_prec2_shape  = 0.1,
  prior_prec2_rate   = 0.1,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 789
);plot(out_probit_trend, ask = FALSE);mcmc_convergence(out_probit_trend)

################################################################################
# 3. LOCAL ACCELERATION MODEL
################################################################################
# The local acceleration model includes level, slope, and acceleration components,
# providing the most flexible framework for capturing complex dynamics.
################################################################################

## 3.1 Local Acceleration with Logit Link
cat("\n=== Running Local Acceleration Model with Logit Link ===\n")
out_logit_accel <- mcmc_normal_mixture_localacceleration(
  y,
  link                    = "logit",
  burnin                  = 50000,       # Increased burn-in for most complex model
  thinning                = 1000,
  n_draws                 = 1000,
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile
  prior_mu01_prec         = 1 / (1 * var(y)),
  prior_prec01_shape      = 01,
  prior_prec01_rate       = 01,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile
  prior_mu02_prec         = 1 / (1 * var(y)),
  prior_prec02_shape      = 01,
  prior_prec02_rate       = 01,
  # Prior for state equation - level
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Prior for state equation - slope
  prior_theta02_mean      = 0,
  prior_theta02_prec      = 1,
  # Prior for state equation - acceleration
  prior_theta03_mean      = 0,
  prior_theta03_prec      = 1,
  # Prior for state precision - level
  prior_prec1_shape       = 01,
  prior_prec1_rate        = 01,
  # Prior for state precision - slope
  prior_prec2_shape       = 01,
  prior_prec2_rate        = 01,
  # Prior for state precision - acceleration
  prior_prec3_shape       = 01,
  prior_prec3_rate        = 01,
  # Adaptive MALA parameters
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 10,
  decay_exponent          = 0.5,          # Slightly lower for better adaptation
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,         # Default: 1/lag_update
  # Output options
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 456
);plot(out_logit_accel, ask = FALSE)

## 3.2 Local Acceleration with Probit Link
cat("\n=== Running Local Acceleration Model with Probit Link ===\n")
out_probit_accel <- mcmc_normal_mixture_localacceleration(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 500,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_shape = 01,
  prior_prec01_rate  = 01,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (1 * var(y)),
  prior_prec02_shape = 01,
  prior_prec02_rate  = 01,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_theta03_mean = 0,
  prior_theta03_prec = 1,
  prior_prec1_shape  = 01,
  prior_prec1_rate   = 01,
  prior_prec2_shape  = 01,
  prior_prec2_rate   = 01,
  prior_prec3_shape  = 01,
  prior_prec3_rate   = 01,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 123
);plot(out_probit_accel, ask = FALSE)

################################################################################
# 4. HALF-CAUCHY PRIOR VARIANTS
################################################################################
# The runs below refit the same three models under the weakly-informative
# Half-Cauchy prior (Gelman, 2006) instead of the Gamma prior on the precisions.
# The Half-Cauchy is placed on the *standard deviation*: for a component
# precision phi_k it acts on the component SD sqrt(1/phi_k), and for a state
# innovation precision 1/W_k on the innovation SD sqrt(W_k). Each precision is
# switched by passing `*_type = "halfcauchy"` together with `*_scale` (the scale
# A > 0); the Gamma `*_shape`/`*_rate` arguments are then unused. "halfcauchy" is
# shorthand for `*_type = "halft"` with `*_df = 1`. All other settings mirror the
# corresponding Gamma runs above.
#
# Scale choices (weakly informative): the component SDs live on the data scale,
# so we use `sd(y)`; the weight-innovation SDs live on the logit/probit latent
# scale, so we use 1.
sd_y <- sd(y)

## 4.1 Local Level with Logit Link (Half-Cauchy)
cat("\n=== Running Local Level Model with Logit Link (Half-Cauchy) ===\n")
out_logit_level_hc <- mcmc_normal_mixture_locallevel(
  y,
  link                    = "logit",
  burnin                  = 50000,
  thinning                = 250,
  n_draws                 = 1000,
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile of y
  prior_mu01_prec         = 1 / (1 * var(y)),
  # Half-Cauchy prior on the SD of the first mixture component
  prior_prec01_type       = "halfcauchy",
  prior_prec01_scale      = sd_y,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile of y
  prior_mu02_prec         = 1 / (1 * var(y)),
  # Half-Cauchy prior on the SD of the second mixture component
  prior_prec02_type       = "halfcauchy",
  prior_prec02_scale      = sd_y,
  # Prior for state equation parameters
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Half-Cauchy prior on the state innovation SD
  prior_prec1_type        = "halfcauchy",
  prior_prec1_scale       = 1,
  # Adaptive MALA parameters
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 1,
  decay_exponent          = 0.5,
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,         # Default: 1/lag_update
  # Output options
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 1456
); plot(out_logit_level_hc, ask = FALSE)

## 4.2 Local Level with Probit Link (Half-Cauchy)
cat("\n=== Running Local Level Model with Probit Link (Half-Cauchy) ===\n")
out_probit_level_hc <- mcmc_normal_mixture_locallevel(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 300,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_type  = "halfcauchy",
  prior_prec01_scale = 10*sd_y,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (1 * var(y)),
  prior_prec02_type  = "halfcauchy",
  prior_prec02_scale = 10*sd_y,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_prec1_type   = "halfcauchy",
  prior_prec1_scale  = 10,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 1789
); plot(out_probit_level_hc, ask = FALSE)

## 4.3 Local Trend with Logit Link (Half-Cauchy)
cat("\n=== Running Local Trend Model with Logit Link (Half-Cauchy) ===\n")
out_logit_trend_hc <- mcmc_normal_mixture_localtrend(
  y,
  link                    = "logit",
  burnin                  = 50000,
  thinning                = 200,
  n_draws                 = 1000,
  prior_mu01_mean         = NULL,
  prior_mu01_prec         = 1 / (1 * var(y)),
  prior_prec01_type       = "halfcauchy",
  prior_prec01_scale      = sd_y,
  prior_mu02_mean         = NULL,
  prior_mu02_prec         = 1 / (1 * var(y)),
  prior_prec02_type       = "halfcauchy",
  prior_prec02_scale      = sd_y,
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  prior_theta02_mean      = 0,
  prior_theta02_prec      = 1,
  # Half-Cauchy prior on both state innovation SDs (level, slope)
  prior_prec1_type        = "halfcauchy",
  prior_prec1_scale       = 1,
  prior_prec2_type        = "halfcauchy",
  prior_prec2_scale       = 1,
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 1,
  decay_exponent          = 0.6,
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 1457
); plot(out_logit_trend_hc, ask = FALSE)

## 4.4 Local Trend with Probit Link (Half-Cauchy)
cat("\n=== Running Local Trend Model with Probit Link (Half-Cauchy) ===\n")
out_probit_trend_hc <- mcmc_normal_mixture_localtrend(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 200,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_type  = "halfcauchy",
  prior_prec01_scale = sd_y,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (1 * var(y)),
  prior_prec02_type  = "halfcauchy",
  prior_prec02_scale = sd_y,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_prec1_type   = "halfcauchy",
  prior_prec1_scale  = 1,
  prior_prec2_type   = "halfcauchy",
  prior_prec2_scale  = 1,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 1790
); plot(out_probit_trend_hc, ask = FALSE)

## 4.5 Local Acceleration with Logit Link (Half-Cauchy)
cat("\n=== Running Local Acceleration Model with Logit Link (Half-Cauchy) ===\n")
out_logit_accel_hc <- mcmc_normal_mixture_localacceleration(
  y,
  link                    = "logit",
  burnin                  = 50000,
  thinning                = 1000,
  n_draws                 = 1000,
  prior_mu01_mean         = NULL,
  prior_mu01_prec         = 1 / (1 * var(y)),
  prior_prec01_type       = "halfcauchy",
  prior_prec01_scale      = sd_y,
  prior_mu02_mean         = NULL,
  prior_mu02_prec         = 1 / (1 * var(y)),
  prior_prec02_type       = "halfcauchy",
  prior_prec02_scale      = sd_y,
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  prior_theta02_mean      = 0,
  prior_theta02_prec      = 1,
  prior_theta03_mean      = 0,
  prior_theta03_prec      = 1,
  # Half-Cauchy prior on all three state innovation SDs (level, slope, accel)
  prior_prec1_type        = "halfcauchy",
  prior_prec1_scale       = 1,
  prior_prec2_type        = "halfcauchy",
  prior_prec2_scale       = 1,
  prior_prec3_type        = "halfcauchy",
  prior_prec3_scale       = 1,
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 10,
  decay_exponent          = 0.5,
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 1458
); plot(out_logit_accel_hc, ask = FALSE)

## 4.6 Local Acceleration with Probit Link (Half-Cauchy)
cat("\n=== Running Local Acceleration Model with Probit Link (Half-Cauchy) ===\n")
out_probit_accel_hc <- mcmc_normal_mixture_localacceleration(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 500,
  n_draws            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (1 * var(y)),
  prior_prec01_type  = "halfcauchy",
  prior_prec01_scale = sd_y,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (1 * var(y)),
  prior_prec02_type  = "halfcauchy",
  prior_prec02_scale = sd_y,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_theta03_mean = 0,
  prior_theta03_prec = 1,
  prior_prec1_type   = "halfcauchy",
  prior_prec1_scale  = 1,
  prior_prec2_type   = "halfcauchy",
  prior_prec2_scale  = 1,
  prior_prec3_type   = "halfcauchy",
  prior_prec3_scale  = 1,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 1123
); plot(out_probit_accel_hc, ask = FALSE)

################################################################################
# 5. RESULTS SUMMARY
################################################################################
cat("\n=== Analysis Complete ===\n")
cat("Models fitted (Gamma prior on precisions):\n")
cat("  1. Local Level        - Logit and Probit links\n")
cat("  2. Local Trend        - Logit and Probit links\n")
cat("  3. Local Acceleration - Logit and Probit links\n")
cat("Models fitted (Half-Cauchy prior on standard deviations):\n")
cat("  4. Local Level / Trend / Acceleration - Logit and Probit links\n")
cat("\nResults stored in:\n")
cat("  Gamma:        out_logit_level,    out_probit_level\n")
cat("                out_logit_trend,    out_probit_trend\n")
cat("                out_logit_accel,    out_probit_accel\n")
cat("  Half-Cauchy:  out_logit_level_hc, out_probit_level_hc\n")
cat("                out_logit_trend_hc, out_probit_trend_hc\n")
cat("                out_logit_accel_hc, out_probit_accel_hc\n")

# Optional: Compare models (uncomment if you have comparison functions)
# compare_models(out_logit_level, out_logit_trend, out_logit_accel)
# compare_models(out_probit_level, out_probit_trend, out_probit_accel)
# compare_models(out_logit_level_hc, out_logit_trend_hc, out_logit_accel_hc)
# compare_models(out_probit_level_hc, out_probit_trend_hc, out_probit_accel_hc)

################################################################################
# END OF SCRIPT
################################################################################
