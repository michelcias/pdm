################################################################################
# MCMC Analysis of Changepoint Data using Normal Mixture Models
################################################################################
# This script applies different state-space models (local level, local trend,
# and local acceleration) to the GBM29 dataset from the changepoint package.
# Each model is tested with both logit and probit link functions.
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
  n_chain                 = 1000,         # Number of samples to keep
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile of y
  prior_mu01_prec         = 1 / (10 * var(y)),
  # Prior for precision of first mixture component
  prior_prec01_shape      = 0.01,
  prior_prec01_rate       = 0.01,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile of y
  prior_mu02_prec         = 1 / (10 * var(y)),
  # Prior for precision of second mixture component
  prior_prec02_shape      = 0.01,
  prior_prec02_rate       = 0.01,
  # Prior for state equation parameters
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Prior for state precision
  prior_prec1_shape       = 0.01,
  prior_prec1_rate        = 0.01,
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
)
plot(out_logit_level)

## 1.2 Local Level with Probit Link
cat("\n=== Running Local Level Model with Probit Link ===\n")
out_probit_level <- mcmc_normal_mixture_locallevel(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 250,
  n_chain            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (10 * var(y)),
  prior_prec01_shape = 0.01,
  prior_prec01_rate  = 0.01,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (10 * var(y)),
  prior_prec02_shape = 0.01,
  prior_prec02_rate  = 0.01,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_prec1_shape  = 0.01,
  prior_prec1_rate   = 0.01,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 789
)
plot(out_probit_level)

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
  burnin                  = 150000,       # Increased burn-in for more complex model
  thinning                = 250,
  n_chain                 = 1000,
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile
  prior_mu01_prec         = 1 / (10 * var(y)),
  prior_prec01_shape      = 0.01,
  prior_prec01_rate       = 0.01,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile
  prior_mu02_prec         = 1 / (10 * var(y)),
  prior_prec02_shape      = 0.01,
  prior_prec02_rate       = 0.01,
  # Prior for state equation - level
  prior_theta01_mean      = 0,
  prior_theta01_prec      = 1,
  # Prior for state equation - slope
  prior_theta02_mean      = 0,
  prior_theta02_prec      = 1,
  # Prior for state precision - level
  prior_prec1_shape       = 0.01,
  prior_prec1_rate        = 0.01,
  # Prior for state precision - slope
  prior_prec2_shape       = 0.01,
  prior_prec2_rate        = 0.01,
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
  seed                    = 456
)
plot(out_logit_trend)

## 2.2 Local Trend with Probit Link
cat("\n=== Running Local Trend Model with Probit Link ===\n")
out_probit_trend <- mcmc_normal_mixture_localtrend(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 250,
  n_chain            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (10 * var(y)),
  prior_prec01_shape = 0.01,
  prior_prec01_rate  = 0.01,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (10 * var(y)),
  prior_prec02_shape = 0.01,
  prior_prec02_rate  = 0.01,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_prec1_shape  = 0.01,
  prior_prec1_rate   = 0.01,
  prior_prec2_shape  = 0.01,
  prior_prec2_rate   = 0.01,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 789
)
plot(out_probit_trend)

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
  burnin                  = 500000,       # Increased burn-in for most complex model
  thinning                = 1,
  n_chain                 = 1000,
  # Prior for mean of first mixture component
  prior_mu01_mean         = NULL,         # Default: 25th percentile
  prior_mu01_prec         = 1 / (10 * var(y)),
  prior_prec01_shape      = 0.01,
  prior_prec01_rate       = 0.01,
  # Prior for mean of second mixture component
  prior_mu02_mean         = NULL,         # Default: 75th percentile
  prior_mu02_prec         = 1 / (10 * var(y)),
  prior_prec02_shape      = 0.01,
  prior_prec02_rate       = 0.01,
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
  prior_prec1_shape       = 0.01,
  prior_prec1_rate        = 0.01,
  # Prior for state precision - slope
  prior_prec2_shape       = 0.01,
  prior_prec2_rate        = 0.01,
  # Prior for state precision - acceleration
  prior_prec3_shape       = 0.01,
  prior_prec3_rate        = 0.01,
  # Adaptive MALA parameters
  lag_update              = 50,
  max_step_size           = 0.1,
  base_adaptation_rate    = 10,
  decay_exponent          = 0.4,          # Slightly lower for better adaptation
  target_acceptance       = 0.44,
  min_deviation_threshold = NULL,         # Default: 1/lag_update
  # Output options
  return_log_sigma        = FALSE,
  return_accept_prop      = TRUE,
  verbose                 = TRUE,
  bar_width               = 60,
  seed                    = 456
)
plot(out_logit_accel, ask = FALSE)

## 3.2 Local Acceleration with Probit Link
cat("\n=== Running Local Acceleration Model with Probit Link ===\n")
out_probit_accel <- mcmc_normal_mixture_localacceleration(
  y,
  link               = "probit",
  burnin             = 50000,
  thinning           = 250,
  n_chain            = 1000,
  prior_mu01_mean    = NULL,
  prior_mu01_prec    = 1 / (10 * var(y)),
  prior_prec01_shape = 0.01,
  prior_prec01_rate  = 0.01,
  prior_mu02_mean    = NULL,
  prior_mu02_prec    = 1 / (10 * var(y)),
  prior_prec02_shape = 0.01,
  prior_prec02_rate  = 0.01,
  prior_theta01_mean = 0,
  prior_theta01_prec = 1,
  prior_theta02_mean = 0,
  prior_theta02_prec = 1,
  prior_theta03_mean = 0,
  prior_theta03_prec = 1,
  prior_prec1_shape  = 0.01,
  prior_prec1_rate   = 0.01,
  prior_prec2_shape  = 0.01,
  prior_prec2_rate   = 0.01,
  prior_prec3_shape  = 0.01,
  prior_prec3_rate   = 0.01,
  verbose            = TRUE,
  bar_width          = 60,
  seed               = 123
)
plot(out_probit_accel)

################################################################################
# 4. RESULTS SUMMARY
################################################################################
cat("\n=== Analysis Complete ===\n")
cat("Models fitted:\n")
cat("  1. Local Level    - Logit and Probit links\n")
cat("  2. Local Trend    - Logit and Probit links\n")
cat("  3. Local Acceleration - Logit and Probit links\n")
cat("\nResults stored in:\n")
cat("  - out_logit_level, out_probit_level\n")
cat("  - out_logit_trend, out_probit_trend\n")
cat("  - out_logit_accel, out_probit_accel\n")

# Optional: Compare models (uncomment if you have comparison functions)
# compare_models(out_logit_level, out_logit_trend, out_logit_accel)
# compare_models(out_probit_level, out_probit_trend, out_probit_accel)

################################################################################
# END OF SCRIPT
################################################################################
