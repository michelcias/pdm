#' @title Gibbs Sampler for a Gaussian Mixture Model with Dynamic Mixture Weights
#'
#' @description Runs a Gibbs sampler for a two-component Gaussian mixture model
#'   with time-varying mixture weights following a local-trend polynomial dynamic
#'   structure. Supports both logit and probit link functions.
#'
#' @details The model is defined as:
#' \deqn{
#' \begin{aligned}
#' y_t | z_t, \mu, \phi &\sim N(z_t \cdot \mu_2 + (1 - z_t) \cdot \mu_1,
#'                              [z_t \cdot \phi_2 + (1 - z_t) \cdot \phi_1]^{-1}), \\
#' z_t | \alpha_t &\sim \text{Bernoulli}(\alpha_t), \\
#' \alpha_t &= T^{-1}(\theta_{t,1}), \\
#' \theta_{t,1} &= \theta_{t-1,1} + \theta_{t-1,2} + u_{t,1}, & u_{t,1} \sim N(0, W_1), \\
#' \theta_{t,2} &= \theta_{t-1,2} + u_{t,2},                  & u_{t,2} \sim N(0, W_2),
#' \end{aligned}
#' }
#' where \eqn{t = 1, 2, \ldots, n}, \eqn{n} is the number of observations,
#' \eqn{z_t \in \{0, 1\}} are latent indicators (0 = component 1, 1 = component 2),
#' \eqn{\mu = (\mu_1, \mu_2)'} are component means, and \eqn{\phi = (\phi_1, \phi_2)'}
#' are component precisions.
#'
#' The link function \eqn{T^{-1}} determines how the latent state \eqn{\theta_{t,1}}
#' maps to the mixture weight \eqn{\alpha_t}:
#'
#' \strong{Logit Link:}
#' \deqn{\alpha_t = \frac{\exp(\theta_{t,1})}{1 + \exp(\theta_{t,1})}}
#' Uses component-wise Metropolis-Hastings with adaptive tuning. Allows optional
#' diagnostics (log_sigma, accept_prop) to monitor proposal adaptation and
#' acceptance rates.
#'
#' \strong{Probit Link:}
#' \deqn{\alpha_t = \Phi(\theta_{t,1})}
#' where \eqn{\Phi(\cdot)} is the standard normal CDF. Uses Albert-Chib (1993)
#' data augmentation for efficient Gibbs sampling. Since this is a pure Gibbs
#' sampler, the acceptance rate is always 1.0.
#'
#' \strong{Prior Distributions:}
#'
#' \emph{Mixture Component Means and Initial States:}
#' \deqn{
#' \begin{aligned}
#' \mu_1 &\sim N(\mu_{01}, \tau_{01}^{-1}), \quad
#' \mu_2 \sim N(\mu_{02}, \tau_{02}^{-1}), \\
#' \theta_{0,1} &\sim N(\mu_{\theta_{01}}, \tau_{\theta_{01}}^{-1}), \quad
#' \theta_{0,2} \sim N(\mu_{\theta_{02}}, \tau_{\theta_{02}}^{-1}).
#' \end{aligned}
#' }
#'
#' \emph{Precision Parameters:}
#'
#' Every precision -- the two mixture component precisions \eqn{\phi_1},
#' \eqn{\phi_2} and the state innovation precisions \eqn{W_1^{-1}},
#' \eqn{W_2^{-1}} -- may take one of two priors, chosen independently through the
#' corresponding `*_type` argument:
#'
#' \emph{(a) Gamma prior on the precision} (default, conjugate):
#' \deqn{\phi_k \sim \text{Gamma}(\nu_{0k}, \eta_{0k}), \qquad
#'       W_j^{-1} \sim \text{Gamma}(\nu_j, \eta_j).}
#'
#' \emph{(b) Half-t prior on the standard deviation} (Gelman, 2006):
#' \deqn{\sqrt{1/\phi_k} \sim \text{Half-}t(\nu_{0k}, A_{0k}), \qquad
#'       \sqrt{W_j} \sim \text{Half-}t(\nu_j, A_j),}
#' i.e. a Half-t on each component standard deviation \eqn{\sqrt{1/\phi_k}} and on
#' each innovation standard deviation \eqn{\sqrt{W_j}}. Here \eqn{A > 0} is a
#' scale hyperparameter and \eqn{\nu > 0} the degrees of freedom; \eqn{\nu = 1}
#' gives the Half-Cauchy prior (also selectable with `type = "halfcauchy"`), and
#' larger \eqn{\nu} approaches a Half-Normal. The Half-t prior is represented by
#' the inverse-gamma scale mixture of Wand et al. (2011), which keeps every Gibbs
#' update closed-form via a single auxiliary variable per precision.
#'
#' The Gamma shape/rate arguments are required only when the corresponding
#' `_type` is `"gamma"`; the Half-t scale/df arguments only when it is `"halft"`
#' or `"halfcauchy"`. Each prior choice is resolved once, before the sampler
#' runs, and never re-evaluated inside the MCMC loop.
#'
#' The prior hyperparameters correspond to function arguments as follows:
#' \tabular{cc}{
#'   \strong{Hyperparameter} \tab \strong{Function Argument} \cr
#'   \eqn{\mu_{01}}, \eqn{\tau_{01}} \tab `prior_mu01_mean`, `prior_mu01_prec` \cr
#'   \eqn{\nu_{01}} (shape/df), \eqn{\eta_{01}} / \eqn{A_{01}} \tab `prior_prec01_{shape,rate,scale,df}` \cr
#'   \eqn{\mu_{02}}, \eqn{\tau_{02}} \tab `prior_mu02_mean`, `prior_mu02_prec` \cr
#'   \eqn{\nu_{02}} (shape/df), \eqn{\eta_{02}} / \eqn{A_{02}} \tab `prior_prec02_{shape,rate,scale,df}` \cr
#'   \eqn{\mu_{\theta_{01}}}, \eqn{\tau_{\theta_{01}}} \tab `prior_theta01_mean`, `prior_theta01_prec` \cr
#'   \eqn{\mu_{\theta_{02}}}, \eqn{\tau_{\theta_{02}}} \tab `prior_theta02_mean`, `prior_theta02_prec` \cr
#'   \eqn{\nu_1} (shape/df), \eqn{\eta_1} / \eqn{A_1} \tab `prior_prec1_{shape,rate,scale,df}` \cr
#'   \eqn{\nu_2} (shape/df), \eqn{\eta_2} / \eqn{A_2} \tab `prior_prec2_{shape,rate,scale,df}`
#' }
#'
#' \strong{Label Switching:}
#' To ensure identifiability, the constraint \eqn{\mu_1 < \mu_2} is enforced
#' by swapping components when necessary during sampling. This means component 1
#' always corresponds to the mixture component with the smaller mean.
#'
#' \strong{Adaptive Metropolis-Hastings Algorithm (Logit Link Only):}
#'
#' When using the logit link, the algorithm employs component-wise Metropolis-Hastings
#' for sampling the latent states \eqn{\theta_{t,1}}, with adaptive proposal tuning
#' based on acceptance proportions.
#'
#' The adaptive algorithm uses a diminishing adaptation schedule that ensures
#' theoretical convergence guarantees (Roberts and Rosenthal, 2007). Adaptation
#' occurs every `lag_update` iterations (e.g., at MCMC iterations 50, 100, 150, ...
#' if `lag_update = 50`), and only after sufficient history has been accumulated
#' (iteration >= `lag_update`).
#'
#' At each adaptation point (MCMC iteration \eqn{m}), for each time point \eqn{t},
#' the proposal log-scale standard deviation is updated according to:
#'
#' \deqn{\log(\sigma_t) \leftarrow \log(\sigma_t) + \text{sign}(\hat{p}_t - p^*) \cdot \gamma_m \cdot \mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}},}
#'
#' where:
#' \itemize{
#'   \item \eqn{\gamma_m} is the diminishing step size at MCMC iteration \eqn{m}, computed as
#'     \deqn{\gamma_m = \min\left(\text{max\_step\_size}, \frac{\text{base\_adaptation\_rate}}{m^\xi}\right);}
#'   \item \eqn{\xi} is the decay exponent (`decay_exponent`) applied to the MCMC iteration number;
#'   \item \eqn{\hat{p}_t} is the empirical acceptance proportion at time point \eqn{t}
#'     over the last `lag_update` MCMC iterations;
#'   \item \eqn{p^*} is the target acceptance rate (`target_acceptance`);
#'   \item \eqn{\tau} is the minimum deviation threshold (`min_deviation_threshold`);
#'   \item \eqn{\mathbb{1}_{\{|\hat{p}_t - p^*| > \tau\}}} is an indicator function that equals 1 when
#'     the condition is true, 0 otherwise.
#' }
#'
#' The update only occurs if the absolute deviation exceeds the threshold:
#' \deqn{|\hat{p}_t - p^*| > \tau.}
#'
#' This threshold-based approach prevents spurious updates due to random fluctuations.
#'
#' \strong{Parameter Interactions:}
#'
#' The adaptive tuning parameters interact as follows:
#' \itemize{
#'   \item \strong{lag_update}: Controls both the sliding window size for computing acceptance
#'     proportions AND the frequency of adaptation. Adaptation occurs at MCMC iterations
#'     \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Larger values
#'     provide more stable estimates but slower adaptation. Common choices: 50-200 iterations.
#'   \item \strong{target_acceptance}: Optimal acceptance rate for the Metropolis-Hastings
#'     algorithm. The value 0.44 is theoretically optimal for univariate random-walk proposals
#'     (Roberts and Rosenthal, 2001). Each time point \eqn{t} has its own acceptance rate
#'     \eqn{\hat{p}_t}.
#'   \item \strong{min_deviation_threshold}: Minimum deviation \eqn{|\hat{p}_t - p^*|}
#'     required to trigger adaptation for time point \eqn{t}. The default `NULL` uses the
#'     practical threshold \eqn{1/\text{lag\_update}}, corresponding to one additional
#'     acceptance/rejection in the sliding window.
#'   \item \strong{max_step_size}: Maximum allowed change in log-scale proposal variance
#'     per adaptation step. Prevents extreme adjustments. Common choices: 0.01-0.1.
#'   \item \strong{base_adaptation_rate}: Controls the overall speed of adaptation before
#'     decay is applied. Higher values lead to faster but potentially less stable adaptation.
#'     Common choices: 0.1-10.0.
#'   \item \strong{decay_exponent}: Controls how quickly the adaptation step size diminishes
#'     over MCMC iterations. As the algorithm runs, \eqn{\gamma_m} decreases according to
#'     \eqn{m^{-\xi}}. Must be in (0.5, 1] for theoretical convergence guarantees.
#'     Common choices: 0.5-0.8.
#' }
#'
#' \strong{Example of Adaptation Schedule:}
#'
#' With `lag_update = 50`, `base_adaptation_rate = 1.0`, `decay_exponent = 0.6`,
#' and `max_step_size = 0.1`:
#' \itemize{
#'   \item At MCMC iteration 50: \eqn{\gamma_{50} = \min(0.1, 1.0/50^{0.6}) \approx 0.0875}.
#'   \item At MCMC iteration 100: \eqn{\gamma_{100} = \min(0.1, 1.0/100^{0.6}) \approx 0.0631}.
#'   \item At MCMC iteration 1000: \eqn{\gamma_{1000} = \min(0.1, 1.0/1000^{0.6}) \approx 0.0158}.
#' }
#'
#' This ensures that adaptation becomes increasingly conservative as the chain progresses,
#' satisfying theoretical requirements for ergodicity.
#'
#' \strong{Recommended Settings:}
#'
#' For most applications:
#' \itemize{
#'   \item `lag_update = 50`: Provides good balance between stability and responsiveness;
#'   \item `max_step_size = 0.1`: Conservative adjustment rate;
#'   \item `base_adaptation_rate = 1.0`: Moderate adaptation speed;
#'   \item `decay_exponent = 0.6`: Standard diminishing adaptation;
#'   \item `target_acceptance = 0.44`: Theoretically optimal for univariate proposals;
#'   \item `min_deviation_threshold = NULL`: Uses practical default of 1/lag_update.
#' }
#'
#' \strong{Progress Bar:}
#' When `verbose = TRUE`, a visual progress bar is displayed showing
#' \itemize{
#' \item{Progress bar with adaptive update frequency (based on `bar_width`)}
#' \item{Elapsed time in HH:MM:SS format}
#' \item{Estimated remaining time in HH:MM:SS format}
#' }
#'
#' The progress bar update frequency is automatically calculated as approximately
#' one update per bar segment, ensuring smooth visual feedback with minimal
#' performance overhead (~0.01% for typical runs).
#'
#' Burn-in and thinning are applied so that exactly `n_chain` posterior
#' samples are returned.
#'
#' @param y Numeric vector of observed data (length \eqn{n}).
#' @param link Character string specifying the link function. Must be either
#'   `"logit"` or `"probit"`. Default is `"logit"`.
#' @param burnin Integer \eqn{\geq 0}, number of burn-in iterations.
#' @param thinning Integer \eqn{\geq 1}, thinning interval.
#' @param n_chain Integer \eqn{\geq 1}, number of posterior samples to retain.
#' @param prior_mu01_mean Numeric, prior mean for the mean of component 1 (\eqn{\mu_1}).
#'   If `NULL` (default), set to the 25th percentile of `y`. The components are
#'   identified by the constraint \eqn{\mu_1 < \mu_2}, which the sampler enforces
#'   by relabelling after each draw, so component 1 is the lower one by
#'   construction; specifying `prior_mu01_mean` above `prior_mu02_mean` asks for
#'   the reverse and raises a warning.
#' @param prior_mu01_prec Numeric > 0, prior precision (inverse variance) for \eqn{\mu_1}.
#'   Default is 0.01 (vague prior).
#' @param prior_prec01_type,prior_prec01_scale,prior_prec01_df Prior on the
#'   component-1 precision \eqn{\phi_1}: `type` is `"gamma"` (default), `"halft"`,
#'   or `"halfcauchy"` (a Half-t / Half-Cauchy prior on \eqn{\sqrt{1/\phi_1}};
#'   Gelman, 2006); `scale` \eqn{> 0} is the Half-t scale (required for Half-t);
#'   `df` \eqn{> 0} the Half-t degrees of freedom (default `1`, Half-Cauchy).
#' @param prior_prec01_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the precision of component 1 (\eqn{\phi_1}). Default is 2, following
#'   Richardson and Green (1997); values below 1 make the prior improper-like on
#'   the data scale and leave the degenerate region (a component collapsing onto
#'   a few observations, driving its precision up without limit) barely
#'   penalised. Used only when `prior_prec01_type = "gamma"`.
#' @param prior_prec01_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{\phi_1}.
#'   If `NULL` (default), set to `prior_prec01_shape * diff(range(y))^2 / 100`,
#'   the Richardson and Green (1997) data-range scaling. Used only when
#'   `prior_prec01_type = "gamma"`.
#' @param prior_mu02_mean Numeric, prior mean for the mean of component 2 (\eqn{\mu_2}).
#'   If `NULL` (default), set to the 75th percentile of `y`.
#' @param prior_mu02_prec Numeric > 0, prior precision (inverse variance) for \eqn{\mu_2}.
#'   Default is 0.01 (vague prior).
#' @param prior_prec02_type,prior_prec02_scale,prior_prec02_df As above for the
#'   component-2 precision \eqn{\phi_2} (Half-t placed on \eqn{\sqrt{1/\phi_2}}).
#' @param prior_prec02_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the precision of component 2 (\eqn{\phi_2}). Default is 2, following
#'   Richardson and Green (1997); values below 1 make the prior improper-like on
#'   the data scale and leave the degenerate region (a component collapsing onto
#'   a few observations, driving its precision up without limit) barely
#'   penalised. Used only when `prior_prec02_type = "gamma"`.
#' @param prior_prec02_rate Numeric > 0, rate parameter of the Gamma prior for \eqn{\phi_2}.
#'   If `NULL` (default), set to `prior_prec02_shape * diff(range(y))^2 / 100`,
#'   the Richardson and Green (1997) data-range scaling. Used only when
#'   `prior_prec02_type = "gamma"`.
#' @param prior_theta01_mean Numeric, prior mean for the initial level state
#'   \eqn{\theta_{0,1}}. Default is 0.
#' @param prior_theta01_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,1}}. Default is 0.01 (vague prior).
#' @param prior_theta02_mean Numeric, prior mean for the initial trend state
#'   \eqn{\theta_{0,2}}. Default is 0.
#' @param prior_theta02_prec Numeric > 0, prior precision (inverse variance) for
#'   \eqn{\theta_{0,2}}. Default is 0.01 (vague prior).
#' @param prior_prec1_type,prior_prec1_scale,prior_prec1_df Prior on the level
#'   innovation precision \eqn{1/W_1} (Half-t placed on \eqn{\sqrt{W_1}}); see
#'   `prior_prec01_type` for the argument meanings. Defaults to
#'   `"halfcauchy"` with `scale = 2` on the link scale since 0.5-0; a bare
#'   `shape`/`rate` pair without a named type is still read as `"gamma"`.
#' @param prior_prec1_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the level innovation precision \eqn{1/W_1}. Default is 0.01 (vague prior).
#'   Used only when `prior_prec1_type = "gamma"`.
#' @param prior_prec1_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_1}. Default is 0.01 (vague prior). Used only when `prior_prec1_type = "gamma"`.
#' @param prior_prec2_type,prior_prec2_scale,prior_prec2_df Prior on the trend
#'   innovation precision \eqn{1/W_2} (Half-t placed on \eqn{\sqrt{W_2}}); see
#'   `prior_prec01_type` for the argument meanings. Defaults to
#'   `"halfcauchy"` with `scale = 2` on the link scale since 0.5-0; a bare
#'   `shape`/`rate` pair without a named type is still read as `"gamma"`.
#' @param prior_prec2_shape Numeric > 0, shape parameter of the Gamma prior for
#'   the trend innovation precision \eqn{1/W_2}. Default is 0.01 (vague prior).
#'   Used only when `prior_prec2_type = "gamma"`.
#' @param prior_prec2_rate Numeric > 0, rate parameter of the Gamma prior for
#'   \eqn{1/W_2}. Default is 0.01 (vague prior). Used only when `prior_prec2_type = "gamma"`.
#' @param lag_update Integer \eqn{\geq 1}, adaptation frequency (sliding window size) for
#'   computing acceptance proportions (logit link only). Adaptation occurs at MCMC iterations
#'   \eqn{m = k \cdot \text{lag\_update}} for \eqn{k = 1, 2, 3, \ldots}. Controls both
#'   the sliding window size for computing acceptance proportions AND the frequency of
#'   adaptation. Larger values provide more stable estimates but slower adaptation.
#'   Default is 50. Ignored when `link = "probit"`.
#' @param max_step_size Numeric > 0, maximum allowed change in log-scale proposal variance
#'   per adaptation step (logit link only). Prevents extreme adjustments in proposal
#'   scaling. Default is 0.1. Ignored when `link = "probit"`.
#' @param base_adaptation_rate Numeric > 0, base rate controlling adaptation speed
#'   (logit link only). Higher values lead to faster but potentially less stable adaptation.
#'   Controls the magnitude of adaptation adjustments before decay is applied. Default
#'   is 1.0. Ignored when `link = "probit"`.
#' @param decay_exponent Numeric > 0, exponent controlling diminishing adaptation
#'   rate (logit link only). The adaptation rate decays as \eqn{m^{-\text{decay\_exponent}}}
#'   to satisfy diminishing adaptation conditions. Must be in (0.5, 1] for theoretical
#'   convergence guarantees. Default is 0.6. Ignored when `link = "probit"`.
#' @param target_acceptance Numeric in (0, 1), target acceptance rate for the
#'   Metropolis-Hastings algorithm (logit link only). The value 0.44 is theoretically
#'   optimal for univariate random-walk proposals (Roberts and Rosenthal, 2001).
#'   Proposal scales are adjusted to achieve this rate. Default is 0.44.
#'   Ignored when `link = "probit"`.
#' @param min_deviation_threshold Numeric \eqn{\geq 0} or `NULL`, minimum absolute deviation
#'   from `target_acceptance` required to trigger adaptation (logit link only). If `NULL`
#'   (default), uses practical threshold of `1.0/lag_update`, corresponding to one additional
#'   acceptance/rejection in the sliding window. Set to `0.0` for maximum sensitivity
#'   (adapt for any deviation). Larger values make adaptation more conservative.
#'   Ignored when `link = "probit"`.
#' @param return_log_sigma Logical, whether to return the log proposal scales
#'   (logit link only). Useful for diagnosing adaptation behavior. Default is
#'   `FALSE`. Ignored when `link = "probit"`.
#' @param return_accept_prop Logical, whether to return the acceptance proportions
#'   (logit link only). Useful for diagnosing MCMC mixing. Default is `FALSE`.
#'   Ignored when `link = "probit"`.
#' @param verbose Logical, whether to display a progress bar during sampling.
#'   Default is `FALSE`.
#' @param bar_width Integer in \[10, 120\], width of the progress bar when
#'   `verbose = TRUE`. Default is `60`.
#' @param seed Optional integer used to set the random number generator seed.
#'   Default is `NULL`, which does not set the seed.
#'   When `chains > 1` it acts as a master seed from which each chain's own
#'   seed is drawn.
#' @param chains Integer \eqn{\geq 1}, number of independent chains to run.
#'   Default is `1`, which returns a single fitted object exactly as before.
#'   With `chains > 1` the sampler is re-run once per chain -- each starting
#'   from its own values drawn from the priors -- and an object of class
#'   `"pdm_mcmc_list"` is returned, which \code{\link{mcmc_convergence}} turns
#'   into R-hat and effective sample sizes. The progress bar is disabled in this
#'   case, since several bars sharing one console interleave.
#' @param parallel Logical, whether to run the chains through
#'   `parallel::mclapply()`. Used only when `chains > 1`. The result does not
#'   depend on this setting: every chain receives an explicit seed, so a given
#'   `seed` reproduces the same output sequentially or in parallel. Default is
#'   `FALSE`.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{mu_1}}{Numeric vector of length `n_chain` of posterior samples for
#'     the mean of component 1 (\eqn{\mu_1}). Component 1 is defined as the
#'     component with the smaller mean due to the label switching constraint.}
#'   \item{\code{prec_1}}{Numeric vector of length `n_chain` of posterior samples for
#'     the precision of component 1 (\eqn{\phi_1 = 1/\sigma_1^2}).}
#'   \item{\code{mu_2}}{Numeric vector of length `n_chain` of posterior samples for
#'     the mean of component 2 (\eqn{\mu_2}). Component 2 is defined as the
#'     component with the larger mean.}
#'   \item{\code{prec_2}}{Numeric vector of length `n_chain` of posterior samples for
#'     the precision of component 2 (\eqn{\phi_2 = 1/\sigma_2^2}).}
#'   \item{\code{theta_1}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent level state \eqn{\theta_{t,1}}.}
#'   \item{\code{theta_2}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the latent trend state \eqn{\theta_{t,2}}.}
#'   \item{\code{theta_01}}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial level state \eqn{\theta_{0,1}}.}
#'   \item{\code{theta_02}}{Numeric vector of length `n_chain` of posterior samples
#'     for the initial trend state \eqn{\theta_{0,2}}.}
#'   \item{\code{prec_theta1}}{Numeric vector of length `n_chain` of posterior samples
#'     for the level innovation precision \eqn{1/W_1}.}
#'   \item{\code{prec_theta2}}{Numeric vector of length `n_chain` of posterior samples
#'     for the trend innovation precision \eqn{1/W_2}.}
#'   \item{\code{alpha}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior
#'     samples for the time-varying mixture weights \eqn{\alpha_t = P(z_t = 1)}.}
#'   \item{\code{z}}{Numeric matrix \eqn{[n_{chain} \times n]} of posterior samples
#'     for the latent component indicators \eqn{z_t \in \{0, 1\}}. A value of 0
#'     indicates component 1 (smaller mean), and 1 indicates component 2 (larger mean).}
#'   \item{\code{log_sigma}}{(Optional, logit link only) Numeric matrix
#'     \eqn{[n_{chain} \times n]} of log proposal scales used in the
#'     Metropolis-Hastings algorithm. Only returned if `return_log_sigma = TRUE`
#'     and `link = "logit"`.}
#'   \item{\code{accept_prop}}{(Optional, logit link only) Numeric matrix
#'     \eqn{[n_{chain} \times n]} of acceptance proportions for each time point.
#'     Only returned if `return_accept_prop = TRUE` and `link = "logit"`.}
#' }
#'
#' When `chains > 1` the return value is instead an object of class
#' `c("pdm_mcmc_list", "list")` holding one such fit per chain; see
#' \code{\link{print.pdm_mcmc_list}}.
#'
#' @examples
#' ## Description
#' # This example demonstrates how to:
#' # 1. Simulate data from a Gaussian mixture with sinusoidal mixture weights
#' # 2. Use `mcmc_normal_mixture_localtrend` to estimate parameters and latent states
#' # 3. Compare logit and probit link functions
#' # 4. Perform a detailed posterior analysis with visualizations
#' # 5. Set a seed for reproducibility
#' # 6. Use progress bar for monitoring MCMC execution
#'
#' ## Simulation of data
#' n <- 200  # Number of observations to simulate
#'
#' # Use a fixed seed for data simulation
#' set.seed(123)
#'
#' # Generate true mixture weights following a sinusoidal pattern
#' grid_vals <- seq_len(n) / n
#' alpha_true <- (sin(2 * pi * grid_vals) + sin(4 * pi * grid_vals) + 2) / 4
#'
#' # Generate latent component indicators
#' z_true <- rbinom(n, size = 1, prob = alpha_true)
#'
#' # Generate observations from mixture of N(0, 1/4) and N(2, 1/4)
#' mu_1_true <- 0
#' mu_2_true <- 2
#' sigma_1_true <- 0.5  # sqrt(1/4)
#' sigma_2_true <- 0.5  # sqrt(1/4)
#'
#' mu_y <- (1 - z_true) * mu_1_true + z_true * mu_2_true
#' sigma_y <- (1 - z_true) * sigma_1_true + z_true * sigma_2_true
#' y <- rnorm(n, mean = mu_y, sd = sigma_y)
#'
#' ## Running the Gibbs sampler with logit link
#' out_logit <- mcmc_normal_mixture_localtrend(
#'   y,
#'   link                    = "logit",
#'   burnin                  = 1000,
#'   thinning                = 10,
#'   n_chain                 = 500,
#'   prior_mu01_mean         = NULL,  # Use default (25th percentile)
#'   prior_mu01_prec         = 0.01,
#'   prior_prec01_shape      = 0.01,
#'   prior_prec01_rate       = 0.01,
#'   prior_mu02_mean         = NULL,  # Use default (75th percentile)
#'   prior_mu02_prec         = 0.01,
#'   prior_prec02_shape      = 0.01,
#'   prior_prec02_rate       = 0.01,
#'   prior_theta01_mean      = 0,
#'   prior_theta01_prec      = 1,
#'   prior_theta02_mean      = 0,
#'   prior_theta02_prec      = 1,
#'   prior_prec1_shape       = 100,
#'   prior_prec1_rate        = 1,
#'   prior_prec2_shape       = 400,
#'   prior_prec2_rate        = 1,
#'   lag_update              = 50,
#'   max_step_size           = 0.1,
#'   base_adaptation_rate    = 1.0,
#'   decay_exponent          = 0.6,
#'   target_acceptance       = 0.44,
#'   min_deviation_threshold = NULL,  # Use default (1/lag_update)
#'   return_log_sigma        = FALSE,
#'   return_accept_prop      = FALSE,
#'   verbose                 = TRUE,
#'   bar_width               = 60,
#'   seed                    = 456
#' )
#'
#' ## Running the Gibbs sampler with probit link
#' out_probit <- mcmc_normal_mixture_localtrend(
#'   y,
#'   link               = "probit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_mu01_mean    = NULL,
#'   prior_mu01_prec    = 0.01,
#'   prior_prec01_shape = 0.01,
#'   prior_prec01_rate  = 0.01,
#'   prior_mu02_mean    = NULL,
#'   prior_mu02_prec    = 0.01,
#'   prior_prec02_shape = 0.01,
#'   prior_prec02_rate  = 0.01,
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_shape  = 100,
#'   prior_prec1_rate   = 1,
#'   prior_prec2_shape  = 400,
#'   prior_prec2_rate   = 1,
#'   verbose            = TRUE,
#'   bar_width          = 60,
#'   seed               = 789
#' )
#'
#' ## Alternative: weakly-informative Half-Cauchy priors (Gelman, 2006)
#' # Put a Half-Cauchy prior on every precision: the component SDs
#' # sqrt(1/phi[1]), sqrt(1/phi[2]) (scale ~ sd(y)) and the level/trend
#' # weight-innovation SDs sqrt(W[1]), sqrt(W[2]) (scale ~ 1).
#' out_hc <- mcmc_normal_mixture_localtrend(
#'   y,
#'   link               = "probit",
#'   burnin             = 1000,
#'   thinning           = 10,
#'   n_chain            = 500,
#'   prior_prec01_type  = "halfcauchy",
#'   prior_prec01_scale = sd(y),
#'   prior_prec02_type  = "halfcauchy",
#'   prior_prec02_scale = sd(y),
#'   prior_theta01_mean = 0,
#'   prior_theta01_prec = 1,
#'   prior_theta02_mean = 0,
#'   prior_theta02_prec = 1,
#'   prior_prec1_type   = "halfcauchy",
#'   prior_prec1_scale  = 1,
#'   prior_prec2_type   = "halfcauchy",
#'   prior_prec2_scale  = 1,
#'   verbose            = FALSE,
#'   seed               = 789
#' )
#'
#' ## Posterior analysis and visualization
#' # Use the plot method for comprehensive diagnostics
#' \donttest{
#'   # --- Logit-link fit (`out_logit`) ---
#'   # Complete dashboard with all diagnostics
#'   plot(out_logit, type = "all")
#'
#'   # Individual diagnostic types
#'   plot(out_logit, type = "mcmc", which = 1:4)   # Mixture params (mu, phi)
#'   plot(out_logit, type = "mcmc", which = 5:8)   # Dynamic-state params
#'   plot(out_logit, type = "params")              # Mixture components
#'   plot(out_logit, type = "states")              # Dynamic states
#'   plot(out_logit, type = "alpha")               # Mixture weights
#'
#'   # Plot with true values for comparison
#'   plot(out_logit, type = "params",
#'        true_values = list(mu_1 = mu_1_true, mu_2 = mu_2_true,
#'                           prec_1 = 1 / sigma_1_true^2,
#'                           prec_2 = 1 / sigma_2_true^2))
#'   plot(out_logit, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#'
#'   # --- Probit-link fit (`out_probit`) ---
#'   # The same diagnostics for the probit-link fit, for comparison.
#'   plot(out_probit, type = "all")
#'
#'   plot(out_probit, type = "params")             # Mixture components
#'   plot(out_probit, type = "states")             # Dynamic states
#'   plot(out_probit, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#'
#'   # --- Half-Cauchy-prior fit (`out_hc`) ---
#'   # The same diagnostics for the weakly-informative Half-Cauchy fit.
#'   plot(out_hc, type = "all")
#'
#'   plot(out_hc, type = "alpha",
#'        true_values = list(alpha = alpha_true, z = z_true))
#' }
#'
#' @references
#' Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous
#' response data. \emph{Journal of the American Statistical Association}, 88(422),
#' 669-679. \doi{10.1080/01621459.1993.10476321}
#'
#' Gelman, A. (2006). Prior distributions for variance parameters in
#' hierarchical models. \emph{Bayesian Analysis}, 1(3), 515-534.
#'
#' Montoril, M. H., Correia, L. T., & Migon, H. S. (2021). Bayesian estimation of
#' dynamic weights in Gaussian mixture models. arXiv:2104.03395.
#'
#' Richardson, S., & Green, P. J. (1997). On Bayesian analysis of mixtures with
#' an unknown number of components (with discussion). \emph{Journal of the Royal
#' Statistical Society: Series B}, 59(4), 731-792.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various
#' Metropolis-Hastings algorithms. \emph{Statistical Science}, 16(4), 351-367.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC.
#' \emph{Journal of Applied Probability}, 44(2), 458-475.
#'
#' Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC.
#' \emph{Journal of Computational and Graphical Statistics}, 18(2), 349-367.
#' \doi{10.1198/jcgs.2009.06134}
#'
#' Wand, M. P., Ormerod, J. T., Padoan, S. A., & Fruhwirth, R. (2011). Mean field
#' variational Bayes for elaborate distributions. \emph{Bayesian Analysis},
#' 6(4), 847-900.
#'
#' @seealso
#'   \code{\link{plot.normal_mixture_localtrend}},
#'   \code{\link{print.normal_mixture_localtrend}},
#'   \code{\link{summary.normal_mixture_localtrend}} for methods on the fitted object;
#'   \code{\link{mcmc_normal_mixture_locallevel}} and
#'   \code{\link{mcmc_normal_mixture_localacceleration}} for the other dynamic orders.
#'
#' @export
mcmc_normal_mixture_localtrend <- function(y,
                                           link = c("logit", "probit"),
                                           burnin,
                                           thinning,
                                           n_chain,
                                           prior_mu01_mean = NULL,
                                           prior_mu01_prec = 0.01,
                                           prior_prec01_shape = 2,
                                           prior_prec01_rate = NULL,
                                           prior_mu02_mean = NULL,
                                           prior_mu02_prec = 0.01,
                                           prior_prec02_shape = 2,
                                           prior_prec02_rate = NULL,
                                           prior_theta01_mean = 0,
                                           prior_theta01_prec = 0.01,
                                           prior_theta02_mean = 0,
                                           prior_theta02_prec = 0.01,
                                           prior_prec1_shape = 0.01,
                                           prior_prec1_rate = 0.01,
                                           prior_prec2_shape = 0.01,
                                           prior_prec2_rate = 0.01,
                                           lag_update = 50,
                                           max_step_size = 0.1,
                                           base_adaptation_rate = 1.0,
                                           decay_exponent = 0.6,
                                           target_acceptance = 0.44,
                                           min_deviation_threshold = NULL,
                                           return_log_sigma = FALSE,
                                           return_accept_prop = FALSE,
                                           verbose = FALSE,
                                           bar_width = 60,
                                           seed = NULL,
                                           prior_prec01_type = c("gamma", "halfcauchy", "halft"),
                                           prior_prec01_scale = NULL,
                                           prior_prec01_df = 1,
                                           prior_prec02_type = c("gamma", "halfcauchy", "halft"),
                                           prior_prec02_scale = NULL,
                                           prior_prec02_df = 1,
                                           prior_prec1_type = c("halfcauchy", "gamma", "halft"),
                                           prior_prec1_scale = 2,
                                           prior_prec1_df = 1,
                                           prior_prec2_type = c("halfcauchy", "gamma", "halft"),
                                           prior_prec2_scale = 2,
                                           prior_prec2_df = 1,
                                           chains = 1,
                                           parallel = FALSE) {

  # --- Multi-chain dispatch ---
  # Re-issues this same call once per chain, each with its own seed, and returns
  # the collection. Kept at the very top so `match.call()` captures the call
  # exactly as the user wrote it.
  validate_chains_args(chains, parallel)
  if (chains > 1) {
    return(run_chains(match.call(), parent.frame(),
                      as.integer(chains), seed, parallel))
  }

  # `missing()` must be read before these arguments are touched.
  prec01_df_user_set <- !missing(prior_prec01_df)
  prec02_df_user_set <- !missing(prior_prec02_df)
  prec1_df_user_set  <- !missing(prior_prec1_df)
  prec2_df_user_set  <- !missing(prior_prec2_df)
  prior_prec01_type  <- match.arg(prior_prec01_type)
  prior_prec02_type  <- match.arg(prior_prec02_type)
  # Whether the type was named decides how a bare shape/rate pair is read
  # (see infer_gamma_from_hyperparams() in R/innovation_priors.R).
  prec1_type_user_set <- !missing(prior_prec1_type)
  prec2_type_user_set <- !missing(prior_prec2_type)

  prior_prec1_type   <- match.arg(prior_prec1_type)
  prior_prec2_type   <- match.arg(prior_prec2_type)

  prior_prec1_type <- infer_gamma_from_hyperparams(
    prior_prec1_type, prec1_type_user_set,
    !missing(prior_prec1_shape) || !missing(prior_prec1_rate)
  )

  prior_prec2_type <- infer_gamma_from_hyperparams(
    prior_prec2_type, prec2_type_user_set,
    !missing(prior_prec2_shape) || !missing(prior_prec2_rate)
  )
  # --- Input Validation ---

  # Validate y
  if (!is.numeric(y)) {
    stop("`y` must be a numeric vector")
  }
  if (!all(is.finite(y))) {
    stop("`y` must contain only finite numeric values (no NA, NaN, Inf)")
  }
  if (length(y) < 3) {
    stop("`y` must have at least 3 observations")
  }

  # Validate and match link argument
  link <- match.arg(link)

  # Validate MCMC control parameters
  if (!is.numeric(burnin) || length(burnin) != 1 || burnin < 0 ||
      burnin != floor(burnin)) {
    stop("`burnin` must be a single non-negative integer")
  }
  if (!is.numeric(thinning) || length(thinning) != 1 || thinning < 1 ||
      thinning != floor(thinning)) {
    stop("`thinning` must be a single positive integer")
  }
  if (!is.numeric(n_chain) || length(n_chain) != 1 || n_chain < 1 ||
      n_chain != floor(n_chain)) {
    stop("`n_chain` must be a single positive integer")
  }

  # Set intelligent defaults for mixture component priors based on data
  if (is.null(prior_mu01_mean)) {
    prior_mu01_mean <- as.numeric(quantile(y, 0.25))
  }
  if (is.null(prior_mu02_mean)) {
    prior_mu02_mean <- as.numeric(quantile(y, 0.75))
  }

  # The components are identified by the constraint mu_1 < mu_2; warn if the
  # priors ask for the reverse (see R/mixture_priors.R). Checked after the NULL
  # defaults are resolved, so the data-driven quantiles are covered too.
  check_mixture_mu_priors(prior_mu01_mean, prior_mu02_mean)

  # Richardson & Green (1997) data-scaled rate for the component precisions
  # (see rg_component_rate() in R/mixture_priors.R). Only the Gamma branch uses
  # a rate, so the Half-t path is left alone.
  if (prior_prec01_type == "gamma" && is.null(prior_prec01_rate)) {
    prior_prec01_rate <- rg_component_rate(y, prior_prec01_shape)
  }
  if (prior_prec02_type == "gamma" && is.null(prior_prec02_rate)) {
    prior_prec02_rate <- rg_component_rate(y, prior_prec02_shape)
  }

  # Validate mixture component prior parameters
  if (!is.numeric(prior_mu01_mean) || length(prior_mu01_mean) != 1) {
    stop("`prior_mu01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_mu01_prec) || length(prior_mu01_prec) != 1 ||
      prior_mu01_prec <= 0) {
    stop("`prior_mu01_prec` must be a single positive numeric value")
  }
  # Resolve the component-1 precision prior (Gamma or Half-t); see R/prec_prior.R.
  prec01_prior <- resolve_prec_prior(
    prior_prec01_type, prior_prec01_shape, prior_prec01_rate,
    prior_prec01_scale, prior_prec01_df, prec01_df_user_set, "prior_prec01"
  )

  if (!is.numeric(prior_mu02_mean) || length(prior_mu02_mean) != 1) {
    stop("`prior_mu02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_mu02_prec) || length(prior_mu02_prec) != 1 ||
      prior_mu02_prec <= 0) {
    stop("`prior_mu02_prec` must be a single positive numeric value")
  }
  # Resolve the component-2 precision prior (Gamma or Half-t).
  prec02_prior <- resolve_prec_prior(
    prior_prec02_type, prior_prec02_shape, prior_prec02_rate,
    prior_prec02_scale, prior_prec02_df, prec02_df_user_set, "prior_prec02"
  )

  # Validate dynamic state prior parameters
  if (!is.numeric(prior_theta01_mean) || length(prior_theta01_mean) != 1) {
    stop("`prior_theta01_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta01_prec) || length(prior_theta01_prec) != 1 ||
      prior_theta01_prec <= 0) {
    stop("`prior_theta01_prec` must be a single positive numeric value")
  }
  if (!is.numeric(prior_theta02_mean) || length(prior_theta02_mean) != 1) {
    stop("`prior_theta02_mean` must be a single numeric value")
  }
  if (!is.numeric(prior_theta02_prec) || length(prior_theta02_prec) != 1 ||
      prior_theta02_prec <= 0) {
    stop("`prior_theta02_prec` must be a single positive numeric value")
  }

  # Resolve the state innovation precision priors 1/W_1, 1/W_2 (Gamma or Half-t).
  prec1_prior <- resolve_prec_prior(
    prior_prec1_type, prior_prec1_shape, prior_prec1_rate,
    prior_prec1_scale, prior_prec1_df, prec1_df_user_set, "prior_prec1"
  )
  prec2_prior <- resolve_prec_prior(
    prior_prec2_type, prior_prec2_shape, prior_prec2_rate,
    prior_prec2_scale, prior_prec2_df, prec2_df_user_set, "prior_prec2"
  )

  # Validate adaptation parameters (used only for logit link)
  if (!is.numeric(lag_update) || length(lag_update) != 1 || lag_update < 1 ||
      lag_update != floor(lag_update)) {
    stop("`lag_update` must be a single positive integer")
  }
  if (!is.numeric(max_step_size) || length(max_step_size) != 1 ||
      max_step_size <= 0) {
    stop("`max_step_size` must be a single positive numeric value")
  }
  if (!is.numeric(base_adaptation_rate) || length(base_adaptation_rate) != 1 ||
      base_adaptation_rate <= 0) {
    stop("`base_adaptation_rate` must be a single positive numeric value")
  }
  if (!is.numeric(decay_exponent) || length(decay_exponent) != 1 ||
      decay_exponent <= 0) {
    stop("`decay_exponent` must be a single positive numeric value")
  }
  if (!is.numeric(target_acceptance) || length(target_acceptance) != 1 ||
      target_acceptance <= 0 || target_acceptance >= 1) {
    stop("`target_acceptance` must be a single numeric value in (0, 1)")
  }

  # Set default for min_deviation_threshold if NULL
  if (is.null(min_deviation_threshold)) {
    min_deviation_threshold <- 1.0 / lag_update
  }
  if (!is.numeric(min_deviation_threshold) || length(min_deviation_threshold) != 1 ||
      min_deviation_threshold < 0) {
    stop("`min_deviation_threshold` must be a single non-negative numeric value")
  }

  # Validate diagnostic output flags
  if (!is.logical(return_log_sigma) || length(return_log_sigma) != 1) {
    stop("`return_log_sigma` must be a single logical value")
  }
  if (!is.logical(return_accept_prop) || length(return_accept_prop) != 1) {
    stop("`return_accept_prop` must be a single logical value")
  }

  # Validate progress bar parameters
  if (!is.logical(verbose) || length(verbose) != 1) {
    stop("`verbose` must be a single logical value")
  }
  if (!is.numeric(bar_width) || length(bar_width) != 1 ||
      bar_width < 10 || bar_width > 120 || bar_width != floor(bar_width)) {
    stop("`bar_width` must be a single integer in [10, 120]")
  }

  # Validate and set seed if provided
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || seed != floor(seed)) {
      stop("`seed` must be a single integer value")
    }
    set.seed(seed)
  }

  # Issue warning if diagnostic flags are set for probit link (they will be ignored)
  if (link == "probit") {
    if (return_log_sigma) {
      warning("Argument `return_log_sigma` is ignored when link = 'probit'")
    }
    if (return_accept_prop) {
      warning("Argument `return_accept_prop` is ignored when link = 'probit'")
    }
  }

  # --- End Input Validation ---

  # Call the C function
  # Call the C function
  result <- .Call(
    "_pdm_C_MCMC_normal_mixture_localtrend",
    as.numeric(y),
    as.character(link),
    as.integer(burnin),
    as.integer(thinning),
    as.integer(n_chain),
    as.numeric(prior_mu01_mean),
    as.numeric(prior_mu01_prec),
    as.integer(prec01_prior$code),
    as.numeric(prec01_prior$shape),
    as.numeric(prec01_prior$rate),
    as.numeric(prec01_prior$scale),
    as.numeric(prec01_prior$df),
    as.numeric(prior_mu02_mean),
    as.numeric(prior_mu02_prec),
    as.integer(prec02_prior$code),
    as.numeric(prec02_prior$shape),
    as.numeric(prec02_prior$rate),
    as.numeric(prec02_prior$scale),
    as.numeric(prec02_prior$df),
    as.numeric(prior_theta01_mean),
    as.numeric(prior_theta01_prec),
    as.numeric(prior_theta02_mean),
    as.numeric(prior_theta02_prec),
    as.integer(prec1_prior$code),
    as.numeric(prec1_prior$shape),
    as.numeric(prec1_prior$rate),
    as.numeric(prec1_prior$scale),
    as.numeric(prec1_prior$df),
    as.integer(prec2_prior$code),
    as.numeric(prec2_prior$shape),
    as.numeric(prec2_prior$rate),
    as.numeric(prec2_prior$scale),
    as.numeric(prec2_prior$df),
    as.integer(lag_update),
    as.numeric(max_step_size),
    as.numeric(base_adaptation_rate),
    as.numeric(decay_exponent),
    as.numeric(target_acceptance),
    as.numeric(min_deviation_threshold),
    as.logical(return_log_sigma),
    as.logical(return_accept_prop),
    as.logical(verbose),
    as.integer(bar_width)
  )

  # Create and validate the class object
  result <- new_normal_mixture_localtrend(
    result = result,
    link = link,
    n_obs = length(y),
    n_chain = n_chain,
    burnin = burnin,
    thinning = thinning,
    y = y
  )

  result <- validate_normal_mixture_localtrend(result)

  # Record the priors used on the component precisions phi_k and the state
  # innovation precisions 1/W_1, 1/W_2 (auxiliary Half-t variables are nuisance
  # parameters and are intentionally not returned).
  attr(result, "prior_prec_phi1")   <- prec01_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_phi2")   <- prec02_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_theta1") <- prec1_prior[c("type", "shape", "rate", "scale", "df")]
  attr(result, "prior_prec_theta2") <- prec2_prior[c("type", "shape", "rate", "scale", "df")]

  # Record the RWMH target acceptance rate so plot(type = "acceptance") can draw
  # the correct reference line (logit link only; unused under the probit sampler).
  attr(result, "target_acceptance") <- as.numeric(target_acceptance)

  return(result)
}
