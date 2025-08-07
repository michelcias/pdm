#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>

/**
 * Computes the inverse logit (logistic) function of the input value.
 *
 * The inverse logit is defined as 1/(1 + exp(-x)) or equivalently exp(x)/(1 + exp(x)).
 * This implementation is numerically stable using Rf_log1pexp().
 *
 * Args:
 *   x (double): Input value.
 *
 * Returns:
 *   double: The result of the inverse logit transformation, in the interval (0, 1).
 */
double ilogit(double x) {
  return exp(-Rf_log1pexp(-x));
}

//----------------------------------------------------------------------

/**
 * Auxiliary function to generate a multivariate normal random vector
 * with precision matrix A, where A is a tridiagonal matrix with specific
 * structure.
 *
 * **Important Assumption:** This function assumes the dimension `n` is strictly greater than 2 (n > 2).
 * Behavior for n <= 2 is undefined as boundary case checks have been removed.
 *
 * The precision matrix A has:
 * - Diagonal elements: A[i,i] = (a + 2b) for i = 0,...,n-2 (0-based index)
 * - Last element: A[n-1,n-1] = (a + b) if add_a == 1, otherwise b
 * - Off-diagonal elements: A[i+1,i] = A[i,i+1] = -b for i = 0,...,n-2
 *
 * The function uses L*L' decomposition (Cholesky) for efficient computation:
 * 1. Decomposes A into L*L' where L is lower triangular
 * 2. Solves L'*x = u where L*u = y (forward substitution to find mean x)
 * 3. Generates random vector s by solving L'*s = z where z ~ N(0,I) (backward substitution)
 * 4. Returns r = x + s (combination of mean and random component)
 *
 * The mean vector of the distribution is the solution to A*x = y.
 *
 * @param r       Output array (vectorized B x n matrix) where the sampled n-variate
 *                normal vector will be stored at row `iter`.
 * @param y       Right-hand side vector (size n) of the system A*x = y, used to calculate the mean.
 * @param a       Constant 'a' for the precision matrix structure.
 * @param b       Constant 'b' for the precision matrix structure.
 * @param n       Dimension of the matrix/vectors (number of elements, indexed 0 to n-1). Must be > 2.
 * @param iter    Row index (0-based) in the output matrix `r` where the sample will be stored.
 * @param add_a   Flag (1 or 0) to determine the last diagonal element A[n-1,n-1].
 *                If 1, A[n-1,n-1] = a+b. If 0, A[n-1,n-1] = b.
 */
void generate_normal_vector(double *r,
                            double *y,
                            double a,
                            double b,
                            int n,
                            int iter,
                            int add_a) {
  /* ========== Matrix Parameters Setup ========== */
  double a11 = a + 2 * b;                                             // Standard diagonal element of A
  double ann = add_a == 1 ? a + b : b;                                // Last diagonal element of A
  int iter_n = iter * n;                                              // Iteration index for the r vector
  int i;                                                              // Loop index for matrix elements

  /* ========== Memory Allocation ========== */
  double *d = (double *)R_alloc(n, sizeof(double));                   // Diagonal of L
  double *l = (double *)R_alloc(n - 1, sizeof(double));               // Subdiagonal of L
  double *u = (double *)R_alloc(n, sizeof(double));                   // Temporary vector (L*u = y)
  double *x = (double *)R_alloc(n, sizeof(double));                   // Solution vector (A*x = y)

  /* ========== Cholesky Decomposition (L*L') and forward substitution (L*u = y) ========== */
  // First element (i = 0)
  d[0] = sqrt(a11);                                                   // L[0,0] = sqrt(A[0,0])
  u[0] = y[0] / d[0];                                                 // Forward substitution first step

  // Middle elements (i = 1 to n-2)
  for (i = 1; i < n - 1; i++) {
    l[i - 1] = -b / d[i - 1];                                         // L[i,i-1] = A[i,i-1]/L[i-1,i-1]
    d[i] = sqrt(a11 - l[i - 1] * l[i - 1]);                           // L[i,i] = sqrt(A[i,i] - L[i,i-1]^2)
    u[i] = (y[i] - l[i - 1] * u[i - 1]) / d[i];                       // Forward substitution
  }

  // Last element (i = n-1)
  l[n - 2] = -b / d[n - 2];                                           // L[n-1,n-2] = A[n-1,n-2]/L[n-2,n-2]
  d[n - 1] = sqrt(ann - l[n - 2] * l[n - 2]);                         // L[n-1,n-1] = sqrt(A[n-1,n-1] - L[n-1,n-2]^2)
  u[n - 1] = (y[n - 1] - l[n - 2] * u[n - 2]) / d[n - 1];             // Final forward step

  /* ========== Backward Substitution ========== */
  // Solutions to L'*x = u and L'*r = z
  // Last element (i = n-1)
  x[n - 1] = u[n - 1] / d[n - 1];                                     // x[n-1] = u[n-1]/L[n-1,n-1]
  r[iter_n + n - 1] = rnorm(0, 1) / d[n - 1];                         // r[iter, n-1] = z[n-1]/L[n-1,n-1]

  // Remaining elements (i = n-2 to 0)
  for (i = n - 2; i >= 0; i--) {
    x[i] = (u[i] - l[i] * x[i + 1]) / d[i];                           // x[i] = (u[i] - L[i+1,i]*x[i+1])/L[i,i]
    r[iter_n + i] = (rnorm(0, 1) - l[i] * r[iter_n + i + 1]) / d[i];  // r[iter, i] = (z[i] - L[i+1,i]*r[iter, i+1])/L[i,i]
    r[iter_n + i + 1] += x[i + 1];                                    // Incrementally add x[i+1] to r[iter, i+1]
  }

  r[iter_n] += x[0];                                                  // Final addition to r[iter, 0]

}

//----------------------------------------------------------------------

/**
 * Adapts CWMH (Component-Wise Metropolis-Hastings) parameters by updating
 * acceptance rates and log-sigma values for proposal variance tuning.
 *
 * This function implements an adaptive MCMC algorithm that automatically adjusts
 * the proposal variance (sigma) for each component of the parameter vector to
 * achieve user-specified target acceptance rates.
 *
 * The adaptation follows the diminishing adaptation principle:
 * - Uses a sliding window of recent iterations to compute acceptance rates
 * - Adjusts log_sigma based on deviation from target acceptance rate
 * - Adaptation step size decreases over iterations to ensure convergence
 *
 * @param theta_updated          Matrix of update indicators (1 if accepted, 0 if rejected).
 *                               Stored as vectorized (B x n) matrix where B = total iterations.
 * @param accrate                Output vector (size n) of acceptance rates for each component.
 * @param log_sigma              Input/output vector (size n) of log proposal standard deviations.
 *                               Will be updated based on current acceptance rates.
 * @param lag_update             Number of recent iterations to use for acceptance rate calculation.
 *                               Should be large enough for stable estimates (e.g., 50-100).
 * @param n                      Number of components in the parameter vector.
 * @param iter                   Current MCMC iteration (1-based). Must be > lag_update.
 * @param max_step_size          Maximum adaptation step size to ensure stability (e.g., 0.01).
 * @param base_adaptation_rate   Base adaptation rate controlling initial adaptation intensity.
 * @param decay_exponent         Exponent controlling decay speed (0.5 = sqrt decay, 1.0 = linear).
 * @param target_acceptance      Target acceptance rate for optimization (e.g., 0.44 for univariate,
 *                               0.234 for multivariate, 0.6 for aggressive burn-in).
 *
 * @note This function assumes iter > lag_update to have sufficient history for adaptation.
 * @note Common target acceptance rates: 0.44 (univariate MH), 0.234 (multivariate MH),
 *       0.6 (burn-in phase), 0.2-0.3 (high-dimensional problems).
 * @note Common values: max_step_size = 0.01, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8 (0.5 for classic Robbins-Monro).
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *accrate,
                           double *log_sigma,
                           int lag_update,
                           int n,
                           int iter,
                           double max_step_size,
                           double base_adaptation_rate,
                           double decay_exponent,
                           double target_acceptance) {

  // Calculate sliding window indices for acceptance rate computation
  int start_idx = (iter - lag_update - 1) * n;
  int end_idx = (iter - 1) * n;
  int k, row;

  // Compute acceptance rates for each component over the sliding window
  for(k = 0; k < n; k++) {
    accrate[k] = 0.0; // Initialize acceptance rate for component k

    // Sum acceptances over the sliding window
    for(row = start_idx; row < end_idx; row += n) {
      accrate[k] += theta_updated[row + k];
    }

    // Calculate average acceptance rate for component k
    accrate[k] /= (double)lag_update;

    // Flexible adaptive step size with user-controlled decay
    double step_size = fmin2(max_step_size,
                             base_adaptation_rate / R_pow((double)iter, decay_exponent));

    // Update log_sigma based on acceptance rate deviation from target
    // If accrate > target: increase sigma (larger steps, lower acceptance)
    // If accrate < target: decrease sigma (smaller steps, higher acceptance)
    double deviation = accrate[k] - target_acceptance;
    log_sigma[k] += (deviation > 0 ? 1.0 : -1.0) * step_size;
  }
}

//----------------------------------------------------------------------

/**
 * Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial dynamic model.
 *
 * This function implements a component-wise Metropolis-Hastings algorithm to sample
 * the level state vector theta_1 when the observation equation follows a binomial
 * distribution with logit link function: y_t ~ Binomial(n_trials, logit^{-1}(theta_{1,t})).
 *
 * The state equation remains linear: theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},
 * but the non-linear observation equation requires MCMC sampling instead of closed-form
 * Gibbs updates.
 *
 * **Important Note on Iteration Timing**: This function uses theta_01[iter-1] and
 * prec_theta_1[iter-1] because these parameters are sampled later in the Gibbs sequence
 * and thus their current iteration values are not yet available.
 *
 * The function handles three cases with appropriate boundary conditions:
 * - First element: incorporates initial states theta_01 and theta_02
 * - Intermediate elements: uses forward-sampling with previously updated values
 * - Last element: simplified structure without future state dependency
 *
 * @param theta_1            Matrix of level states (vectorized B x n), input/output.
 * @param theta_2            Matrix of trend states (vectorized B x n), input only.
 * @param theta_01           Vector of initial level states (size B).
 * @param theta_02           Vector of initial trend states (size B).
 * @param theta_1_updated    Matrix of acceptance indicators (vectorized B x n), output.
 * @param alpha              Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1       Vector of level precision parameters (size B).
 * @param pred_label         Vector of observed binomial counts (size n).
 * @param log_sigma          Vector of log proposal standard deviations (size n).
 * @param hat_theta_1        Temporary vector for conditional means (size n).
 * @param theta_1_new        Temporary vector for proposed values (size n).
 * @param log_accept_prob    Temporary vector for log acceptance probabilities (size n).
 * @param updated            Temporary vector for acceptance indicators (size n).
 * @param n_trials           Number of Bernoulli trials for binomial distribution.
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based).
 */
void CWMH_alpha_logit_binomial(double *theta_1,
                               double *theta_2,
                               double *theta_01,
                               double *theta_02,
                               double *theta_1_updated,
                               double *alpha,
                               double *prec_theta_1,
                               double *pred_label,
                               double *log_sigma,
                               double *hat_theta_1,
                               double *theta_1_new,
                               double *log_accept_prob,
                               int *updated,
                               double n_trials,
                               int n,
                               int iter) {

  int k;

  /* ========== Iteration Index Setup ========== */
  int iter_n = iter * n;                              // Current iteration start index
  int iterm1_n = iter_n - n;                          // Previous iteration start index
  int prev_iter = iter - 1;                           // Previous iteration for theta_01 and prec_theta_1

  /* ========== First Element (k = 0) ========== */
  double sigma_val = exp(log_sigma[0]);                // Proposal standard deviation

  // Conditional mean incorporating initial states from previous iteration
  hat_theta_1[0] = 0.5 * (theta_1[iterm1_n + 1] - theta_2[iter_n] +
    theta_01[prev_iter] + theta_02[prev_iter]);

  // Generate proposal from random walk
  theta_1_new[0] = rnorm(theta_1[iterm1_n], sigma_val);

  // Calculate log acceptance probability using previous iteration's precision
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], 1.0 / sqrt(prec_theta_1[prev_iter] * 2), 1);
  double lp2n = dbinom(pred_label[0], n_trials, ilogit(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1[iterm1_n], hat_theta_1[0], 1.0 / sqrt(prec_theta_1[prev_iter] * 2), 1);
  double lp2o = dbinom(pred_label[0], n_trials, ilogit(theta_1[iterm1_n]), 1);

  log_accept_prob[0] = lp1n + lp2n - lp1o - lp2o;
  updated[0] = log(runif(0, 1)) <= log_accept_prob[0];

  if (!updated[0]) {
    theta_1_new[0] = theta_1[iterm1_n];               // Reject: keep old value
  }

  /* ========== Intermediate Elements (k = 1 to n-2) ========== */
  for (k = 1; k < (n - 1); k++) {
    sigma_val = exp(log_sigma[k]);

    // Conditional mean using forward-sampling: previously updated theta_1_new[k-1]
    hat_theta_1[k] = 0.5 * (theta_1[iterm1_n + k + 1] - theta_2[iter_n + k] +
      theta_1_new[k - 1] - theta_2[iter_n + k - 1]);

    theta_1_new[k] = rnorm(theta_1[iterm1_n + k], sigma_val);

    // Log densities using previous iteration's precision
    lp1n = dnorm(theta_1_new[k], hat_theta_1[k], 1.0 / sqrt(prec_theta_1[prev_iter] * 2), 1);
    lp2n = dbinom(pred_label[k], n_trials, ilogit(theta_1_new[k]), 1);
    lp1o = dnorm(theta_1[iterm1_n + k], hat_theta_1[k], 1.0 / sqrt(prec_theta_1[prev_iter] * 2), 1);
    lp2o = dbinom(pred_label[k], n_trials, ilogit(theta_1[iterm1_n + k]), 1);

    log_accept_prob[k] = lp1n + lp2n - lp1o - lp2o;
    updated[k] = log(runif(0, 1)) <= log_accept_prob[k];

    if (!updated[k]) {
      theta_1_new[k] = theta_1[iterm1_n + k];
    }
  }

  /* ========== Last Element (k = n-1) ========== */
  sigma_val = exp(log_sigma[n - 1]);

  // Simplified conditional mean: no future state dependency
  hat_theta_1[n - 1] = theta_1_new[n - 2] + theta_2[iter_n + n - 2];

  theta_1_new[n - 1] = rnorm(theta_1[iterm1_n + (n - 1)], sigma_val);

  // Log densities using previous iteration's precision (different structure for boundary)
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], 1.0 / sqrt(prec_theta_1[prev_iter]), 1);
  lp2n = dbinom(pred_label[n - 1], n_trials, ilogit(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1[iterm1_n + (n - 1)], hat_theta_1[n - 1], 1.0 / sqrt(prec_theta_1[prev_iter]), 1);
  lp2o = dbinom(pred_label[n - 1], n_trials, ilogit(theta_1[iterm1_n + (n - 1)]), 1);

  log_accept_prob[n - 1] = lp1n + lp2n - lp1o - lp2o;
  updated[n - 1] = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!updated[n - 1]) {
    theta_1_new[n - 1] = theta_1[iterm1_n + (n - 1)];
  }

  /* ========== Update Output Arrays ========== */
  for (k = 0; k < n; k++) {
    theta_1[iter_n + k] = theta_1_new[k];                      // Store sampled states
    theta_1_updated[iter_n + k] = updated[k];                  // Store acceptance indicators
    alpha[iter_n + k] = ilogit(theta_1_new[k]);                // Store transformed probabilities
  }
}

//----------------------------------------------------------------------

/**
 * R interface wrapper for the ilogit function.
 *
 * Takes a numeric SEXP object from R, applies the ilogit transformation to its first element,
 * and returns the result as a scalar real SEXP.
 *
 * Args:
 *   x (SEXP): Numeric input from R (only the first element is used).
 *
 * Returns:
 *   SEXP: Scalar real containing the result of ilogit(x).
 */
SEXP C_ILogit(SEXP x) {
  double val = ilogit(REAL(x)[0]);
  return ScalarReal(val);
}
