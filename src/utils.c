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
                             base_adaptation_rate / pow((double)iter, decay_exponent));

    // Update log_sigma based on acceptance rate deviation from target
    // If accrate > target: increase sigma (larger steps, lower acceptance)
    // If accrate < target: decrease sigma (smaller steps, higher acceptance)
    double deviation = accrate[k] - target_acceptance;
    log_sigma[k] += (deviation > 0 ? 1.0 : -1.0) * step_size;
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
