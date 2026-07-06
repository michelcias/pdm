/**
 * @file utils.c
 * @brief Utility functions for polynomial dynamic models (pdm) package.
 * @author Michel H. Montoril
 * @date 2025-08-09
 * @version 1.0
 *
 * @details This file contains core utility functions including:
 *          - Inverse logit transformation (numerically stable)
 *          - Multivariate normal random vector generation with tridiagonal precision
 *          - R interface wrappers for core C functions with input validation
 */

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <float.h>          /* DBL_EPSILON */

/**
 * @brief Computes the inverse logit (logistic) function of the input value.
 *
 * @details The inverse logit is defined as 1/(1 + exp(-x)) or equivalently
 *          exp(x)/(1 + exp(x)). This implementation is numerically stable
 *          using Rf_log1pexp().
 *
 * @param x Input value.
 * @return The result of the inverse logit transformation, in the interval (0, 1).
 */
double ilogit(double x) {
  return exp(-Rf_log1pexp(-x));
}

//----------------------------------------------------------------------

/**
 * @brief Draw a Gamma variate guaranteed to be strictly positive.
 *
 * @details See utils.h for the full rationale. In short: with small shape
 *          parameters rgamma can underflow to a subnormal value or exactly
 *          0.0, and a zero precision poisons the MCMC (Inf/NaN via 1/prec,
 *          sqrt(1/prec) and zero Cholesky pivots). Flooring at DBL_EPSILON
 *          is statistically inert (variance ~4.5e15) and keeps every
 *          downstream computation finite. The negated comparison also traps
 *          NaN, so a degenerate draw resets to the floor instead of
 *          propagating.
 *
 * @param shape  Gamma shape parameter (> 0).
 * @param scale  Gamma scale parameter (> 0). Note: scale = 1/rate.
 * @return       Gamma(shape, scale) draw, floored at DBL_EPSILON.
 */
double rgamma_positive(double shape, double scale) {
  double x = rgamma(shape, scale);

  /* !(x > floor) is true for x <= floor AND for NaN */
  if (!(x > DBL_EPSILON)) {
    x = DBL_EPSILON;
  }

  return x;
}

//----------------------------------------------------------------------

/**
 * @brief Generate a multivariate normal random vector with structured tridiagonal precision matrix.
 *
 * @details This function generates samples from a multivariate normal distribution N(mu, Sigma)
 *          where the precision matrix A = Sigma^(-1) has a specific tridiagonal structure.
 *
 *          **Precision Matrix Structure:**
 *          - Diagonal elements: A[i,i] = (a + 2*b) for i = 0,...,n-2
 *          - Last diagonal: A[n-1,n-1] = (a + b) if add_a == 1, otherwise b
 *          - Off-diagonal: A[i+1,i] = A[i,i+1] = -b for i = 0,...,n-2
 *
 *          **Algorithm Steps:**
 *          1. Cholesky decomposition: A = L*L' where L is lower triangular
 *          2. Forward substitution: L*u = y to find intermediate vector u
 *          3. Backward substitution: L'*x = u to find mean vector x = A^(-1)*y
 *          4. Generate random component: L'*s = z where z ~ N(0,I)
 *          5. Return r = x + s ~ N(A^(-1)*y, A^(-1))
 *
 *          The algorithm exploits the tridiagonal structure for O(n) complexity
 *          instead of O(n^3) for general matrices.
 *
 * @param r       Output vector [n] for sampled multivariate normal vector.
 * @param y       Right-hand side vector [n] of the system A*x = y.
 *                Determines the mean of the generated distribution.
 * @param a       Scalar parameter 'a' for precision matrix structure.
 * @param b       Scalar parameter 'b' for precision matrix structure.
 * @param n       Dimension of vectors/matrix (must be > 2).
 * @param add_a   Flag controlling last diagonal element:
 *                - 1: A[n-1,n-1] = a + b
 *                - 0: A[n-1,n-1] = b
 *
 * @complexity O(n) time, O(n) space
 * @memory Allocates 4n doubles: d[n], l[n-1], u[n], x[n], released via
 *         vmaxget()/vmaxset() before returning so per-iteration callers
 *         do not accumulate R_alloc stack memory across a long MCMC run.
 *
 * @note **Critical Assumption**: n > 2 required for algorithm stability.
 * @note Uses R's memory allocation (R_alloc), reclaimed on exit via vmaxset.
 * @note Numerical stability depends on condition number of precision matrix A.
 * @note Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 *
 * @warning Undefined behavior for n <= 2 (boundary checks removed for performance).
 * @warning No validation of matrix positive definiteness - may fail silently.
 * @warning Parameters a, b should ensure A is positive definite.
 *
 * @see rnorm, R_alloc
 * @since version 1.0
 *
 */
void generate_normal_vector(double       *r,
                            const double *y,
                            double        a,
                            double        b,
                            int           n,
                            int           add_a) {

  /* ========== Debug Validation ========== */
#ifdef DEBUG
  if (n <= 2) {
    error("generate_normal_vector: dimension n must be > 2, got %d", n);
  }
#endif

  /* ========== Matrix Parameters Setup ========== */
  double a11 = a + 2 * b;                                             // Standard diagonal element of A
  double ann = add_a == 1 ? a + b : b;                                // Last diagonal element of A
  int i;                                                              // Loop index for matrix elements

  /* ========== Memory Allocation ========== */
  /* The R_alloc stack is only unwound when the enclosing .Call returns, and
   * this function runs once per Gibbs iteration. Save the stack pointer here
   * and restore it on exit so the scratch buffers are reclaimed every call
   * instead of accumulating O(iterations * n) peak memory. */
  const void *vmax = vmaxget();
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
  r[n - 1] = rnorm(0, 1) / d[n - 1];                                  // r[n-1] = z[n-1]/L[n-1,n-1]

  // Remaining elements (i = n-2 to 0)
  for (i = n - 2; i >= 0; i--) {
    x[i] = (u[i] - l[i] * x[i + 1]) / d[i];                           // x[i] = (u[i] - L[i+1,i]*x[i+1])/L[i,i]
    r[i] = (rnorm(0, 1) - l[i] * r[i + 1]) / d[i];                    // r[i] = (z[i] - L[i+1,i]*r[i+1])/L[i,i]
    r[i + 1] += x[i + 1];                                             // Incrementally add x[i+1] to r[i+1]
  }

  r[0] += x[0];                                                       // Final addition to r[0]

  /* Release the scratch buffers (d, l, u, x); only r carries results out. */
  vmaxset(vmax);
}
