/**
 * @file utils.h
 * @brief Header file for utility functions in PDM package.
 * @author Michel H. Montoril
 * @date 2025-08-09
 * @version 1.0
 *
 * @details This header declares core utility functions including:
 *          - Inverse logit transformation (numerically stable)
 *          - Multivariate normal random vector generation with tridiagonal precision
 *          - R interface wrappers for core C functions with input validation
 */

#ifndef UTILS_H
#define UTILS_H

#include <R.h>
#include <Rinternals.h>

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
double ilogit(double x);

//----------------------------------------------------------------------

/**
 * @brief Draw a Gamma variate guaranteed to be strictly positive.
 *
 * @details Thin wrapper around R's rgamma that floors the draw at DBL_EPSILON
 *          (~2.2e-16). With small shape parameters (e.g., weakly informative
 *          precision priors such as Gamma(0.01, 0.01), or empty mixture
 *          components where the posterior shape reduces to the prior shape),
 *          the Gamma distribution places substantial mass so close to zero
 *          that rgamma underflows to a subnormal value or to exactly 0.0.
 *
 *          A zero (or subnormal) precision then propagates through the MCMC:
 *          1/prec and sqrt(1/prec) overflow to Inf, zero pivots appear in the
 *          Cholesky factorization of the state precision matrix, and squared
 *          innovations overflow, permanently poisoning the chain with Inf/NaN.
 *
 *          Flooring at DBL_EPSILON is statistically inert: a precision below
 *          machine epsilon corresponds to a variance above ~4.5e15, which is
 *          indistinguishable from a flat prior for any real dataset, while
 *          every downstream quantity (1/prec, sqrt(1/prec), squared
 *          innovations) remains comfortably finite. The guard also catches
 *          NaN produced by degenerate parameters (e.g., scale = 1/Inf after
 *          an upstream overflow), allowing the chain to recover instead of
 *          propagating NaN forever.
 *
 * @param shape  Gamma shape parameter (> 0).
 * @param scale  Gamma scale parameter (> 0). Note: scale = 1/rate.
 * @return       Gamma(shape, scale) draw, floored at DBL_EPSILON.
 *
 * @note Intended for sampling precision parameters (initialization draws from
 *       the prior and conjugate Gamma posterior updates).
 * @note Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 *
 * @see rgamma
 */
double rgamma_positive(double shape, double scale);

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
                            int           add_a);

#endif /* UTILS_H */
