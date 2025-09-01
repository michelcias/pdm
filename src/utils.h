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
 * @param r       Output array (vectorized B x n matrix). The sampled n-variate normal
 *                vector will be stored at row `iter` (r[iter*n:(iter+1)*n-1]).
 * @param y       Right-hand side vector (size n) of the system A*x = y.
 *                Determines the mean of the generated distribution.
 * @param a       Scalar parameter 'a' for precision matrix structure.
 * @param b       Scalar parameter 'b' for precision matrix structure.
 * @param n       Dimension of vectors/matrix (must be > 2).
 * @param iter    Row index (0-based) in output matrix `r` for storage.
 * @param add_a   Flag controlling last diagonal element:
 *                - 1: A[n-1,n-1] = a + b
 *                - 0: A[n-1,n-1] = b
 *
 * @complexity O(n) time, O(n) space
 * @memory Allocates 4n doubles: d[n], l[n-1], u[n], x[n]
 *
 * @note **Critical Assumption**: n > 2 required for algorithm stability.
 * @note Uses R's memory allocation (R_alloc) - automatically garbage collected.
 * @note Thread-safe if different threads use different `iter` values.
 * @note Numerical stability depends on condition number of precision matrix A.
 *
 * @warning Undefined behavior for n <= 2 (boundary checks removed for performance).
 * @warning No validation of matrix positive definiteness - may fail silently.
 * @warning Parameters a, b should ensure A is positive definite.
 *
 * @see rnorm, R_alloc
 * @since version 1.0
 *
 */
void generate_normal_vector(double *r,
                            double *y,
                            double a,
                            double b,
                            int n,
                            int iter,
                            int add_a);

#endif /* UTILS_H */
