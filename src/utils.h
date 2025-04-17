#ifndef UTILS_H
#define UTILS_H

#include <R.h>
#include <Rinternals.h>

/**
 * Computes the inverse logit (logistic) function of the input value.
 *
 * The inverse logit is defined as 1/(1 + exp(-x)) or equivalently exp(x)/(1 + exp(x)).
 * This implementation is numerically stable using Rf_log1pexp().
 *
 * @param x  Input value.
 * @return   The result of the inverse logit transformation, in the interval (0, 1).
 */
double ilogit(double x);

/**
 * Auxiliary function to generate a multivariate normal random vector
 * with precision matrix A, where A is a tridiagonal matrix with specific structure.
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
                            int add_a);

/**
 * R interface wrapper for the ilogit function.
 *
 * Takes a numeric SEXP object from R, applies the ilogit transformation to its first element,
 * and returns the result as a scalar real SEXP.
 *
 * @param x  Numeric input from R (only the first element is used).
 * @return   Scalar real containing the result of ilogit(x).
 */
SEXP C_ILogit(SEXP x);

#endif // UTILS_H