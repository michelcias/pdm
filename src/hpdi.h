/**
 * @file hpdi.h
 * @brief Header file for the Highest Posterior Density Interval (HPDI) routine.
 * @author Michel H. Montoril
 * @date 2026-06-18
 * @version 1.0
 *
 * @details Declares the C entry point that computes the shortest contiguous
 *          interval containing a given probability mass of a posterior sample.
 *          Accepts either a numeric vector (a single chain) or a numeric matrix
 *          whose columns are independent chains (e.g. one column per time point
 *          of a latent state trajectory).
 */

#ifndef HPDI_H
#define HPDI_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Compute the Highest Posterior Density Interval (HPDI) of MCMC samples.
 *
 * @details For a vector of \c n samples, the HPDI at level \c prob is the
 *          shortest interval \c [L, U] that contains a proportion \c prob of the
 *          draws. It is obtained by sorting the sample and, over all windows of
 *          fixed span \c m = floor(prob * n), selecting the one with minimum
 *          width \c sorted[i + m] - sorted[i]. For a matrix input the same
 *          computation is applied independently to each column.
 *
 *          Unlike an equal-tailed (quantile) interval, the HPDI is the shortest
 *          contiguous interval and may be asymmetric for skewed posteriors.
 *
 * @param data A REALSXP vector (length \c n) or matrix (\c n rows by \c p
 *             columns). Rows are MCMC samples; columns are independent chains.
 *             Values are assumed finite (no \c NA / \c NaN).
 * @param prob A scalar probability in the open interval (0, 1).
 *
 * @return For a vector input, a named numeric vector of length 2
 *         (\c lower, \c upper). For a matrix input, a \c p by 2 numeric matrix
 *         with column names \c lower and \c upper, one row per input column.
 *
 * @note The input is not modified; each column is copied into a scratch buffer
 *       before sorting.
 */
SEXP C_hpdi(SEXP data, SEXP prob);

#endif /* HPDI_H */
