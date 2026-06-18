/**
 * @file hpdi.c
 * @brief Highest Posterior Density Interval (HPDI) computation in C.
 * @author Michel H. Montoril
 * @date 2026-06-18
 * @version 1.0
 *
 * @details Implements the shortest-interval HPDI estimator for MCMC output.
 *          The hot path is a per-column quicksort (R_qsort) followed by a single
 *          linear scan over candidate windows, avoiding the per-column R-level
 *          overhead and intermediate matrix allocation of an apply()-based
 *          implementation. Columns are processed independently and the input is
 *          never mutated (each column is copied into a scratch buffer).
 */

#include "hpdi.h"

#include <R_ext/Utils.h> /* R_qsort */
#include <string.h>      /* memcpy  */
#include <math.h>        /* floor   */

/**
 * @brief Locate the shortest window of fixed span in a sorted vector.
 *
 * @param sorted Ascending-sorted values of length \c n.
 * @param n      Number of values.
 * @param m      Index span between the lower and upper ends of each window
 *               (\c m = floor(prob * n)), with \c 1 <= m <= n - 1.
 * @param lower  Output: lower end of the shortest window.
 * @param upper  Output: upper end of the shortest window.
 */
static void shortest_window(const double *sorted, int n, int m,
                            double *lower, double *upper)
{
  int n_windows = n - m;
  double best = sorted[m] - sorted[0];
  int best_i = 0;
  for (int i = 1; i < n_windows; i++) {
    double w = sorted[i + m] - sorted[i];
    if (w < best) {
      best = w;
      best_i = i;
    }
  }
  *lower = sorted[best_i];
  *upper = sorted[best_i + m];
}

SEXP C_hpdi(SEXP data, SEXP prob)
{
  if (!Rf_isReal(data)) {
    Rf_error("'data' must be a numeric (double) vector or matrix.");
  }

  double p_level = Rf_asReal(prob);
  if (!R_FINITE(p_level) || p_level <= 0.0 || p_level >= 1.0) {
    Rf_error("'prob' must be a single value strictly between 0 and 1.");
  }

  SEXP dim = Rf_getAttrib(data, R_DimSymbol);
  int is_matrix = (dim != R_NilValue && Rf_length(dim) == 2);

  int n, p;
  if (is_matrix) {
    n = INTEGER(dim)[0];
    p = INTEGER(dim)[1];
  } else {
    n = Rf_length(data);
    p = 1;
  }

  if (n < 2) {
    Rf_error("Need at least 2 samples to compute an HPD interval.");
  }

  /* Index span of each candidate window. Clamp to keep at least one window
     and a non-degenerate interval for extreme prob values. */
  int m = (int) floor(p_level * (double) n);
  if (m < 1) m = 1;
  if (m > n - 1) m = n - 1;

  const double *x = REAL(data);

  /* Scratch buffer for one column; freed automatically on .Call return. */
  double *buf = (double *) R_alloc((size_t) n, sizeof(double));

  if (is_matrix) {
    SEXP res = PROTECT(Rf_allocMatrix(REALSXP, p, 2));
    double *out = REAL(res);
    double *lower = out;       /* column 0 */
    double *upper = out + p;   /* column 1 */

    for (int j = 0; j < p; j++) {
      memcpy(buf, x + (size_t) j * n, (size_t) n * sizeof(double));
      R_qsort(buf, 1, n); /* 1-indexed: sorts buf[0..n-1] ascending */
      shortest_window(buf, n, m, &lower[j], &upper[j]);
    }

    /* Column names: lower, upper (no row names). */
    SEXP dn = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(dn, 0, R_NilValue);
    SEXP cn = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(cn, 0, Rf_mkChar("lower"));
    SET_STRING_ELT(cn, 1, Rf_mkChar("upper"));
    SET_VECTOR_ELT(dn, 1, cn);
    Rf_setAttrib(res, R_DimNamesSymbol, dn);

    UNPROTECT(3);
    return res;
  } else {
    memcpy(buf, x, (size_t) n * sizeof(double));
    R_qsort(buf, 1, n);

    SEXP res = PROTECT(Rf_allocVector(REALSXP, 2));
    double *out = REAL(res);
    shortest_window(buf, n, m, &out[0], &out[1]);

    SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(nm, 0, Rf_mkChar("lower"));
    SET_STRING_ELT(nm, 1, Rf_mkChar("upper"));
    Rf_setAttrib(res, R_NamesSymbol, nm);

    UNPROTECT(2);
    return res;
  }
}
