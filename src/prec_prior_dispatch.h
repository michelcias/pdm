/**
 * @file prec_prior_dispatch.h
 * @brief Shared Gamma/Half-t prior dispatch for innovation & observation precisions
 * @author Michel H. Montoril
 * @date 2026-07-22
 * @version 1.0
 *
 * @details Every Gibbs driver in the package samples one or more precision
 *          parameters (innovation precisions 1/W_k and, for Gaussian models, the
 *          observation precision 1/V). Each precision may carry one of two
 *          priors, chosen independently:
 *            - a conjugate Gamma prior on the precision (the historical default), or
 *            - a Half-t prior on the corresponding standard deviation
 *              sqrt(W_k) ~ Half-t(df, A) (Gelman, 2006), represented by the
 *              inverse-gamma scale mixture of Wand et al. (2011) / Huang & Wand
 *              (2013) so that every full conditional stays closed-form.
 *
 *          This header collects the machinery that used to live inside
 *          `mcmc_normal_locallevel.c` so that all drivers reuse a single
 *          implementation:
 *            - the prior-kind integer codes shared with the R wrappers,
 *            - the `prec_prior_t` hyperparameter bundle,
 *            - one function-pointer typedef per precision-sampler kind
 *              (terminal `theta_p`, intermediate `theta_k`, observation `data`),
 *            - the six `static inline` step functions (each sampler kind × prior
 *              kind), and
 *            - `pdm_init_prec_prior`, the one-time draw of the starting precision.
 *
 *          **Design contract (see docs/half-t-prior.md):**
 *          The prior decision must NOT happen inside the MCMC loop. Each driver
 *          resolves a function pointer per precision once, before the Gibbs loop
 *          (a plain ternary on the integer code), and the per-iteration hot path
 *          calls through the pointer without ever re-testing the prior kind. All
 *          validation and the `"halfcauchy"` -> Half-t(df = 1) alias
 *          normalisation happen in R (`resolve_prec_prior()`); the C layer only
 *          ever sees an already-decided integer code plus finite hyperparameters.
 *
 *          The Half-t step reuses the existing conjugate Gamma samplers with
 *          shape `df/2` and prior rate `df * b`, where `b = 1/a` is the auxiliary
 *          variable refreshed in place after each precision draw via
 *          ::generate_halft_aux. `df = 1` is the Half-Cauchy special case.
 *
 *          The step functions are `static inline`: each translation unit that
 *          includes this header gets its own copy of only the ones it calls, and
 *          the unused ones emit no `-Wunused-function` warning under `-Wall`.
 *
 * @see conditional_precision.h
 * @see generate_halft_aux
 */

#ifndef PREC_PRIOR_DISPATCH_H
#define PREC_PRIOR_DISPATCH_H

#include "conditional_precision.h"  /* generate_precision_*, generate_halft_aux */
#include "utils.h"                  /* rgamma_positive */

/* Prior-kind codes shared with the R wrappers (see resolve_prec_prior() in
 * R/prec_prior.R). The C layer receives one of these already-decided codes. */
#define PDM_PREC_PRIOR_GAMMA 0
#define PDM_PREC_PRIOR_HALFT 1

/* Hyperparameters for one precision prior. Only the fields relevant to the
 * selected kind are read by the corresponding step function. */
typedef struct {
  double shape;    /* Gamma prior shape nu   (Gamma kind)  */
  double rate;     /* Gamma prior rate  eta  (Gamma kind)  */
  double df;       /* Half-t degrees of freedom nu (Half-t kind) */
  double hc_scale; /* Half-t scale A         (Half-t kind) */
} prec_prior_t;

/* Step signatures — one per precision-sampler kind. `aux` points to the Half-t
 * auxiliary b = 1/a, updated in place by the Half-t steps and ignored by the
 * Gamma steps. */
typedef double (*prec_thetap_step_t)(double theta_0p, const double *theta_p,
                                     int n, const prec_prior_t *pr, double *aux);
typedef double (*prec_thetak_step_t)(double theta_0k, double theta_0kp1,
                                     const double *theta_k, const double *theta_kp1,
                                     int n, const prec_prior_t *pr, double *aux);
typedef double (*prec_data_step_t)(const double *y, const double *theta_1,
                                   int n, const prec_prior_t *pr, double *aux);

/* --- Terminal innovation precision W_p^{-1} (random-walk boundary component) --- */
static inline double step_prec_thetap_gamma(double theta_0p, const double *theta_p,
                                            int n, const prec_prior_t *pr,
                                            double *aux) {
  (void) aux;  /* Gamma prior carries no auxiliary variable */
  return generate_precision_theta_p(theta_0p, theta_p, pr->shape, pr->rate, n);
}

static inline double step_prec_thetap_halft(double theta_0p, const double *theta_p,
                                            int n, const prec_prior_t *pr,
                                            double *aux) {
  /* Precision: conjugate Gamma sampler with shape df/2 and prior rate df * b. */
  double prec = generate_precision_theta_p(theta_0p, theta_p,
                                           0.5 * pr->df, pr->df * (*aux), n);
  /* Auxiliary refresh: b | prec ~ Gamma((df+1)/2, rate = df*prec + 1/A^2). */
  *aux = generate_halft_aux(prec, pr->hc_scale, pr->df);
  return prec;
}

/* --- Intermediate innovation precision W_k^{-1} (k < p) --- */
static inline double step_prec_thetak_gamma(double theta_0k, double theta_0kp1,
                                            const double *theta_k,
                                            const double *theta_kp1,
                                            int n, const prec_prior_t *pr,
                                            double *aux) {
  (void) aux;
  return generate_precision_theta_k(theta_0k, theta_0kp1, theta_k, theta_kp1,
                                    pr->shape, pr->rate, n);
}

static inline double step_prec_thetak_halft(double theta_0k, double theta_0kp1,
                                            const double *theta_k,
                                            const double *theta_kp1,
                                            int n, const prec_prior_t *pr,
                                            double *aux) {
  double prec = generate_precision_theta_k(theta_0k, theta_0kp1,
                                           theta_k, theta_kp1,
                                           0.5 * pr->df, pr->df * (*aux), n);
  *aux = generate_halft_aux(prec, pr->hc_scale, pr->df);
  return prec;
}

/* --- Observation precision V^{-1} --- */
static inline double step_prec_data_gamma(const double *y, const double *theta_1,
                                         int n, const prec_prior_t *pr,
                                         double *aux) {
  (void) aux;
  return generate_precision_data(y, theta_1, pr->shape, pr->rate, n);
}

static inline double step_prec_data_halft(const double *y, const double *theta_1,
                                         int n, const prec_prior_t *pr,
                                         double *aux) {
  double prec = generate_precision_data(y, theta_1,
                                        0.5 * pr->df, pr->df * (*aux), n);
  *aux = generate_halft_aux(prec, pr->hc_scale, pr->df);
  return prec;
}

/**
 * @brief Draw a precision from its Gamma/Half-t full conditional given
 *        precomputed sufficient statistics.
 *
 * @details For precisions whose posterior is assembled from a running count and
 *          sum-of-squares (e.g. the mixture component precisions phi_k, where
 *          `count` = #observations in the component and `sumsq` =
 *          Sum (y_t - mu_k)^2), rather than from a state trajectory. The
 *          posterior is Gamma(shape, rate) with
 *            - Gamma prior:  shape = pr->shape + count/2, rate = pr->rate + sumsq/2;
 *            - Half-t prior: shape = df/2 + count/2,      rate = df*aux + sumsq/2.
 *          The Half-t auxiliary is **not** refreshed here — the caller controls
 *          the refresh ordering (see ::pdm_refresh_halft_aux), which matters when
 *          a label-switch swap can reassign precisions between component slots.
 *
 * @param kind   Prior-kind code (::PDM_PREC_PRIOR_GAMMA or ::PDM_PREC_PRIOR_HALFT).
 * @param pr     Hyperparameters for the selected kind.
 * @param count  Sufficient statistic: (weighted) number of observations.
 * @param sumsq  Sufficient statistic: sum of squared deviations.
 * @param aux    Current Half-t auxiliary b = 1/a (read-only; ignored for Gamma).
 * @return Sampled precision (> 0), floored via ::rgamma_positive.
 */
static inline double pdm_draw_prec_suffstat(int kind, const prec_prior_t *pr,
                                            double count, double sumsq,
                                            double aux) {
  double shape, rate;
  if (kind == PDM_PREC_PRIOR_HALFT) {
    shape = 0.5 * pr->df + 0.5 * count;
    rate  = pr->df * aux + 0.5 * sumsq;
  } else {
    shape = pr->shape + 0.5 * count;
    rate  = pr->rate  + 0.5 * sumsq;
  }
  return rgamma_positive(shape, 1.0 / rate);
}

/**
 * @brief Refresh a Half-t auxiliary from a freshly drawn precision (no-op for Gamma).
 *
 * @details Draws b | prec ~ Gamma((df+1)/2, rate = df*prec + 1/A^2) via
 *          ::generate_halft_aux. Kept separate from ::pdm_draw_prec_suffstat so
 *          the caller can refresh **after** a mixture label-switch, keeping each
 *          auxiliary paired with the precision that ends up in its slot under
 *          that slot's own prior.
 *
 * @param kind  Prior-kind code; the refresh is skipped unless Half-t.
 * @param pr    Hyperparameters (Half-t df and scale A are read).
 * @param prec  Current precision (> 0) to condition on.
 * @param aux   Out-parameter receiving the refreshed auxiliary; untouched for Gamma.
 */
static inline void pdm_refresh_halft_aux(int kind, const prec_prior_t *pr,
                                         double prec, double *aux) {
  if (kind == PDM_PREC_PRIOR_HALFT) {
    *aux = generate_halft_aux(prec, pr->hc_scale, pr->df);
  }
}

/**
 * @brief Draw the starting precision from the chosen prior (once, before the loop).
 *
 * @details Under the Gamma prior the initial precision is drawn from the prior
 *          directly. Under the Half-t prior it is drawn through the scale-mixture
 *          representation: b ~ Gamma(1/2, scale = A^2), then
 *          prec ~ Gamma(df/2, rate = df * b). This mirrors exactly the branches
 *          that used to be inlined in each driver's initialisation block, so the
 *          RNG stream is unchanged.
 *
 * @param kind  Prior-kind code (::PDM_PREC_PRIOR_GAMMA or ::PDM_PREC_PRIOR_HALFT).
 * @param pr    Hyperparameters for the selected kind.
 * @param aux   Out-parameter receiving the initial Half-t auxiliary b = 1/a;
 *              set to 0 under the Gamma prior (where it is never read).
 * @return The starting precision (> 0), floored via ::rgamma_positive.
 *
 * @note Requires GetRNGstate()/PutRNGstate() bracket in the calling function.
 */
static inline double pdm_init_prec_prior(int kind, const prec_prior_t *pr,
                                         double *aux) {
  if (kind == PDM_PREC_PRIOR_HALFT) {
    *aux = rgamma_positive(0.5, pr->hc_scale * pr->hc_scale);
    return rgamma_positive(0.5 * pr->df, 1.0 / (pr->df * (*aux)));
  }
  *aux = 0.0;
  return rgamma_positive(pr->shape, 1.0 / pr->rate);
}

#endif /* PREC_PRIOR_DISPATCH_H */
