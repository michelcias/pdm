/**
 * @file link_guard.h
 * @brief Numerical guard on link-transformed probabilities
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.0
 *
 * @details Provides the bound applied to every probability produced by an
 *          inverse link, so that neither a likelihood evaluation nor a reported
 *          success probability ever sees an exact 0 or 1.
 *
 *          **Why the bounds are where they are:**
 *          The constants track the numerical saturation point of the logit
 *          transform: @f$\mathrm{ilogit}(-36) = 2.3\times 10^{-16}@f$ and
 *          @f$\mathrm{ilogit}(36) = 1 - 2.3\times 10^{-16}@f$. Beyond
 *          @f$|\theta| \approx 36@f$ the transform is already numerically
 *          indistinguishable from the boundary, so constraining the probability
 *          there leaves alpha essentially unchanged while removing the
 *          exact-boundary hazard. The lower floor is marginally wider
 *          (@f$2\times 10^{-16}@f$) for symmetry with the probit guard.
 *
 *          Unlike a state clamp, this guard never distorts the sampled latent
 *          state: theta is stored exactly as drawn, and only the derived
 *          probability is protected.
 *
 *          **Why a header-only helper:**
 *          Identical copies of these definitions previously lived in
 *          `cwmh_binomial.c` and `generate_alpha_binomial.c`. Two copies of a
 *          numerical guard is a latent inconsistency: a correction to one would
 *          not reach the other. Following the pattern of
 *          `prec_prior_dispatch.h`, this file intentionally has no `.c`
 *          companion — each translation unit that includes it gets its own
 *          inline copy of the helpers it calls, and unused ones emit no
 *          `-Wunused-function` warning.
 */

#ifndef LINK_GUARD_H
#define LINK_GUARD_H

#include "utils.h"          /* ilogit */

#define LINK_ALPHA_MIN 2e-16
#define LINK_ALPHA_MAX (1.0 - 2.3e-16)

/**
 * @brief Constrain a probability away from the exact boundaries.
 *
 * @param p   Probability to bound.
 * @return p clamped to [LINK_ALPHA_MIN, LINK_ALPHA_MAX].
 */
static inline double clamp_link_alpha(double p) {
  if (p < LINK_ALPHA_MIN) return LINK_ALPHA_MIN;
  if (p > LINK_ALPHA_MAX) return LINK_ALPHA_MAX;
  return p;
}

/**
 * @brief Inverse-logit transform with the probability guard applied.
 *
 * @details Computes alpha = ilogit(theta) and constrains it to
 *          [LINK_ALPHA_MIN, LINK_ALPHA_MAX]. Invoked at every point where the
 *          logit link g(theta) is evaluated.
 *
 * @param theta   Latent state value.
 * @return Guarded success probability in [LINK_ALPHA_MIN, LINK_ALPHA_MAX].
 */
static inline double ilogit_guarded(double theta) {
  return clamp_link_alpha(ilogit(theta));
}

#endif /* LINK_GUARD_H */
