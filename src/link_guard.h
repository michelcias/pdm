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
 *          **That paragraph describes the logit only. Do not read it as a
 *          statement about the probit.** @f$\Phi@f$ saturates far earlier:
 *          @f$\Phi(-8.14) = 2\times 10^{-16}@f$, so under the probit link the
 *          floor is reached at @f$|\theta| \approx 8.14@f$, deep inside the
 *          @f$\pm 36@f$ state clamp of `clamp_probit_state()`
 *          (`generate_alpha_binomial.c`). Over the whole band
 *          @f$8.14 < |\theta| < 36@f$ the bound is load-bearing, not inert, and
 *          it is nowhere near "essentially unchanged": @f$\Phi(-36)@f$ is
 *          @f$4.2\times 10^{-284}@f$ against a floor of @f$2\times 10^{-16}@f$.
 *          This is deliberate — it is precisely what keeps the probit
 *          likelihood evaluable in that band, and the two guards divide the
 *          work as the block above `clamp_probit_state()` sets out. What is not
 *          safe is to carry the logit's "inert" reading across to the probit.
 *
 *          One consequence for anyone diagnosing a probit fit: inside that band
 *          distinct theta values all map to the same floored alpha, so alpha
 *          draws tie where theta draws do not. Rank-based statistics see the
 *          ties; see `docs/multichain-rhat.md`, *R-hat on alpha is not R-hat on
 *          theta*.
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
