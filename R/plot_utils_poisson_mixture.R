#' Shared plot dispatcher for the Poisson mixture family
#'
#' @description The body of `plot.poisson_mixture_locallevel()` and its two
#'   siblings. Every branch it dispatches to already takes the polynomial order
#'   from the object itself, so the three orders differ only in the class name
#'   used in error messages -- which is what this function takes as an argument
#'   instead of being written out three times.
#'
#' @param x A fitted object of one of the three `poisson_mixture_*` classes.
#' @param type Character, already matched by the calling method.
#' @param which Integer vector or `NULL`; meaning depends on `type`.
#' @param ask Logical or `NULL`; `NULL` means `interactive()` under
#'   `type = "all"`.
#' @param overlay_data Logical; overlay the observed counts on the weight plot.
#' @param ci Logical; draw credible bands where supported.
#' @param ci_level Numeric in (0, 1); credible level.
#' @param true_values Named list of true values, or `NULL`.
#' @param generator Character, the generator's name, used in the message that
#'   tells the user how to obtain the acceptance diagnostic.
#' @param ... Additional arguments passed to the plotting functions.
#'
#' @return `x`, invisibly.
#'
#' @keywords internal
#' @noRd
plot_poisson_mixture_dispatch <- function(x,
                                          type,
                                          which,
                                          ask,
                                          overlay_data,
                                          ci,
                                          ci_level,
                                          true_values,
                                          generator,
                                          ...) {

  # Check if acceptance proportions are available only when specifically requested
  if (type == "acceptance" && is.null(x$accept_prop)) {
    if (identical(attr(x, "link"), "probit")) {
      stop("Acceptance proportions are not available for the probit link, ",
           "which uses a data-augmentation Gibbs sampler with no ",
           "Metropolis-Hastings step. They are produced only for link = \"logit\".")
    }
    stop("Acceptance proportions are not available. ",
         "Re-run ", generator, "() with return_accept_prop = TRUE.")
  }

  if (is.null(ask)) {
    ask <- interactive() && type == "all"
  }

  # Extract target_acceptance with fallback for backward compatibility
  target_acc <- attr(x, "target_acceptance")
  if (is.null(target_acc)) {
    target_acc <- 0.44  # Default fallback for objects created before this feature
  }

  switch(type,
         all = {
           oldpar <- par(no.readonly = TRUE)
           on.exit(par(oldpar))
           if (ask) {
             oldask <- par(ask = TRUE)
             on.exit(par(oldask), add = TRUE)
           }
           plot_all_mixture_generic_base(x,
                                         ask = FALSE,
                                         ci = ci,
                                         ci_level = ci_level,
                                         overlay_data = overlay_data,
                                         obs_data = attr(x, "y"),
                                         true_values = true_values,
                                         ...)
           # Plot acceptance proportions only if available
           if (!is.null(x$accept_prop)) {
             plot_acceptance_proportions_base(x$accept_prop,
                                              target_acceptance = target_acc,
                                              ...)
           }
         },
         mcmc = plot_mcmc_diagnostics_generic(x,
                                              which = which,
                                              true_values = true_values,
                                              ...),
         params = plot_poisson_mixture_params_base(x$lambda_1,
                                                   x$lambda_2,
                                                   which = which,
                                                   true_values = true_values,
                                                   ...),
         states = plot_dynamic_states_generic_base(x,
                                                   which = which,
                                                   ci = ci,
                                                   ci_level = ci_level,
                                                   true_values = true_values,
                                                   ...),
         alpha = plot_mixture_weights_base(x$alpha,
                                           x$z,
                                           ci = ci,
                                           ci_level = ci_level,
                                           overlay_data = overlay_data,
                                           obs_data = attr(x, "y"),
                                           true_alpha = true_values$alpha,
                                           true_z = true_values$z,
                                           ...),
         acceptance = {
           if (!is.null(x$accept_prop)) {
             plot_acceptance_proportions_base(x$accept_prop,
                                              target_acceptance = target_acc,
                                              ...)
           }
         }
  )

  invisible(x)
}
