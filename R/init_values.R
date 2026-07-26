#' Resolve the starting values of a sampler
#'
#' Internal helper shared by every `mcmc_*` wrapper that accepts `init`. It
#' validates the user's `init` list and returns the complete starting state of
#' the chain: the value the user supplied wherever one was supplied, and a draw
#' from that parameter's own prior everywhere else.
#'
#' The draws happen here, in R, rather than in the C driver. That keeps the
#' package's standing contract -- the C layer only ever receives already-decided
#' finite values -- and it makes the starting state known to the wrapper, so
#' `record_provenance()` can record where the chain actually started.
#'
#' @section Draw order:
#' The parameters are drawn in exactly the order the C drivers used to draw them
#' themselves: every `theta_0k` in ascending `k`, then each precision in the
#' order given, with a Half-t auxiliary drawn immediately before its precision.
#' The RNG stream is therefore consumed exactly as before, and an `init = NULL`
#' run reproduces a pre-0.9-0 fit of the same seed bit for bit.
#'
#' @section The Half-t auxiliary:
#' A Half-t precision is represented by the inverse-gamma scale mixture of Wand
#' et al. (2011), which carries one auxiliary variable \eqn{b = 1/a} per
#' precision. It is a nuisance parameter and deliberately not user-settable, but
#' it must hold a value, because it enters the first precision update as the
#' prior rate. Two cases:
#'
#' - the precision is **drawn**: the pair comes from the prior as before,
#'   \eqn{b \sim \text{Gamma}(1/2, \text{scale} = A^2)} and then the precision
#'   given \eqn{b};
#' - the precision is **supplied**: \eqn{b} is drawn from its full conditional
#'   given that precision, so the starting pair is the coherent state the
#'   sampler would hold just after a Half-t precision step, rather than two
#'   unrelated draws.
#'
#' Under a Gamma prior the auxiliary is 0 and never read.
#'
#' @param init The user's `init` argument: `NULL`, or a named list whose names
#'   are a subset of `names(states)` and `names(precs)`.
#' @param states Named list of the initial-state parameters, in the C driver's
#'   draw order; each element is `list(mean = , prec = )`, the Normal prior.
#' @param precs Named list of the precision parameters, in the C driver's draw
#'   order; each element is a `resolve_prec_prior()` result.
#'
#' @return A list with two elements:
#'   \describe{
#'     \item{`values`}{Numeric vector for `.Call()`: the states in order, then
#'       the precisions in order, then one auxiliary per precision in the same
#'       order.}
#'     \item{`init`}{Named list of the user-settable starting values -- the
#'       states and precisions, not the auxiliaries -- for
#'       `record_provenance()`.}
#'   }
#'
#' @keywords internal
#' @noRd
resolve_init <- function(init, states, precs) {

  init  <- validate_init(init, c(names(states), names(precs)))
  start <- list()

  for (nm in names(states)) {
    s <- states[[nm]]
    start[[nm]] <- if (is.null(init[[nm]])) {
      rnorm(1L, s$mean, sqrt(1 / s$prec))
    } else {
      init_scalar(init[[nm]], nm, positive = FALSE)
    }
  }

  aux        <- numeric(length(precs))
  names(aux) <- names(precs)

  for (nm in names(precs)) {
    pr <- precs[[nm]]

    # `code` is the integer the C layer receives: 1 = Half-t, 0 = Gamma
    # (see resolve_prec_prior() in R/prec_prior.R).
    if (pr$code == 1L) {
      if (is.null(init[[nm]])) {
        aux[[nm]]   <- rgamma_floor(0.5, pr$scale^2)
        start[[nm]] <- rgamma_floor(0.5 * pr$df, 1 / (pr$df * aux[[nm]]))
      } else {
        start[[nm]] <- init_scalar(init[[nm]], nm, positive = TRUE)
        # b | W^-1 ~ Gamma((df + 1)/2, rate = df * W^-1 + 1/A^2), the same full
        # conditional generate_halft_aux() applies after every precision draw.
        aux[[nm]] <- rgamma_floor(
          0.5 * (pr$df + 1),
          1 / (pr$df * start[[nm]] + 1 / pr$scale^2)
        )
      }
    } else {
      start[[nm]] <- if (is.null(init[[nm]])) {
        rgamma_floor(pr$shape, 1 / pr$rate)
      } else {
        init_scalar(init[[nm]], nm, positive = TRUE)
      }
      aux[[nm]] <- 0
    }
  }

  list(values = c(unlist(start, use.names = FALSE), unname(aux)),
       init   = start)
}


#' Validate the shape and the names of an `init` list
#'
#' Names are checked against the model's own parameter set, and an unrecognised
#' one is an error rather than being ignored. Silently dropping a misspelled
#' name is the classic way for a starting value to have no effect at all while
#' the user believes it took.
#'
#' @param init The user's `init` argument.
#' @param known Character vector of the names this model accepts.
#'
#' @return `init` as a list (empty when it was `NULL`).
#'
#' @keywords internal
#' @noRd
validate_init <- function(init, known) {

  if (is.null(init)) return(list())
  if (!is.list(init)) {
    stop("`init` must be a named list of starting values, or NULL")
  }
  if (length(init) == 0L) return(list())

  nms <- names(init)
  if (is.null(nms) || !all(nzchar(nms))) {
    # The list-of-lists form is what `chains > 1` takes, and reaching here with
    # one means it was passed to a single-chain call. Say that, rather than
    # reporting a missing name and leaving the user to work it out.
    if (all(vapply(init, function(e) is.null(e) || is.list(e), logical(1L)))) {
      stop("`init` holds one list per chain, which is valid only when `chains > 1`")
    }
    stop("every element of `init` must be named; valid names for this model are: ",
         paste(known, collapse = ", "))
  }

  unknown <- setdiff(nms, known)
  if (length(unknown) > 0L) {
    stop("unknown name(s) in `init`: ", paste(unknown, collapse = ", "),
         ". Valid names for this model are: ", paste(known, collapse = ", "))
  }

  dups <- unique(nms[duplicated(nms)])
  if (length(dups) > 0L) {
    stop("`init` has duplicated name(s): ", paste(dups, collapse = ", "))
  }

  init
}


#' Validate one user-supplied starting value
#'
#' @param value The value as supplied.
#' @param nm Its name within `init`, used to build the error message.
#' @param positive Whether the parameter is constrained to be positive (every
#'   precision is).
#'
#' @return `value` as a length-one numeric.
#'
#' @keywords internal
#' @noRd
init_scalar <- function(value, nm, positive) {

  if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
    stop(sprintf("`init$%s` must be a single finite numeric value", nm))
  }
  if (positive && value <= 0) {
    stop(sprintf("`init$%s` must be a single positive finite numeric value", nm))
  }

  as.numeric(value)
}


#' R twin of the C `rgamma_positive()`
#'
#' Draws `Gamma(shape, scale)` and floors the result at `DBL_EPSILON`, exactly
#' as `rgamma_positive()` does in `src/utils.c`. The floor matters for the same
#' reason it does there: a zero or subnormal precision poisons the rate of the
#' next update, and a zero auxiliary poisons the precision that follows it.
#'
#' @param shape,scale Gamma parameters, in the shape-scale parameterization
#'   `stats::rgamma()` and the C `rgamma()` share.
#'
#' @return A single positive numeric.
#'
#' @keywords internal
#' @noRd
rgamma_floor <- function(shape, scale) {
  x <- rgamma(1L, shape = shape, scale = scale)
  # `!(x > floor)` is true for x <= floor AND for NaN, as in the C guard.
  if (isTRUE(x > .Machine$double.eps)) x else .Machine$double.eps
}


#' Split an `init` argument into one entry per chain
#'
#' @section Why a single list is refused:
#' Every chain drawing its own starting point from the priors is what makes the
#' Gelman-Rubin statistic meaningful -- the between-chain variance it compares
#' against the within-chain variance has to come from somewhere. One starting
#' point shared by every chain removes that dispersion, and the resulting R-hat
#' is optimistic without saying so. That is a wrong answer rather than an
#' inconvenience, so it is an error and not a warning.
#'
#' @param init The user's `init` argument: `NULL`, or a list of `chains`
#'   starting-value lists.
#' @param chains Integer >= 2, number of chains about to be run.
#'
#' @return A list of length `chains`, each element `NULL` or a named list.
#'
#' @keywords internal
#' @noRd
split_init_by_chain <- function(init, chains) {

  if (is.null(init)) return(vector("list", chains))
  if (!is.list(init)) {
    stop("`init` must be a list, or NULL")
  }

  per_chain <- length(init) > 0L &&
    all(vapply(init, function(e) is.null(e) || is.list(e), logical(1L)))
  if (!per_chain) {
    stop("with `chains > 1`, `init` must be a list of ", chains,
         " starting-value lists, one per chain: a single starting point shared ",
         "by every chain removes the dispersion the Gelman-Rubin statistic is ",
         "computed from")
  }
  if (length(init) != chains) {
    stop("`init` must hold one starting-value list per chain: got ",
         length(init), " for ", chains, " chains")
  }

  unname(init)
}
