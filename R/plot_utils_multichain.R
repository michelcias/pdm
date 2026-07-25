#' Four-panel diagnostic page for one parameter across several chains
#'
#' @description Multi-chain sibling of `plot_param_diagnostics_base()`. Every
#'   panel draws one series per chain instead of one for the whole sample, which
#'   is the point of running several chains: disagreement between them is what
#'   you are looking for, and pooling hides it.
#'
#' @details
#' The two functions are kept separate rather than merged behind a flag. They
#' differ in all four panels — one draws a single series, the other loops over
#' chains — so a merged version would be two implementations interleaved by
#' `if (multi)` rather than shared code, in a function that 15 plot methods
#' already depend on.
#'
#' Panels:
#' \enumerate{
#'   \item \strong{Trace}, one line per chain. Chains that have mixed overlap
#'     and are indistinguishable; one wandering off is immediately visible.
#'   \item \strong{Autocorrelation}, the mean of the per-chain ACFs. Pooling the
#'     draws first would splice unrelated draws at each chain boundary and
#'     understate the autocorrelation.
#'   \item \strong{Posterior density}, one curve per chain. Curves that do not
#'     coincide are the same failure the folded \eqn{\hat{R}} reports
#'     numerically.
#'   \item \strong{Running mean}, one line per chain, all converging to the
#'     pooled mean when the sampler has settled.
#' }
#'
#' The header carries \eqn{\hat{R}} and the bulk ESS so the numbers and the
#' picture are read together.
#'
#' @param draws Numeric matrix of draws, iterations in rows and chains in
#'   columns.
#' @param param_label Expression used to label the axes.
#' @param param_name Quoted expression naming the parameter (the `name` field of
#'   a `get_param_config()` entry), rendered in the page header.
#' @param true_value Optional numeric; drawn as a dashed black reference line.
#' @param ... Additional arguments (currently unused).
#'
#' @return `NULL`, invisibly. Called for the plots it produces.
#'
#' @keywords internal
#' @noRd
plot_param_diagnostics_multi <- function(draws,
                                         param_label,
                                         param_name,
                                         true_value = NULL,
                                         ...) {

  n_draw <- nrow(draws)
  n_ch   <- ncol(draws)
  pooled <- as.vector(draws)
  cols   <- grDevices::hcl.colors(max(n_ch, 2L), "Dark 3")[seq_len(n_ch)]

  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  par(mfrow = c(2, 2),
      mar   = c(4, 4, 3, 1),
      oma   = c(0, 0, 3, 0),
      mgp   = c(2.5, 1, 0))

  chain_legend <- function(pos = "topright") {
    legend(pos,
           legend = paste("Chain", seq_len(n_ch)),
           col    = cols,
           lwd    = 1.2,
           horiz  = n_ch <= 4,
           bty    = "n",
           cex    = 0.7)
  }

  # =========================================================================
  # Panel 1: Trace, one line per chain
  # =========================================================================

  range_trace <- range(pooled)
  range_trace[2] <- range_trace[2] + 0.3 * diff(range_trace)

  plot(NA,
       xlim = c(1, n_draw),
       ylim = range_trace,
       xlab = "Iteration",
       ylab = param_label,
       main = "Trace by chain")
  for (m in seq_len(n_ch)) {
    lines(seq_len(n_draw), draws[, m], col = cols[m], lwd = 0.8)
  }
  if (!is.null(true_value)) {
    segments(1, true_value, n_draw, true_value, col = "black", lwd = 2, lty = 2)
  }
  chain_legend()
  grid()

  # =========================================================================
  # Panel 2: Autocorrelation, averaged over chains
  # =========================================================================

  lag_max  <- min(30L, n_draw - 1L)
  acf_each <- vapply(seq_len(n_ch), function(m) {
    drop(acf(draws[, m], lag.max = lag_max, plot = FALSE)$acf)
  }, numeric(lag_max + 1L))
  acf_mean <- rowMeans(acf_each)

  # Neutral grey on purpose: this series belongs to no single chain, and using
  # a chain colour here would read as "chain 1".
  plot(0:lag_max, acf_mean,
       type = "h",
       lwd  = 2,
       col  = "gray30",
       ylim = range(0, 1, acf_mean),
       xlab = "Lag",
       ylab = "ACF",
       main = "Autocorrelation (mean over chains)")
  abline(h = 0, col = "gray50")
  grid()

  # =========================================================================
  # Panel 3: Posterior density, one curve per chain
  # =========================================================================

  dens_each <- lapply(seq_len(n_ch), function(m) density(draws[, m]))
  xlim <- range(vapply(dens_each, function(d) range(d$x), numeric(2L)))
  ylim <- c(0, max(vapply(dens_each, function(d) max(d$y), numeric(1L))) * 1.25)

  plot(NA,
       xlim = xlim,
       ylim = ylim,
       xlab = param_label,
       ylab = "Density",
       main = "Posterior density by chain")
  for (m in seq_len(n_ch)) {
    lines(dens_each[[m]], col = cols[m], lwd = 1.5)
  }
  if (!is.null(true_value)) {
    segments(true_value, 0, true_value, ylim[2L] / 1.25,
             col = "black", lwd = 2, lty = 2)
  }
  chain_legend()
  grid()

  # =========================================================================
  # Panel 4: Running mean, one line per chain
  # =========================================================================

  running <- vapply(seq_len(n_ch), function(m) {
    cumsum(draws[, m]) / seq_len(n_draw)
  }, numeric(n_draw))

  range_run <- range(running, mean(pooled))
  range_run[2] <- range_run[2] + 0.3 * diff(range_run)

  plot(NA,
       xlim = c(1, n_draw),
       ylim = range_run,
       xlab = "Iteration",
       ylab = param_label,
       main = "Running mean by chain")
  for (m in seq_len(n_ch)) {
    lines(seq_len(n_draw), running[, m], col = cols[m], lwd = 1.2)
  }
  segments(1, mean(pooled), n_draw, mean(pooled),
           col = "gray30", lwd = 2, lty = 3)
  if (!is.null(true_value)) {
    segments(1, true_value, n_draw, true_value, col = "black", lwd = 2, lty = 2)
  }
  grid()

  # =========================================================================
  # Page header: the numbers that go with the picture
  # =========================================================================

  rhat <- rhat_rank_normalized(draws)
  bulk <- ess_bulk(draws)

  stats_txt <- if (is.na(rhat)) {
    sprintf("  -  %d chains x %d draws", n_ch, n_draw)
  } else {
    sprintf("  -  %d chains x %d draws   |   R-hat = %.4f   |   ESS(bulk) = %.0f",
            n_ch, n_draw, rhat, bulk)
  }

  # `param_name` is a quoted expression (e.g. quote(V^{-1})), so the header is
  # assembled with bquote() rather than sprintf() to render it as the axes do.
  mtext(bquote(.(param_name) * .(stats_txt)),
        outer = TRUE, cex = 1.05, font = 2)

  invisible(NULL)
}
