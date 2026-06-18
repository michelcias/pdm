# =============================================================================
# Benchmark and validation for hpdi()
# =============================================================================
#
#' @title Benchmark and correctness check for the HPDI implementation
#' @author Michel H. Montoril
#' @date 2026-06-18
#' @description
#' Compares three implementations of the Highest Posterior Density Interval:
#'   1. hpdi_orig  - the original R version (apply + full intermediate matrix)
#'   2. hpdi_R_opt - an optimized pure-R version (no apply, no intermediate)
#'   3. hpdi       - the package C implementation (.Call _pdm_C_hpdi)
#'
#' It (a) asserts that all three agree on vector and matrix inputs and
#' (b) times them at realistic MCMC sizes (n samples x p time points).
#'
#' Run with the package loaded, e.g.:
#'   devtools::load_all(); source("inst/scripts/benchmark_hpdi.R")
# =============================================================================

# ---- Reference implementation 1: original ----------------------------------
hpdi_orig <- function(data, prob = 0.9, n, p) {
  if (is.matrix(data)) {
    dat.ord <- apply(X = data, MARGIN = 2, FUN = sort.int, method = "quick")
    idx <- n - floor(x = prob * n)
    upp.vals <- dat.ord[-seq_len(length.out = n - idx), ]
    low.vals <- dat.ord[seq_len(length.out = idx), ]
    idx.opt <- apply(X = upp.vals - low.vals, MARGIN = 2, FUN = which.min)
    low.vals.opt <- low.vals[cbind(idx.opt, seq_len(length.out = p))]
    upp.vals.opt <- upp.vals[cbind(idx.opt, seq_len(length.out = p))]
    interv <- cbind(lower = low.vals.opt, upper = upp.vals.opt)
  } else {
    dat.ord <- sort.int(x = data, method = "quick")
    idx <- n - floor(x = prob * n)
    upp.vals <- dat.ord[-seq_len(length.out = n - idx)]
    low.vals <- dat.ord[seq_len(length.out = idx)]
    idx.opt <- which.min(x = upp.vals - low.vals)
    low.vals.opt <- low.vals[idx.opt]
    upp.vals.opt <- upp.vals[idx.opt]
    interv <- c(lower = low.vals.opt, upper = upp.vals.opt)
  }
  return(interv)
}

# ---- Reference implementation 2: optimized pure R --------------------------
hpdi_R_opt <- function(data, prob = 0.9) {
  one <- function(col) {
    n <- length(col)
    s <- sort.int(col, method = "quick")
    m <- floor(prob * n)
    if (m < 1L) m <- 1L
    if (m > n - 1L) m <- n - 1L
    nw <- n - m
    i <- which.min(s[(1L + m):(nw + m)] - s[1L:nw])
    c(lower = s[i], upper = s[i + m])
  }
  if (is.matrix(data)) {
    res <- vapply(seq_len(ncol(data)), function(j) one(data[, j]),
                  numeric(2))
    t(res)  # p x 2 with colnames lower/upper
  } else {
    one(data)
  }
}

# ---- Correctness checks ----------------------------------------------------
set.seed(123)
stopifnot(requireNamespace("pdm", quietly = TRUE) || exists("hpdi"))

check <- function(label, a, b, tol = 1e-10) {
  ok <- isTRUE(all.equal(unname(as.matrix(a)), unname(as.matrix(b)),
                         tolerance = tol))
  cat(sprintf("  %-40s %s\n", label, if (ok) "OK" else "*** MISMATCH ***"))
  if (!ok) {
    cat("    a:\n"); print(a); cat("    b:\n"); print(b)
  }
  invisible(ok)
}

cat("Correctness:\n")
for (prob in c(0.5, 0.9, 0.95)) {
  v <- rgamma(2000, shape = 2, rate = 1)
  M <- matrix(rnorm(2000 * 50), nrow = 2000)

  check(sprintf("vector, prob=%.2f (orig vs C)", prob),
        hpdi_orig(v, prob, n = length(v)), hpdi(v, prob))
  check(sprintf("vector, prob=%.2f (Ropt vs C)", prob),
        hpdi_R_opt(v, prob), hpdi(v, prob))
  check(sprintf("matrix, prob=%.2f (orig vs C)", prob),
        hpdi_orig(M, prob, n = nrow(M), p = ncol(M)), hpdi(M, prob))
  check(sprintf("matrix, prob=%.2f (Ropt vs C)", prob),
        hpdi_R_opt(M, prob), hpdi(M, prob))
}

# ---- Timing ----------------------------------------------------------------
cat("\nTiming (n = 2000 samples, p = 1000 time points, prob = 0.9):\n")
n <- 2000L; p <- 1000L
M <- matrix(rnorm(n * p), nrow = n, ncol = p)

bench <- function(expr, label, times = 20L) {
  expr <- substitute(expr)
  t <- replicate(times, system.time(eval(expr, parent.frame()))[["elapsed"]])
  cat(sprintf("  %-28s median %.4f s  (min %.4f)\n",
              label, median(t), min(t)))
}

bench(hpdi_orig(M, 0.9, n = n, p = p), "original (apply)")
bench(hpdi_R_opt(M, 0.9),              "optimized R")
bench(hpdi(M, 0.9),                    "C (.Call)")

cat("\nDone.\n")
