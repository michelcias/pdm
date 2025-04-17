#' Internal wrapper for the C ilogit function
#'
#' @keywords internal
ilogit <- function(x) {
  .Call("_pdm_ilogit", as.numeric(x))
}
