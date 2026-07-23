# =====================================================================
# .Rprofile  --  startup for the pdm package
#
# Sourced by every R session started from the project root. Order matters:
# renv must activate FIRST so the project-local library is on the path.
#
# Everything after that runs ONLY in interactive sessions and is wrapped so
# it can never break startup. This deliberately excludes `R CMD build/check`,
# Rscript calls, and CI -- none of which should trigger a git pull or print
# a banner.
#
# NOTE: no library() calls here on purpose. Loading packages at startup
# fights renv and hurts reproducibility.
#
# To disable the automatic pull for a session:  set PDM_NO_AUTOPULL=1
# (e.g. `PDM_NO_AUTOPULL=1 R`, or Sys.setenv(PDM_NO_AUTOPULL = 1)).
# =====================================================================

# --- 1. Activate renv (must come first) ------------------------------
source("renv/activate.R")

# --- 2. Interactive-only conveniences --------------------------------
if (interactive()) local({

  ## small helper: run git, capture stdout, never error out
  git <- function(args) tryCatch(
    suppressWarnings(system2("git", args, stdout = TRUE, stderr = FALSE)),
    error = function(e) character()
  )

  in_worktree <- identical(git(c("rev-parse", "--is-inside-work-tree")), "true")

  ## -- 2a. Automatic 'git pull' on startup --------------------------
  ## Skips when: opted out via PDM_NO_AUTOPULL, not a git work tree,
  ## the tree has uncommitted changes, or there is no upstream branch.
  if (in_worktree && !nzchar(Sys.getenv("PDM_NO_AUTOPULL"))) {
    dirty        <- length(git(c("status", "--porcelain"))) > 0L
    has_upstream <- length(git(c("rev-parse", "--abbrev-ref",
                                 "--symbolic-full-name", "@{u}"))) > 0L
    if (dirty) {
      message("[pdm] Uncommitted changes -> skipping automatic 'git pull'.")
    } else if (!has_upstream) {
      message("[pdm] No upstream branch -> skipping automatic 'git pull'.")
    } else {
      before <- git(c("rev-parse", "HEAD"))
      message("[pdm] Checking for updates (git pull)...")
      out <- tryCatch(
        suppressWarnings(system2("git", "pull", stdout = TRUE, stderr = TRUE)),
        error = function(e) paste("git pull failed:", conditionMessage(e))
      )
      if (length(out)) message(paste(out, collapse = "\n"))
      after <- git(c("rev-parse", "HEAD"))
      if (length(before) && length(after) && !identical(before, after)) {
        changed <- git(c("diff", "--name-only", before, after))
        if (any(changed == "renv.lock"))
          message("[pdm] renv.lock changed -> run renv::restore() to sync your library.")
        if (any(grepl("^src/", changed)))
          message("[pdm] Compiled code changed -> re-run devtools::load_all() to rebuild.")
      }
    }
  }

  ## -- 2b. Welcome banner -------------------------------------------
  branch <- git(c("rev-parse", "--abbrev-ref", "HEAD"))
  banner <- c(
    "----------------------------------------------------------------",
    " pdm | Polynomial Dynamic Models -- R package",
    if (length(branch)) sprintf(" branch : %s", branch),
    " load   : devtools::load_all()",
    " docs   : devtools::document()",
    " test   : devtools::test()",
    " check  : devtools::check()",
    "----------------------------------------------------------------"
  )
  message(paste(banner, collapse = "\n"))
})
