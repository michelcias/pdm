# Activate the renv environment for project-specific package management
# This ensures that the correct package versions specified in renv.lock are used.
source("renv/activate.R")

# --- Automatic Git Pull on Startup ---
# Display a message indicating the process is starting
message("Attempting automatic git pull...")

# Execute the 'git pull' command in the system's shell
# Capture the exit status (0 for success, non-zero for failure)
# Note: Git's own output (stdout/stderr) will print directly to the console
pull_status <- system("git pull")

# Check the exit status returned by the system() call
if (pull_status == 0) {
  # Success message (exit status 0 usually indicates success)
  message("--> Success! Repository updated via 'git pull'.")
  message("--> If package dependencies (renv.lock) or the package source changed, consider restarting R or running renv::restore().")
} else {
  # Error message (a non-zero exit status indicates an error)
  message(paste("--> Error running 'git pull'. Exit code:", pull_status))
  message("--> Check the console for Git error messages and your repository status.")
}

# Clean up the status variable to avoid polluting the global environment
# This prevents the 'pull_status' variable from remaining in your R session.
rm(pull_status)

# --- End of Automatic Git Pull ---

# You can add other R startup commands below if needed.
