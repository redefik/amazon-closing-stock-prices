# Returns the dataset imported from the given file.
# The input file is a comma-separated csv with an header.

import_dataset <- function(filename) {
  cat("Import Dataset...\n")
  out <- read.csv(filename, header = TRUE, sep = ",")
  cat("Done\n")
  return(out)
}