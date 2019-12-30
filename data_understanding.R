# Returns an object that reports various information about the given dataset

data_understanding <- function(dataset) {
  
  cat("Data Understanding...\n")
  
  # Information about rows
  num_of_instances <- nrow(dataset)
  
  # Information about columns
  num_of_fields <- ncol(dataset)
  name_of_fields <- names(dataset)
  
  # Search for duplicated
  is_dup <- duplicated(dataset)
  num_of_duplicated <- length(Filter(isTRUE, is_dup))
  
  # Search for missing values
  is_missing <- !complete.cases(dataset)
  missing_values <- length(Filter(isTRUE, is_missing))
  
  # Assembling output
  output <- list(num_of_instances, num_of_fields, name_of_fields,
                 num_of_duplicated, missing_values)
  names(output) <- c("Size", "Number of Fields", "Fields", "Num of duplicated",
                     "Missing Values")
  cat("Done\n")
  return(output)
  
}