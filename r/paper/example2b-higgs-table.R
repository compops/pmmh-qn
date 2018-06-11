setwd("~/src/pmmh-qn")
source("r/paper/helper-table.R")

library("jsonlite")
library("xtable")
options(xtable.floating = FALSE)
options(xtable.timestamp = "")

output_path <- "~/src/pmmh-qn/results/example2-higgs"
algorithms <- c("mh0", "mh2", "qmh-bfgs", "qmh-ls", "qmh-sr1")
noSimulations <- 10
offset <- 7000

noAlgorithms <- length(algorithms)
output <- array(0, dim = c(6, noSimulations, noAlgorithms))

for (i in 1:(noAlgorithms)) {
  for (j in 1:noSimulations) {
    file_path <- paste("example2", paste(algorithms[i], j-1, sep="-"), sep="-")
    file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")

    data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
    result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
    settings <- read_json(paste(file_path, "/settings.json.gz", sep=""), simplifyVector = TRUE)

    output[, j, i] <- helper_table(data, result, settings, memLength=memLength, offset=offset)
    print(output[, j, i])
  }
}

medianOutput <- matrix(0, nrow = noAlgorithms, ncol = 6)
for (i in 1:noAlgorithms) {
  outputMethod <- matrix(as.numeric(output[-1, , i]), nrow = noSimulations, ncol = 5, byrow = TRUE)
  medianOutput[i, 1] <- median(outputMethod[, 1], na.rm = TRUE)
  medianOutput[i, 2] <- median(outputMethod[, 2], na.rm = TRUE)
  medianOutput[i, 3] <- median(outputMethod[, 3], na.rm = TRUE)
  medianOutput[i, 4] <- IQR(outputMethod[, 3], na.rm = TRUE)
  medianOutput[i, 5] <- median(1000 * outputMethod[, 4], na.rm = TRUE)
  medianOutput[i, 6] <- median(1000 * outputMethod[, 5], na.rm = TRUE)
}

medianOutput[, 1] <- round(medianOutput[, 1], 2)
medianOutput[, 2] <- round(medianOutput[, 2], 2)
medianOutput[, 3] <- round(medianOutput[, 3], 0)
medianOutput[, 4] <- round(medianOutput[, 4], 0)
medianOutput[, 5] <- round(medianOutput[, 5], 2)
medianOutput[, 6] <- round(medianOutput[, 6], 2)

row.names(medianOutput) <- algorithms
xtable(medianOutput, digits = c(0, 2, 2, 0, 0, 2, 2))

