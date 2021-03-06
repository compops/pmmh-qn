###############################################################################
#
# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
#
# (c) 2018 Johan Dahlin
# uni (at) johandahlin.com
#
# Code function
# Recreates Table 2 in the paper after running the Python code for the
# first experiment
###############################################################################

library(jsonlite)
library(xtable)

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/helpers.R")

###############################################################################
# Settings
###############################################################################
options(xtable.floating = FALSE)
options(xtable.timestamp = "")
output_path <- "~/src/pmmh-qn/results/example3-stochastic-volatility"
algorithms <- c("mh0", "mh1", "mh2", "qmh_bfgs", "qmh_ls", "qmh_sr1")
noSimulations <- 10
offset <- 12000
memLength <- 1

###############################################################################
# Data pre-processing
###############################################################################
noAlgorithms <- length(algorithms)
output <- array(0, dim = c(6, noSimulations, noAlgorithms))
post_var <- array(0, dim = c(4, noSimulations, noAlgorithms))

for (i in 1:(noAlgorithms)) {
  for (j in 1:noSimulations) {
    file_path <- paste("example3", paste(algorithms[i], j-1, sep="_"), sep="-")
    file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")

    data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
    result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
    settings <- read_json(paste(file_path, "/settings.json.gz", sep=""), simplifyVector = TRUE)

    output[, j, i] <- helper_table(data, result, settings, memLength=memLength, offset=offset)
    post_var[, j, i] <- helper_post_var(result, memLength=memLength, offset=offset)
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

###############################################################################
# Create table
###############################################################################
row.names(medianOutput) <- algorithms
xtable(medianOutput, digits = c(0, 2, 2, 0, 0, 2, 2))

###############################################################################
# Comparing posterior covariances
###############################################################################

post_var_rel_bfgs <- matrix(0, nrow=noSimulations, ncol=4)
post_var_rel_ls <- matrix(0, nrow=noSimulations, ncol=4)
post_var_rel_sr1 <- matrix(0, nrow=noSimulations, ncol=4)

for (j in 1:noSimulations) {
  post_var_rel_bfgs[j,] <- post_var[, j, 4] / post_var[, j, 3]
  post_var_rel_ls[j,] <- post_var[, j, 5] / post_var[, j, 3]
  post_var_rel_sr1[j,] <- post_var[, j, 6] / post_var[, j, 3]
}

c(median(post_var_rel_bfgs), median(post_var_rel_ls), median(post_var_rel_sr1))

###############################################################################
# End of file
###############################################################################