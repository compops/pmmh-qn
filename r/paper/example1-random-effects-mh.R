library("jsonlite")
source("~/src/nlssm-base/r/paper/helper-example1.R")
source("~/src/nlssm-base/r/paper/helper-ggplot.R")
setwd("~/src/nlssm-base")

output_path <- "~/src/nlssm-base/results-draft1/example1-random-effects"
algorithms <- c("mh0", "mh1")
sigmaU <- c("0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0")

noAlgorithms <- length(algorithms)
noSigmaU <- length(sigmaU)
noSimulations <- 5
output <- array(0, dim = c(3, noSimulations, noAlgorithms, noSigmaU))

for (i in 1:(noAlgorithms)) {
  for (j in 1:noSimulations) {
    for (k in 1:noSigmaU) {
      file_path <- paste("example1", paste(algorithms[i], j-1, sep="-"), sep="-")
      file_path <- paste("sigmau", paste(sigmaU[k], paste(file_path), sep="/"), sep="-")
      file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")

      data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
      result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
      settings <- read_json(paste(file_path, "/settings.json.gz", sep=""), simplifyVector = TRUE)

      output[1, j, i, k] <- sigmaU[k]
      output[2:3, j, i, k] <- helper_table(data, result, settings, memLength=1)
      print(output[, j, i, k])
    }
  }
}

outputMedian <- array(0, dim = c(3, noAlgorithms, noSigmaU))
for (i in 1:(noAlgorithms)) {
  for (k in 1:noSigmaU) {
    outputMedian[1, i, k] <- sigmaU[k]
    outputMedian[2, i, k] <- median(as.numeric(output[2, , i, k]), na.rm=TRUE)
    outputMedian[3, i, k] <- median(as.numeric(output[3, , i, k]), na.rm=TRUE)
  }
}

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example1-mh0.pdf", height = 8, width = 8)
plot(outputMedian[2, 1,],
     outputMedian[3, 1,],
     type="p",
     pch=19,
     xlab="acceptance rate",
     ylab="ESS"
)
dev.off()

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example1-mh1.pdf", height = 8, width = 8)
plot(outputMedian[2, 2,],
     outputMedian[3, 2,],
     type="p",
     pch=19,
     xlab="acceptance rate",
     ylab="ESS"
)
dev.off()