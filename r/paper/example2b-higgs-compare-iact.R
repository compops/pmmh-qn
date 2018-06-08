library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2");

setwd("~/src/nlssm-base")
source("~/src/nlssm-base/r/paper/helper-ggplot.R")

removeIterations <- 12000
output_path <- "~/src/nlssm-base/results-draft1/example3-higgs"

### pmMH using random walk proposal
file_path <- "mh0/example3-mh0-0/mcmc_output.json.gz"

result <- read_json(paste(output_path, file_path, sep="/"), simplifyVector = TRUE)
traces <- result$params[-(1:removeIterations),]
noParameters <- dim(traces)[2]

mh0MeanIACT <- 0
for (i in 1:noParameters) {
    foo <- acf(traces[, i], lag.max = 1000, plot = FALSE)
    mh0MeanIACT <- mh0MeanIACT + 2.0 * sum(foo$acf) / noParameters
}

(mh0MeanTES <- mh0MeanIACT * result$time_per_iteration)


### pmMH using LS proposal
file_path <- "qmh-ls/example3-qmh-ls-0/mcmc_output.json.gz"

result <- read_json(paste(output_path, file_path, sep="/"), simplifyVector = TRUE)
traces <- result$params[-(1:removeIterations),]
noParameters <- dim(traces)[2]

lesMeanIACT <- 0
for (i in 1:noParameters) {
    foo <- acf(traces[, i], lag.max = 1000, plot = FALSE)
    lesMeanIACT <- lesMeanIACT + 2.0 * sum(foo$acf) / noParameters
}

(lsMeanTES <- lesMeanIACT * result$time_per_iteration)