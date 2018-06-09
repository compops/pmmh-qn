library("jsonlite")
source("~/src/pmmh-qn/r/tests/helper_plotting.R")

setwd("~/src/pmmh-qn/results/example3-stochastic-volatility")
file_path <- ""
algorithms <- c("mh2/example3-mh2_0", "qmh_sr1/example3-qmh_sr1_0", "qmh_ls/example3-qmh_ls_0")
memory_lengths <- rep(1, 3)

offset <- c(0)
noItersToPlot <- 200
savePlotToFile <- TRUE
paramsScale <- c(1, 3.5, 0.85, 1, 0.2, 0.6, -0.6, 0.2)

for (i in 1:length(algorithms)) {
  algorithm <- algorithms[i]

  data <- read_json(paste(file_path,
                          paste(algorithm, "/data.json.gz", sep=""),
                          sep=""),
                    simplifyVector = TRUE)
  result <- read_json(paste(file_path,
                            paste(algorithm, "/mcmc_output.json.gz", sep=""),
                            sep=""),
                      simplifyVector = TRUE)
  settings <- read_json(paste(file_path,
                              paste(algorithm, "/settings.json.gz", sep=""),
                              sep=""),
                        simplifyVector = TRUE)

  iact <- helper_plotting(data=data,
                          result=result,
                          settings=settings,
                          algorithm=algorithm,
                          noItersToPlot=noItersToPlot,
                          savePlotToFile=savePlotToFile,
                          paramsScale=paramsScale,
                          folderToSaveTo=file_path,
                          memory_length=memory_lengths[i],
                          yrange=c(-16, 16),
                          xrange=c(-4, 4))
}
