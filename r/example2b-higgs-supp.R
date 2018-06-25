###############################################################################
#
# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
#
# (c) 2018 Johan Dahlin
# uni (at) johandahlin.com
#
# Code function
# Recreates plots in supplimentary material after running the Python code for
# the first experiment
###############################################################################

library(jsonlite)
library(ggplot2)
library(RColorBrewer)

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/helpers.R")

###############################################################################
# Helper
###############################################################################
makeSubplotPosteriors <- function(data, true_params, index, xlabel) {
    noIterations <- dim(traces)[2]
    trace <- data.frame(th=t(traces[, , index]), x=seq(1, noIterations))
    ggplot(data=trace, aes(x=th.1)) +
       geom_density(aes(x=th.1), alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
       geom_density(aes(x=th.2), alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
       geom_density(aes(x=th.3), alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
       geom_density(aes(x=th.4), alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
       labs(x = xlabel, y = "posterior") +
       geom_vline(xintercept = true_params[index]) +
       theme_minimal() +
       theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
}

makeSubplotTraces <- function(data, true_params, index, ylabel) {
    noIterations <- dim(traces)[2]
    trace <- data.frame(th=t(data[, , index]), x=seq(1, noIterations))
    ggplot(data=trace, aes(x=x, y=th.1)) +
       geom_line(aes(x=x, y=th.2), col=plotColors[4]) +
       geom_line(aes(x=x, y=th.3), col=plotColors[5]) +
       geom_line(aes(x=x, y=th.4), col=plotColors[6]) +
       geom_line(aes(x=x, y=th.1), col=plotColors[3]) +
       labs(y = ylabel, x = "iteration") +
       geom_hline(yintercept = true_params[index]) +
       lims(x = c(8800, 9000)) +
       theme_minimal() +
       theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
}

###############################################################################
# Settings
###############################################################################
plotColors <- brewer.pal(8, "Dark2");
output_path <- "~/src/pmmh-qn/results/example2-higgs"
filePaths <- c("mh2/example2-mh2-0/mcmc_output.json.gz", "qmh-bfgs/example2-qmh-bfgs-0/mcmc_output.json.gz", "qmh-ls/example2-qmh-ls-0/mcmc_output.json.gz", "qmh-sr1/example2-qmh-sr1-0/mcmc_output.json.gz")
paramToPlot <- 2
noIterations <- 10000
removeIterations <- 17000
sgd_beta <- c(-0.15259617, -0.26108665, -0.00668518, -0.00197801, -0.38769742, -0.00151412, 0.30569725, -0.00671408, -0.00120516, -0.12841105, -0.04335106, 0.01007512, -0.00070083, -0.17339557, -0.04404842,  0.00269722, 0.00543087, -0.11826687, 0.04320192, -0.00406945, 0.00722225, -0.06106104)

###############################################################################
# Data pre-processing
###############################################################################
traces <- array(0, dim=c(length(filePaths), noIterations, 22))

for (i in 1:length(filePaths)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      for (j in 1:22) {
        traces[i,,j] <- as.numeric(result$params[-(1:removeIterations), j])
      }
}

###############################################################################
# Plotting posteriors
###############################################################################

p02 <- makeSubplotPosteriors(traces, sgd_beta, 2, bquote(beta[1]))
p03 <- makeSubplotPosteriors(traces, sgd_beta, 3, bquote(beta[2]))
p04 <- makeSubplotPosteriors(traces, sgd_beta, 4, bquote(beta[3]))
p05 <- makeSubplotPosteriors(traces, sgd_beta, 5, bquote(beta[4]))
p06 <- makeSubplotPosteriors(traces, sgd_beta, 6, bquote(beta[5]))
p07 <- makeSubplotPosteriors(traces, sgd_beta, 7, bquote(beta[6]))
p08 <- makeSubplotPosteriors(traces, sgd_beta, 8, bquote(beta[7]))
p09 <- makeSubplotPosteriors(traces, sgd_beta, 9, bquote(beta[8]))
p10 <- makeSubplotPosteriors(traces, sgd_beta, 10, bquote(beta[9]))
p11 <- makeSubplotPosteriors(traces, sgd_beta, 11, bquote(beta[10]))
p12 <- makeSubplotPosteriors(traces, sgd_beta, 12, bquote(beta[11]))
p13 <- makeSubplotPosteriors(traces, sgd_beta, 13, bquote(beta[12]))
p14 <- makeSubplotPosteriors(traces, sgd_beta, 14, bquote(beta[13]))
p15 <- makeSubplotPosteriors(traces, sgd_beta, 15, bquote(beta[14]))
p16 <- makeSubplotPosteriors(traces, sgd_beta, 16, bquote(beta[15]))
p17 <- makeSubplotPosteriors(traces, sgd_beta, 17, bquote(beta[16]))
p18 <- makeSubplotPosteriors(traces, sgd_beta, 18, bquote(beta[17]))
p19 <- makeSubplotPosteriors(traces, sgd_beta, 19, bquote(beta[18]))
p20 <- makeSubplotPosteriors(traces, sgd_beta, 20, bquote(beta[19]))
p21 <- makeSubplotPosteriors(traces, sgd_beta, 21, bquote(beta[20]))
p22 <- makeSubplotPosteriors(traces, sgd_beta, 22, bquote(beta[21]))

cairo_pdf("~/src/uon-papers/pmmh-qn/supplementary-draft1/images/example2b-higgs-supp-post.pdf", width=8, height=10)
      layout=matrix(seq(1, 21), nrow=7, byrow=TRUE)
      multiplot(p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, layout=layout)
dev.off()

###############################################################################
# Plotting traces
###############################################################################

p02 <- makeSubplotTraces(traces, sgd_beta, 2, bquote(beta[1]))
p03 <- makeSubplotTraces(traces, sgd_beta, 3, bquote(beta[2]))
p04 <- makeSubplotTraces(traces, sgd_beta, 4, bquote(beta[3]))
p05 <- makeSubplotTraces(traces, sgd_beta, 5, bquote(beta[4]))
p06 <- makeSubplotTraces(traces, sgd_beta, 6, bquote(beta[5]))
p07 <- makeSubplotTraces(traces, sgd_beta, 7, bquote(beta[6]))
p08 <- makeSubplotTraces(traces, sgd_beta, 8, bquote(beta[7]))
p09 <- makeSubplotTraces(traces, sgd_beta, 9, bquote(beta[8]))
p10 <- makeSubplotTraces(traces, sgd_beta, 10, bquote(beta[9]))
p11 <- makeSubplotTraces(traces, sgd_beta, 11, bquote(beta[10]))
p12 <- makeSubplotTraces(traces, sgd_beta, 12, bquote(beta[11]))
p13 <- makeSubplotTraces(traces, sgd_beta, 13, bquote(beta[12]))
p14 <- makeSubplotTraces(traces, sgd_beta, 14, bquote(beta[13]))
p15 <- makeSubplotTraces(traces, sgd_beta, 15, bquote(beta[14]))
p16 <- makeSubplotTraces(traces, sgd_beta, 16, bquote(beta[15]))
p17 <- makeSubplotTraces(traces, sgd_beta, 17, bquote(beta[16]))
p18 <- makeSubplotTraces(traces, sgd_beta, 18, bquote(beta[17]))
p19 <- makeSubplotTraces(traces, sgd_beta, 19, bquote(beta[18]))
p20 <- makeSubplotTraces(traces, sgd_beta, 20, bquote(beta[19]))
p21 <- makeSubplotTraces(traces, sgd_beta, 21, bquote(beta[20]))
p22 <- makeSubplotTraces(traces, sgd_beta, 22, bquote(beta[21]))

cairo_pdf("~/src/uon-papers/pmmh-qn/supplementary-draft1/images/example2b-higgs-supp-trace.pdf", width=8, height=10)
      layout=matrix(seq(1, 21), nrow=7, byrow=TRUE)
      multiplot(p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, layout=layout)
dev.off()




###############################################################################
# End of file
###############################################################################