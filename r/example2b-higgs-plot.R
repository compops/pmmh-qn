###############################################################################
#
# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
#
# (c) 2018 Johan Dahlin
# uni (at) johandahlin.com
#
# Code function
# Recreates Figure 4 in the paper after running the Python code for the
# first experiment
###############################################################################

library(jsonlite)
library(ggplot2)
library(RColorBrewer)

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/helpers.R")

###############################################################################
# Settings
###############################################################################
plotColors <- brewer.pal(8, "Dark2");
output_path <- "~/src/pmmh-qn/results/example2-higgs"
filePaths <- c("mh2/example2-mh2-0/mcmc_output.json.gz", "qmh-bfgs/example2-qmh-bfgs-0/mcmc_output.json.gz", "qmh-ls/example2-qmh-ls-0/mcmc_output.json.gz", "qmh-sr1/example2-qmh-sr1-0/mcmc_output.json.gz")
paramToPlot <- 12
noIterations <- 15000
removeIterations <- 12000
sgd_beta <- c(-0.1259617, -0.36108665, -0.00668518, -0.00197801, -0.38769742, -0.00151412, 0.30569725, -0.00671408, -0.00120516, -0.12841105, -0.04335106, 0.01007512, -0.00070083, -0.17339557, -0.04404842,  0.00269722, 0.00543087, -0.11826687, 0.04320192, -0.00406945, 0.00722225, -0.06106104)[paramToPlot]


###############################################################################
# Data pre-processing
###############################################################################
traces <- matrix(0, nrow=length(filePaths), ncol=noIterations)

for (i in 1:length(filePaths)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      traces[i,] <- result$params[-(1:removeIterations), paramToPlot]
}

###############################################################################
# Plotting
###############################################################################
trace1 <- data.frame(th=traces[1, ], x=seq(1, noIterations))
trace2 <- data.frame(th=traces[2, ], x=seq(1, noIterations))
trace3 <- data.frame(th=traces[3, ], x=seq(1, noIterations))
trace4 <- data.frame(th=traces[4, ], x=seq(1, noIterations))

acf1 <- acf(trace1$th, lag.max = 250, plot = FALSE)
acf2 <- acf(trace2$th, lag.max = 250, plot = FALSE)
acf3 <- acf(trace3$th, lag.max = 250, plot = FALSE)
acf4 <- acf(trace4$th, lag.max = 250, plot = FALSE)

acf1_df <- data.frame(acf = acf1$acf, lag = acf1$lag)
acf2_df <- data.frame(acf = acf2$acf, lag = acf2$lag)
acf3_df <- data.frame(acf = acf3$acf, lag = acf3$lag)
acf4_df <- data.frame(acf = acf4$acf, lag = acf4$lag)

p1 <- ggplot(data=trace1, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_vline(xintercept = sgd_beta) +
      labs(x = expression(beta[12]), y = "posterior") +
      theme_minimal() +
      lims(x = c(-0.1, 0.15)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p2 <- ggplot(data=trace2, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_vline(xintercept = sgd_beta) +
      labs(x = expression(beta[12]), y = "posterior") +
      theme_minimal() +
      lims(x = c(-0.1, 0.15)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p3 <- ggplot(data=trace3, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_vline(xintercept = sgd_beta) +
      labs(x = expression(beta[12]), y = "posterior") +
      theme_minimal() +
      lims(x = c(-0.1, 0.15)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p4 <- ggplot(data=trace4, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      geom_vline(xintercept = sgd_beta) +
      labs(x = expression(beta[12]), y = "posterior") +
      theme_minimal() +
      lims(x = c(-0.1, 0.15)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))



a1 <- ggplot(data=acf1_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[3]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
      labs(x = "", y = expression("ACF of " * beta[12])) +
      theme_minimal() +
      coord_cartesian(ylim=c(-0.3, 1.0)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a2 <- ggplot(data=acf2_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[4]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
      labs(x = "", y = expression("ACF of " * beta[12])) +
      theme_minimal() +
      coord_cartesian(ylim=c(-0.3, 1.0)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a3 <- ggplot(data=acf3_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[5]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[5]) +
      labs(x = "", y = expression("ACF of " * beta[12])) +
      theme_minimal() +
      coord_cartesian(ylim=c(-0.3, 1.0)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a4 <- ggplot(data=acf4_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[6]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[6]) +
      labs(x = "", y = expression("ACF of " * beta[12])) +
      theme_minimal() +
      coord_cartesian(ylim=c(-0.3, 1.0)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

cairo_pdf("~/src/uon-papers/pmmh-qn/draft1/images/example2b-higgs.pdf", width=4, height=6)
      layout=matrix(seq(1, 8), nrow=4, byrow=TRUE)
      multiplot(p1, a1, p2, a2, p3, a3, p4, a4, layout=layout)
dev.off()

###############################################################################
# End of file
###############################################################################