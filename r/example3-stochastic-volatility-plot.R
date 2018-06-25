###############################################################################
#
# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
#
# (c) 2018 Johan Dahlin
# uni (at) johandahlin.com
#
# Code function
# Recreates Figure 5 in the paper after running the Python code for the
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
plotColors <- brewer.pal(8, "Dark2")
output_path <- "~/src/pmmh-qn/results/example3-stochastic-volatility"
filePaths <- c("mh2/example3-mh2_0/mcmc_output.json.gz", "qmh_bfgs/example3-qmh_bfgs_0/mcmc_output.json.gz", "qmh_ls/example3-qmh_ls_0/mcmc_output.json.gz", "qmh_sr1/example3-qmh_sr1_0/mcmc_output.json.gz")
noIterations <- 15000
noIterationsMH <- floor(noIterations * 18 / 101)
MHiters <- seq(1, noIterations - noIterationsMH)

###############################################################################
# Data pre-processing
###############################################################################
j <- 1
traces <- array(0, dim=c(length(filePaths), noIterations, 4))
for (i in 1:length(filePaths)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      traces[i, ,] <- result$params[12001:27000,]
      if (i == 1) {
            traces[i, seq(noIterationsMH, noIterations), ] <- NA
      }
      print(mean(result$accepted))
}

dates <- seq(as.POSIXct("2016-11-07 01:00:00 CET"), as.POSIXct("2017-11-02 01:00:00 CET"), by = "1 day")
data <- read_json(paste(output_path, "mh2/example3-mh2_0/data.json.gz", sep="/"), simplifyVector = TRUE)
data <- data.frame(y=data$observations, x=dates)

results_ls <- read_json(paste(output_path, filePaths[3], sep="/"), simplifyVector = TRUE)
state_estimate <- data.frame(mean=apply(results_ls$state_trajectory, 2, mean), lower_ci=apply(results_ls$state_trajectory, 2, quantile, probs=0.025), upper_ci=apply(results_ls$state_trajectory, 2, quantile, probs=0.975), x=dates)

###############################################################################
# Plotting
###############################################################################
grid <- seq(1, noIterations)
trace_mu <- data.frame(mh2=traces[1,,1], bfgs=traces[2,,1], ls=traces[3,,1], sr1=traces[4,,1], x=grid)
trace_phi <- data.frame(mh2=traces[1,,2], bfgs=traces[2,,2], ls=traces[3,,2], sr1=traces[4,,2], x=grid)
trace_sigma <- data.frame(mh2=traces[1,,3], bfgs=traces[2,,3], ls=traces[3,,3], sr1=traces[4,,3], x=grid)
trace_rho <- data.frame(mh2=traces[1,,4], bfgs=traces[2,,4], ls=traces[3,,4], sr1=traces[4,,4], x=grid)

d1 <- ggplot(data=data, aes(x=x, y=y)) +
      geom_line(col=plotColors[1]) +
      geom_ribbon(aes(ymin=-14, ymax=y), alpha=0.25, fill=plotColors[1]) +
      labs(x = "date", y = "log-return") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

s1 <- ggplot(data=state_estimate, aes(x=x, y=mean)) +
      geom_line(col=plotColors[2]) +
      geom_ribbon(aes(ymin=lower_ci, ymax=upper_ci), alpha=0.25, fill=plotColors[2]) +
      labs(x = "date", y = "log-volatility") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p1 <- ggplot(data=trace_mu, aes(x=mh2)) +
      geom_density(aes(x=mh2), alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_density(aes(x=bfgs), alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_density(aes(x=ls), alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_density(aes(x=sr1), alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      labs(x = expression(mu), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p2 <- ggplot(data=trace_phi, aes(x=mh2)) +
      geom_density(aes(x=mh2), alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_density(aes(x=bfgs), alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_density(aes(x=ls), alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_density(aes(x=sr1), alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      labs(x = expression(phi), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p3 <- ggplot(data=trace_sigma, aes(x=mh2)) +
      geom_density(aes(x=mh2), alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_density(aes(x=bfgs), alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_density(aes(x=ls), alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_density(aes(x=sr1), alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      labs(x = expression(sigma[v]), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p4 <- ggplot(data=trace_rho, aes(x=mh2)) +
      geom_density(aes(x=mh2), alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_density(aes(x=bfgs), alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_density(aes(x=ls), alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_density(aes(x=sr1), alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      labs(x = expression(rho), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

# Write to file
cairo_pdf("~/src/uon-papers/pmmh-qn/draft1/images/example3-stochastic-volatility.pdf", width=4, height=6)
      layout=matrix(c(1, 1, 2, 2, 3, 4, 5, 6), nrow=4, byrow=TRUE)
      multiplot(d1, s1, p1, p2, p3, p4, layout=layout)
dev.off()


###############################################################################
# End of file
###############################################################################