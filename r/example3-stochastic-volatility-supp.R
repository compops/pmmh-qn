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
# Settings
###############################################################################
plotColors <- brewer.pal(8, "Dark2")
output_path <- "~/src/pmmh-qn/results/example3-stochastic-volatility"
filePaths <- c("mh2/example3-mh2_0/mcmc_output.json.gz", "qmh_bfgs/example3-qmh_bfgs_0/mcmc_output.json.gz", "qmh_ls/example3-qmh_ls_0/mcmc_output.json.gz", "qmh_sr1/example3-qmh_sr1_0/mcmc_output.json.gz")
noIterations <- 15000

###############################################################################
# Data pre-processing
###############################################################################
j <- 1
traces <- array(0, dim=c(length(filePaths), noIterations, 4))
for (i in 1:length(filePaths)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      traces[i,, ] <- result$params[12001:27000, ]
      print(mean(result$accepted))
}

###############################################################################
# Plotting
###############################################################################
file_name <- c("~/src/uon-papers/pmmh-qn/draft1/example3-stochastic-volatility-supp-mh2.pdf", "~/src/uon-papers/pmmh-qn/draft1/example3-stochastic-volatility-supp-bfgs.pdf", "~/src/uon-papers/pmmh-qn/draft1/example3-stochastic-volatility-supp-ls.pdf",
"~/src/uon-papers/pmmh-qn/draft1/example3-stochastic-volatility-supp-sr1.pdf")


for (i in 1:4) {
    trace <- data.frame(th=traces[i,, ], x=seq(1, noIterations))

    acf_mu <- acf(trace[,1], lag.max = 250, plot = FALSE)
    acf_phi <- acf(trace[,2], lag.max = 250, plot = FALSE)
    acf_sigma <- acf(trace[,3], lag.max = 250, plot = FALSE)
    acf_rho <- acf(trace[,4], lag.max = 250, plot = FALSE)

    acf_mu <- data.frame(acf = acf_mu$acf, lag = acf_mu$lag)
    acf_phi <- data.frame(acf = acf_phi$acf, lag = acf_phi$lag)
    acf_sigma <- data.frame(acf = acf_sigma$acf, lag = acf_sigma$lag)
    acf_rho <- data.frame(acf = acf_rho$acf, lag = acf_rho$lag)

    t1 <- ggplot(data=trace, aes(x=x, y=th.1)) +
        geom_line(col=plotColors[3]) +
        lims(x = c(8800, 9000), y = c(0.0, 3.5)) +
        labs(y = expression(mu), x = "iteration") +
        theme_minimal() #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    t2 <- ggplot(data=trace, aes(x=x, y=th.2)) +
        geom_line(col=plotColors[4]) +
        lims(x = c(8800, 9000), y = c(0.8, 1.0)) +
        labs(y = expression(phi), x = "iteration") +
        theme_minimal() #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    t3 <- ggplot(data=trace, aes(x=x, y=th.3)) +
        geom_line(col=plotColors[5]) +
        lims(x = c(8800, 9000), y = c(0.2, 0.7)) +
        labs(y = expression(sigma[v]), x = "iteration") +
        theme_minimal() #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    t4 <- ggplot(data=trace, aes(x=x, y=th.4)) +
        geom_line(col=plotColors[6]) +
        lims(x = c(8800, 9000), y = c(-0.5, 0.2)) +
        labs(y = expression(rho), x = "iteration") +
        theme_minimal() #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    p1 <- ggplot(data=trace, aes(x=th.1)) +
        geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
        labs(x = expression(mu), y = "posterior") +
        theme_minimal() +
        lims(x = c(0.0, 3.5)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    p2 <- ggplot(data=trace, aes(x=th.2)) +
        geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
        labs(x = expression(phi), y = "posterior") +
        theme_minimal() +
        lims(x = c(0.8, 1.0)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    p3 <- ggplot(data=trace, aes(x=th.3)) +
        geom_density(alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
        labs(x = expression(sigma[v]), y = "posterior") +
        theme_minimal() +
        lims(x = c(0.2, 0.7)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    p4 <- ggplot(data=trace, aes(x=th.4)) +
        geom_density(alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
        labs(x = expression(rho), y = "posterior") +
        theme_minimal() +
        lims(x = c(-0.5, 0.2)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    a1 <- ggplot(data=acf_mu, aes(x=lag, y=acf)) +
        geom_line(col=plotColors[3]) +
        geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
        labs(x = "lag", y = expression("ACF of " * mu)) +
        theme_minimal() +
        lims(y = c(-0.3, 1.0)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    a2 <- ggplot(data=acf_phi, aes(x=lag, y=acf)) +
        geom_line(col=plotColors[4]) +
        geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
        labs(x = "lag", y = expression("ACF of " * phi)) +
        theme_minimal() +
        lims(y = c(-0.3, 1.0)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    a3 <- ggplot(data=acf_sigma, aes(x=lag, y=acf)) +
        geom_line(col=plotColors[5]) +
        labs(x = "lag", y = expression("ACF of " * sigma[v])) +
        geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[5]) +
        theme_minimal() +
        lims(y = c(-0.3, 1.0)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    a4 <- ggplot(data=acf_rho, aes(x=lag, y=acf)) +
        geom_line(col=plotColors[6]) +
        labs(x = "lag", y = expression("ACF of " * rho)) +
        geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[6]) +
        theme_minimal() +
        lims(y = c(-0.3, 1.0)) #+
        #theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

    # Write to file
    cairo_pdf(file_name[i], width=8, height=10)
        layout=matrix(1:12, nrow=4, byrow=TRUE)
        multiplot(p1, t1, a1, p2, t2, a2, p3, t3, a3, p4, t4, a4, layout=layout)
    dev.off()
}


###############################################################################
# End of file
###############################################################################