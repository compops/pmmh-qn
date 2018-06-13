library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2");

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")

output_path <- "~/src/pmmh-qn/results/example2-higgs"
filePaths <- c("mh0/example2-mh0-0/mcmc_output.json.gz", "mh2/example2-mh2-0/mcmc_output.json.gz", "qmh-ls/example2-qmh-ls-0/mcmc_output.json.gz")
paramToPlot <- 2
noIterations <- 20000
removeIterations <- 7000

traces <- matrix(0, nrow=length(filePaths), ncol=noIterations)

for (i in 1:length(filePaths)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      traces[i,] <- result$params[-(1:removeIterations), paramToPlot]
}

# Plotting some traces
trace1 <- data.frame(th=traces[1, ], x=seq(1, noIterations))
trace2 <- data.frame(th=traces[2, ], x=seq(1, noIterations))
trace3 <- data.frame(th=traces[3, ], x=seq(1, noIterations))

acf1 <- acf(trace1$th, lag.max = 250, plot = FALSE)
acf2 <- acf(trace2$th, lag.max = 250, plot = FALSE)
acf3 <- acf(trace3$th, lag.max = 250, plot = FALSE)

acf1_df <- data.frame(acf = acf1$acf, lag = acf1$lag)
acf2_df <- data.frame(acf = acf2$acf, lag = acf2$lag)
acf3_df <- data.frame(acf = acf3$acf, lag = acf3$lag)

p1 <- ggplot(data=trace1, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_vline(xintercept = -0.264) +
      labs(x = expression(beta[1]), y = "posterior") +
      theme_minimal() +
      lims(y = c(-0.5, 0.1)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p2 <- ggplot(data=trace2, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      geom_vline(xintercept = -0.264) +
      labs(x = expression(beta[1]), y = "posterior") +
      theme_minimal() +
      lims(y = c(-0.5, 0.1)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p3 <- ggplot(data=trace3, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_vline(xintercept = -0.264) +
      labs(x = expression(beta[1]), y = "posterior") +
      theme_minimal() +
      lims(y = c(-0.5, 0.1)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))


a1 <- ggplot(data=acf1_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[4]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
      labs(x = "", y = expression("ACF of " * beta[1])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a2 <- ggplot(data=acf2_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[5]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[5]) +
      labs(x = "", y = expression("ACF of " * beta[1])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a3 <- ggplot(data=acf3_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[3]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
      labs(x = "", y = expression("ACF of " * beta[1])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))


cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example2b-higgs.pdf", width=4, height=6)
layout=matrix(seq(1, 6), nrow=3, byrow=TRUE)
multiplot(p1, a1, p2, a2, p3, a3, layout=layout)
dev.off()