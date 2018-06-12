library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2");

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")

output_path <- "~/src/pmmh-qn/results/example2-higgs"
file_path <- "qmh-ls/example2-qmh-ls-0/mcmc_output.json.gz"
paramsToPlot <- c(2, 18)
removeIterations <- 12000

result <- read_json(paste(output_path, file_path, sep="/"), simplifyVector = TRUE)
traces <- result$params[-(1:removeIterations),]
noIterations <- dim(traces)[1]


# Plotting some traces
trace1 <- data.frame(th=traces[, paramsToPlot[1]], x=seq(1, noIterations))
trace2 <- data.frame(th=traces[, paramsToPlot[2]], x=seq(1, noIterations))

acf1 <- acf(trace1$th, lag.max = 1000, plot = FALSE)
acf2 <- acf(trace2$th, lag.max = 1000, plot = FALSE)

acf1_df <- data.frame(acf = acf1$acf, lag = acf1$lag)
acf2_df <- data.frame(acf = acf2$acf, lag = acf2$lag)

p1 <- ggplot(data=trace1, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      geom_vline(xintercept = -0.264) +
      labs(x = expression(beta[1]), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p2 <- ggplot(data=trace2, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      geom_vline(xintercept = -0.121) +
      labs(x = expression(beta[17]), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

t1 <- ggplot(data=trace1, aes(x=x, y=th)) +
      geom_line(col=plotColors[3]) +
      labs(x = "iteration", y = expression(beta[1])) +
      coord_cartesian(xlim=c(5000, 5100)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

t2 <- ggplot(data=trace2, aes(x=x, y=th)) +
      geom_line(col=plotColors[4]) +
      labs(x = "iteration", y = expression(beta[17])) +
      coord_cartesian(xlim=c(5000, 5100)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a1 <- ggplot(data=acf1_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[3]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
      labs(x = "", y = expression("ACF of " * beta[1])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a2 <- ggplot(data=acf2_df, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[4]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
      labs(x = "", y = expression("ACF of " * beta[17])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example2b-higgs.pdf", width=4, height=6)
layout=matrix(seq(1, 6), nrow=3, byrow=TRUE)
multiplot(p1, p2, t1, t2, a1, a2, layout=layout)
dev.off()