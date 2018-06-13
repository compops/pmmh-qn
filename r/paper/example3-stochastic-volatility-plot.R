library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2")

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")

output_path <- "~/src/pmmh-qn/results/example3-stochastic-volatility"
#filePaths <- c("mh0/example3-mh0_0/mcmc_output.json.gz", "mh2/example3-mh2_0/mcmc_output.json.gz", "qmh_ls/example3-qmh_ls_0/mcmc_output.json.gz")
filePaths <- c("mh0/example3-mh0_0/mcmc_output.json.gz", "mh2-N75/example3-mh2_0/mcmc_output.json.gz", "qmh_ls/example3-qmh_ls_0/mcmc_output.json.gz")
noIterations <- 20000

j <- 1
traces <- matrix(0, nrow=length(algorithms), ncol=noIterations)

for (i in 1:length(algorithms)) {
      result <- read_json(paste(output_path, filePaths[i], sep="/"), simplifyVector = TRUE)
      traces[i, ] <- result$params[7001:27000, 4]
      print(mean(result$accepted))
}

dates <- seq(as.POSIXct("2016-11-07 01:00:00 CET"), as.POSIXct("2017-11-02 01:00:00 CET"), by = "1 day")
data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
data <- data.frame(y=data$observations, x=dates)

# Plotting some traces
trace_mh0 <- data.frame(th=traces[1, ], x=seq(1, noIterations))
trace_mh2 <- data.frame(th=traces[2, ], x=seq(1, noIterations))
trace_ls <- data.frame(th=traces[3, ], x=seq(1, noIterations))

acf_mh0 <- acf(traces[1,], lag.max = 250, plot = FALSE)
acf_mh2 <- acf(traces[2,], lag.max = 250, plot = FALSE)
acf_ls <- acf(traces[3,], lag.max = 250, plot = FALSE)

acf_mh0 <- data.frame(acf = acf_mh0$acf, lag = acf_mh0$lag)
acf_mh2 <- data.frame(acf = acf_mh2$acf, lag = acf_mh2$lag)
acf_ls <- data.frame(acf = acf_ls$acf, lag = acf_ls$lag)


d1 <- ggplot(data=data, aes(x=x, y=y)) +
      geom_line(col=plotColors[1]) +
      geom_ribbon(aes(ymin=-14, ymax=y), alpha=0.25, fill=plotColors[1]) +
      labs(x = "date", y = "log-return") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p1 <- ggplot(data=trace_mh0, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      labs(x = "", y = "posterior") +
      coord_cartesian(xlim=c(-0.5, 0.2)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p2 <- ggplot(data=trace_mh2, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      labs(x = "", y = "posterior") +
      coord_cartesian(xlim=c(-0.5, 0.2)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p3 <- ggplot(data=trace_ls, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      labs(x = "", y = "posterior") +
      coord_cartesian(xlim=c(-0.5, 0.2)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a1 <- ggplot(data=acf_mh0, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[4]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
      labs(x = "", y = expression("ACF of " * rho)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a2 <- ggplot(data=acf_mh2, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[5]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[5]) +
      labs(x = "", y = expression("ACF of " * rho)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a3 <- ggplot(data=acf_ls, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[3]) +
      labs(x = "", y = expression("ACF of " * rho)) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))


cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example3-stochastic-volatility.pdf", width=4, height=6)
layout=matrix(c(1, 1, 2, 3, 4, 5, 6, 7), nrow=4, byrow=TRUE)
multiplot(d1, p1, a1, p2, a2, p3, a3, layout=layout)
dev.off()