library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2");

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")

output_path <- "~/src/pmmh-qn/results/example3-stochastic-volatility"
algorithms <- list.dirs(output_path, full.names = FALSE, recursive=FALSE)
noIterations <- 20000

j <- 8
traces <- matrix(0, nrow=5, ncol=noIterations)

for (k in 1:4) {
      i <- c(1, 3, 4, 5)[k]
      file_path <- paste("example3", paste(algorithms[i], j-1, sep="_"), sep="-")
      file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")
      result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
      traces[k, ] <- result$params[, 4]
      print(mean(result$accepted))
}

data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
data <- data.frame(y=data$observations, x=seq(1900, 2013))

# Plotting some traces
trace_mh0 <- data.frame(th=traces[1, ], x=seq(1, noIterations))
trace_mh2 <- data.frame(th=traces[2, ], x=seq(1, noIterations))
trace_bfgs <- data.frame(th=traces[3, ], x=seq(1, noIterations))
trace_ls <- data.frame(th=traces[4, ], x=seq(1, noIterations))
trace_sr1 <- data.frame(th=traces[5, ], x=seq(1, noIterations))

acf_mh0 <- acf(traces[1,], lag.max = 250, plot = FALSE)
acf_mh2 <- acf(traces[2,], lag.max = 250, plot = FALSE)
acf_bfgs <- acf(traces[3,], lag.max = 250, plot = FALSE)
acf_ls <- acf(traces[4,], lag.max = 250, plot = FALSE)
acf_sr1 <- acf(traces[5,], lag.max = 250, plot = FALSE)

acf_mh0 <- data.frame(acf = acf_mh0$acf, lag = acf_mh0$lag)
acf_mh2 <- data.frame(acf = acf_mh2$acf, lag = acf_mh2$lag)
acf_bfgs <- data.frame(acf = acf_bfgs$acf, lag = acf_bfgs$lag)
acf_ls <- data.frame(acf = acf_ls$acf, lag = acf_ls$lag)
acf_sr1 <- data.frame(acf = acf_sr1$acf, lag = acf_sr1$lag)


d1 <- ggplot(data=data, aes(x=x, y=y)) +
      geom_line(col=plotColors[1]) +
      geom_ribbon(aes(ymin=0, ymax=y), alpha=0.25, fill=plotColors[1]) +
      labs(x = "year", y = "no. earthquakes") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

p1 <- ggplot(data=trace_mh0, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[3], col=plotColors[3]) +
      labs(x = "", y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
p2 <- ggplot(data=trace_mh2, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[4], col=plotColors[4]) +
      labs(x = "", y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
p3 <- ggplot(data=trace_bfgs, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[5], col=plotColors[5]) +
      labs(x = "", y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
p4 <- ggplot(data=trace_ls, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[6], col=plotColors[6]) +
      labs(x = expression(sigma[v]), y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
p5 <- ggplot(data=trace_sr1, aes(x=th)) +
      geom_density(alpha=0.25, fill=plotColors[7], col=plotColors[7]) +
      labs(x = "", y = "posterior") +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

t1 <- ggplot(data=trace_mh0, aes(x=x, y=th)) +
      geom_line(col=plotColors[3]) +
      labs(x = "", y = expression(sigma[v])) +
      coord_cartesian(xlim=c(5000, 5500)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
t2 <- ggplot(data=trace_mh2, aes(x=x, y=th)) +
      geom_line(col=plotColors[4]) +
      labs(x = "", y = expression(sigma[v])) +
      coord_cartesian(xlim=c(5000, 5500)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
t3 <- ggplot(data=trace_bfgs, aes(x=x, y=th)) +
      geom_line(col=plotColors[5]) +
      labs(x = "", y = expression(sigma[v])) +
      coord_cartesian(xlim=c(5000, 5500)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
t4 <- ggplot(data=trace_sr1, aes(x=x, y=th)) +
      geom_line(col=plotColors[6]) +
      labs(x = "iteration", y = expression(sigma[v])) +
      coord_cartesian(xlim=c(5000, 5500)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
t5 <- ggplot(data=trace_ls, aes(x=x, y=th)) +
      geom_line(col=plotColors[7]) +
      labs(x = "", y = expression(sigma[v])) +
      coord_cartesian(xlim=c(5000, 5500)) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))

a1 <- ggplot(data=acf_mh0, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[3]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[3]) +
      labs(x = "", y = expression("ACF of " * sigma[v])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
a2 <- ggplot(data=acf_mh2, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[4]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[4]) +
      labs(x = "", y = expression("ACF of " * sigma[v])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
a3 <- ggplot(data=acf_bfgs, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[5]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[5]) +
      labs(x = "", y = expression("ACF of " * sigma[v])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
a4 <- ggplot(data=acf_ls, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[6]) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[6]) +
      labs(x = "lag", y = expression("ACF of " * sigma[v])) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
a5 <- ggplot(data=acf_sr1, aes(x=lag, y=acf)) +
      geom_line(col=plotColors[7]) +
      labs(x = "", y = expression("ACF of " * sigma[v])) +
      geom_ribbon(aes(ymin=0, ymax=acf), alpha=0.25, fill=plotColors[7]) +
      theme_minimal() +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))


cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example3-stochastic-volatility.pdf", width=4, height=8)
layout=matrix(c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), nrow=6, byrow=TRUE)
multiplot(d1, p1, a1, p2, a2, p3, a3, p4, a4, p5, p6, layout=layout)
dev.off()