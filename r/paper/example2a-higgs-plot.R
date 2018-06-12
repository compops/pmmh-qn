library(jsonlite)
library(ggplot2)
library(RColorBrewer)
plotColors = brewer.pal(8, "Dark2");

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")

output_path <- "~/src/pmmh-qn/results/example2-higgs"
noRuns <- 10
memory_lengths <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50)

err_bfgs <- matrix(0, nrow=noRuns, ncol=length(memory_lengths))
err_ls <- matrix(0, nrow=noRuns, ncol=length(memory_lengths))
err_sr1 <- matrix(0, nrow=noRuns, ncol=length(memory_lengths))

for (i in 1:noRuns) {
      file_path <- paste(paste("qmhb/example2-qmhb-", i-1, sep=""), "/benchmark.json.gz", sep="")
      result <- read_json(paste(output_path, file_path, sep="/"), simplifyVector = TRUE)
      err_bfgs[i,] <- apply(result$error_bfgs_fro, 2, median)
      err_ls[i,] <- apply(result$error_ls_fro, 2, median)
      err_sr1[i,] <- apply(result$error_sr1_fro, 2, median)
}

iqr_err_bfgs <- apply(err_bfgs, 2, IQR)
iqr_err_ls <- apply(err_ls, 2, IQR)
iqr_err_sr1 <- apply(err_sr1, 2, IQR)

err_bfgs <- apply(err_bfgs, 2, median)
err_ls <- apply(err_ls, 2, median)
err_sr1 <- apply(err_sr1, 2, median)

df <- data.frame(cbind(err_bfgs, err_ls, err_sr1, iqr_err_bfgs, iqr_err_ls, iqr_err_sr1, memory_lengths))

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example2a-higgs.pdf", width=4, height=3)
ggplot(data=df, aes(x=memory_lengths)) +
      geom_ribbon(aes(ymin=err_bfgs-iqr_err_bfgs, ymax=err_bfgs+iqr_err_bfgs), fill=plotColors[1], alpha=0.25) +
      geom_ribbon(aes(ymin=err_ls-iqr_err_ls, ymax=err_ls+iqr_err_ls), fill=plotColors[2], alpha=0.25) +
      geom_ribbon(aes(ymin=err_sr1-iqr_err_sr1, ymax=err_sr1+iqr_err_sr1), fill=plotColors[3], alpha=0.25) +
      geom_line(aes(x=memory_lengths, y=err_bfgs), col=plotColors[1], size=1) +
      geom_line(aes(x=memory_lengths, y=err_ls), col=plotColors[2], size=1) +
      geom_line(aes(x=memory_lengths, y=err_sr1), col=plotColors[3], size=1) +
      labs(x = "memory length", y = "Hessian error") +
      theme_minimal() +
      lims(y = c(0.0, 0.08)) +
      theme(axis.text=element_text(size=7), axis.title=element_text(size=8))
dev.off()

