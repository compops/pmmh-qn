library("jsonlite")
source("~/src/pmmh-qn/r/paper/helper-example1.R")
source("~/src/pmmh-qn/r/paper/helper-ggplot.R")
setwd("~/src/pmmh-qn")

output_path <- "~/src/pmmh-qn/results/example1-random-effects"
load("~/src/pmmh-qn/results/example1-random-effects/example1.RData")

# algorithms <- c("qmh-bfgs", "qmh-ls", "qmh-sr1")
# memoryLengths <- c(5, 10, 15, 20, 25, 30, 35, 40)
# sigmaU <- c("0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0")

# noSimulations <- 5
# noAlgorithms <- length(algorithms)
# noSigmaU <- length(sigmaU)
# noMemoryLengths <- length(memoryLengths)
# output <- array(0, dim = c(4, noSimulations, noAlgorithms, noSigmaU, noMemoryLengths))

# for (i in 1:(noAlgorithms)) {
#   for (j in 1:noSimulations) {
#     for (k in 1:noSigmaU) {
#       for (l in 1:noMemoryLengths) {
#         file_path <- paste("example1", paste(algorithms[i], j-1, sep="-"), sep="-")
#         file_path <- paste("M", paste(memoryLengths[l], paste(file_path), sep="/"), sep="-")
#         file_path <- paste("sigmau", paste(sigmaU[k], paste(file_path), sep="/"), sep="-")
#         file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")

#         data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
#         result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
#         settings <- read_json(paste(file_path, "/settings.json.gz", sep=""), simplifyVector = TRUE)

#         output[1:2, j, i, k, l] <- c(sigmaU[k], memoryLengths[l])
#         output[3:4, j, i, k, l] <- helper_table(data, result, settings, memLength=1)
#         print(output[, j, i, k, l])
#       }
#     }
#   }
# }
# save.image("~/src/pmmh-qn/results/example1-random-effects/example1.RData")

iactBFGS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
iactSR1 <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
iactLS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobBFGS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobSR1 <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobLS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
i <- 1
for (k in 1:noSigmaU) {
  for (l in 1:noMemoryLengths) {
    foo <- as.numeric(output[4, , 1, k, l]) * which(as.numeric(output[3, , 1, k, l]) > 0.01)
    foo <- median(as.numeric(foo), na.rm=TRUE)
    iactBFGS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 1, k, l])
    foo <- median(as.numeric(foo), na.rm=TRUE)
    aprobBFGS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[4, , 2, k, l]) * which(as.numeric(output[3, , 2, k, l]) > 0.01)
    foo <- median(as.numeric(foo), na.rm=TRUE)
    iactSR1[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 2, k, l])
    foo <- median(as.numeric(foo), na.rm=TRUE)
    aprobSR1[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[4, , 3, k, l]) * which(as.numeric(output[3, , 3, k, l]) > 0.01)
    foo <- median(as.numeric(foo), na.rm=TRUE)
    iactLS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 2, k, l])
    foo <- median(as.numeric(foo), na.rm=TRUE)
    aprobLS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    i <- i + 1
  }
}

library(ggplot2)
library(cowplot)
library(akima)
iactBFGSinterp <- interp(iactBFGS[, 1], iactBFGS[, 2], iactBFGS[, 3])
iactSR1interp <- interp(iactSR1[, 1], iactSR1[, 2], iactSR1[, 3])
iactLSinterp <- interp(iactLS[, 1], iactLS[, 2], iactLS[, 3])

gdat1 <- interp2xyz(iactBFGSinterp, data.frame=TRUE)
gdat2 <- interp2xyz(iactSR1interp, data.frame=TRUE)
gdat3 <- interp2xyz(iactLSinterp, data.frame=TRUE)

p1 <- ggplot(gdat1) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.5), guide=FALSE, direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M)) +
      theme(axis.text=element_text(size=12))

p2<- ggplot(gdat2) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.5), guide=FALSE, direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M)) +
      theme(axis.text=element_text(size=12))

p3 <- ggplot(gdat3) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.5), direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M), fill = "") +
      theme(axis.text=element_text(size=12))

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example1-qn.pdf", height = 4, width = 12)
p <- multiplot(p1, p2, p3, layout=matrix(c(rep(1, 8), rep(2, 8), rep(3, 10)), nrow=1, byrow=TRUE))
dev.off()


###############################################################################
###############################################################################

n <- length(aprobBFGS[,3])
aprob <- c(aprobBFGS[,3], aprobLS[,3], aprobSR1[,3])
tes <- c(iactBFGS[,3], iactLS[,3], iactSR1[,3])
#col <- c(rep(1, n), rep(2, n), rep(3, n))
col <- c(rep("BFGS", n), rep("LS", n), rep("SR1", n))
col <- as.factor(col)
tes_aprob <- data.frame(cbind(aprob, tes, col))
colnames(tes_aprob) <- c("aprob", "tes", "col")
tes_aprob$col <- as.factor(tes_aprob$col)

cairo_pdf("~/src/uon-papers/pmmh-memory-journal2018/draft1/images/example1-qn-aprob.pdf", height = 3, width = 4)
ggplot(tes_aprob, aes(x = aprob, y = tes, color = col, fill = col)) +
  geom_smooth(alpha = 0.25) +
  lims(x = c(0, 0.75), y = c(0, 1.25)) +
  scale_color_brewer(guide=FALSE, palette = "Dark2", name = "") +
  scale_fill_brewer(guide=FALSE, palette = "Dark2", name = "") +
  labs(x = "acceptance probability", y = "time per effective sample") +
  theme_minimal() +
  theme(axis.text=element_text(size=8), axis.title=element_text(size=9))
dev.off()
