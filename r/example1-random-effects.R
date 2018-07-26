###############################################################################
#
# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
#
# (c) 2018 Johan Dahlin
# uni (at) johandahlin.com
#
# Code function
# Recreates Figures 1 and 2 in the paper after running the Python code for the
# first experiment
###############################################################################

library(jsonlite)
library(ggplot2)
library(cowplot)
library(akima)

setwd("~/src/pmmh-qn")
source("~/src/pmmh-qn/r/helpers.R")

###############################################################################
# Settings
###############################################################################
output_path <- "~/src/pmmh-qn/results/example1-random-effects"
saved_workspace <- "~/src/pmmh-qn/results/example1-random-effects/example1.RData"
noSimulations <- 5
algorithms <- c("qmh-bfgs", "qmh-ls", "qmh-sr1")
memoryLengths <- c(5, 10, 15, 20, 25, 30, 35, 40)
sigmaU <- c("0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0")

###############################################################################
# Data pre-processing
###############################################################################

# Compute the IACT for each run or use the results already saved to file
if (file.exists(saved_workspace)) {

  load(saved_workspace)

} else {

  noAlgorithms <- length(algorithms)
  noSigmaU <- length(sigmaU)
  noMemoryLengths <- length(memoryLengths)
  output <- array(0, dim = c(4, noSimulations, noAlgorithms, noSigmaU, noMemoryLengths))

  for (i in 1:noAlgorithms) {
    for (j in 1:noSimulations) {
      for (k in 1:noSigmaU) {
        for (l in 1:noMemoryLengths) {
          file_path <- paste("example1", paste(algorithms[i], j-1, sep="-"), sep="-")
          file_path <- paste("M", paste(memoryLengths[l], paste(file_path), sep="/"), sep="-")
          file_path <- paste("sigmau", paste(sigmaU[k], paste(file_path), sep="/"), sep="-")
          file_path <- paste(output_path, paste(algorithms[i], paste(file_path), sep="/"), sep="/")

          data <- read_json(paste(file_path, "/data.json.gz", sep=""), simplifyVector = TRUE)
          result <- read_json(paste(file_path, "/mcmc_output.json.gz", sep=""), simplifyVector = TRUE)
          settings <- read_json(paste(file_path, "/settings.json.gz", sep=""), simplifyVector = TRUE)

          output[1:2, j, i, k, l] <- c(sigmaU[k], memoryLengths[l])
          output[3:4, j, i, k, l] <- helper_table_example1(data, result, settings, memLength=1)
          print(output[, j, i, k, l])
        }
      }
    }
  }
  save.image(saved_workspace)
}

# Compile the output and extract the IACT. Set the IACT to NAN if the acceptance
# rate is too low as this gives inaccurate IACT values.

iactBFGS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
iactSR1 <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
iactLS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobBFGS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobSR1 <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)
aprobLS <- matrix(0, nrow=noSigmaU * noMemoryLengths, ncol=3)

i <- 1
for (k in 1:noSigmaU) {
  for (l in 1:noMemoryLengths) {
    foo <- as.numeric(output[4, , 1, k, l])
    foo[which(as.numeric(output[3, , 1, k, l]) < 0.01)] = NA
    foo <- median(foo, na.rm=TRUE)
    iactBFGS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 1, k, l])
    foo <- median(foo, na.rm=TRUE)
    aprobBFGS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[4, , 2, k, l])
    foo[which(as.numeric(output[3, , 2, k, l]) < 0.01)] = NA
    foo <- median(foo, na.rm=TRUE)
    iactSR1[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 2, k, l])
    foo <- median(foo, na.rm=TRUE)
    aprobSR1[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[4, , 3, k, l])
    foo[which(as.numeric(output[3, , 3, k, l]) < 0.01)] = NA
    foo <- median(foo, na.rm=TRUE)
    iactLS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    foo <- as.numeric(output[3, , 2, k, l])
    foo <- median(foo, na.rm=TRUE)
    aprobLS[i,] <- c(as.numeric(sigmaU[k]), as.numeric(memoryLengths[l]), foo)

    i <- i + 1
  }
}


###############################################################################
# Contour maps of TSE versus M and sigma_u
###############################################################################

# Create contour plots by interpolating the data
iactBFGSinterp <- interp(iactBFGS[, 1], iactBFGS[, 2], iactBFGS[, 3])
iactSR1interp <- interp(iactSR1[, 1], iactSR1[, 2], iactSR1[, 3])
iactLSinterp <- interp(iactLS[, 1], iactLS[, 2], iactLS[, 3])
gdat1 <- interp2xyz(iactBFGSinterp, data.frame=TRUE)
gdat2 <- interp2xyz(iactLSinterp, data.frame=TRUE)
gdat3 <- interp2xyz(iactSR1interp, data.frame=TRUE)

# Create plots
p1 <- ggplot(gdat1) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.2), guide=FALSE, direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M)) #+
      #theme(axis.text=element_text(size=12))

p2<- ggplot(gdat2) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.2), guide=FALSE, direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M)) #+
      #theme(axis.text=element_text(size=12))

p3 <- ggplot(gdat3) +
      aes(x = x, y = y, z = z, fill = z) +
      geom_tile() +
      geom_contour(color = "black", alpha = 0.5) +
      scale_fill_distiller(palette="Spectral", na.value="white", limits=c(0.0, 0.2), direction=1) +
      theme_minimal() +
      labs(x = expression(sigma[u]), y = expression(M), fill = "") #+
      #theme(axis.text=element_text(size=12))

# Write to file
cairo_pdf("~/src/uon-papers/pmmh-qn/draft1/example1-qn.pdf", height = 4, width = 12)
  multiplot(p1, p2, p3, layout=matrix(c(rep(1, 8), rep(2, 8), rep(3, 10)), nrow=1, byrow=TRUE))
dev.off()


###############################################################################
# Plot of TES versus acceptance probabilty
###############################################################################

n <- length(aprobBFGS[,3])
aprob <- c(aprobBFGS[,3], aprobLS[,3], aprobSR1[,3])
tes <- c(iactBFGS[,3], iactLS[,3], iactSR1[,3])

col <- c(rep("BFGS", n), rep("LS", n), rep("SR1", n))
col <- as.factor(col)
tes_aprob <- data.frame(cbind(aprob, tes, col))
colnames(tes_aprob) <- c("aprob", "tes", "col")
tes_aprob$col <- as.factor(tes_aprob$col)

# Write to file
cairo_pdf("~/src/uon-papers/pmmh-qn/draft1/example1-qn-aprob.pdf", height = 5, width = 8)
  ggplot(tes_aprob, aes(x = aprob, y = tes, color = col, fill = col)) +
  geom_smooth(alpha = 0.25) +
  scale_color_brewer(guide=FALSE, palette = "Dark2", name = "") +
  scale_fill_brewer(guide=FALSE, palette = "Dark2", name = "") +
  labs(x = "acceptance probability", y = "time per effective sample") +
  theme_minimal() +
  coord_cartesian(ylim=c(0.0, 0.3)) #+
  #theme(axis.text=element_text(size=8), axis.title=element_text(size=9))
dev.off()


###############################################################################
# End of file
###############################################################################