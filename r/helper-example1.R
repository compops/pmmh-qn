helper_table <- function(data, result, settings, memLength=1) {

  paramsTrace <- result$params
  noIterations <- dim(paramsTrace)[1]
  noParameters <- dim(paramsTrace)[2]

  idx <- seq(1, noIterations, memLength)
  iact <- rep(0, noParameters)

  for (k in 1:noParameters) {
    acf_res <- acf(paramsTrace[idx, k], plot = FALSE, lag.max = 250)
    iact[k] <- 1 + 2 * sum(acf_res$acf[-1])
  }

  acceptRate <- mean(result$accepted)

  essPerSec <- result$time_per_iteration * memLength * mean(iact)

  c(acceptRate, essPerSec)
}

library("RColorBrewer")

helper_plotting <- function(data, result, settings, algorithm, noItersToPlot,
                            savePlotToFile, paramsScale, folderToSaveTo,
                            xrange, yrange, memory_length = 1) {

  plotColors = brewer.pal(8, "Dark2");
  plotColors = c(plotColors, plotColors)

  paramsNames <- c(expression(mu),
                   expression(phi),
                   expression(sigma[v]),
                   expression(rho))
  paramsNamesACF <- c(expression("ACF of " * mu),
                      expression("ACF of " * phi),
                      expression("ACF of " * sigma[v]),
                      expression("ACF of " * rho))

  obs <- data$observations
  noIters <- settings$no_iters - settings$no_burnin_iters

  paramsTrace <- result$params
  statesTrace <- result$state_trajectory

  # Estimate the posterior mean and the corresponding standard deviation
  paramsEstMean   <- colMeans(paramsTrace)
  paramsEstStDev  <- apply(paramsTrace, 2, sd)

  # Estimate the log-volatility and the corresponding standad deviation
  statesEstMean    <- colMeans(statesTrace)
  statesEstStDev   <- apply(statesTrace, 2, sd)

  # Plot the parameter posterior estimate, solid black line indicate posterior mean
  # Plot the trace of the Markov chain after burn-in, solid black line indicate posterior mean
  if (savePlotToFile) {
    fileName <- paste(folderToSaveTo, paste(algorithm, ".pdf", sep=""), sep="")
    cairo_pdf(fileName, height = 10, width = 8)}

  if (length(paramsEstMean) == 3) {
    layout(matrix(c(1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 5, 3, byrow = TRUE))
  }
  if ((length(paramsEstMean) == 3) && (sum(statesEstMean) == 0)) {
    layout(matrix(c(1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 4, 3, byrow = TRUE))
  }

  if (length(paramsEstMean) == 4) {
    layout(matrix(c(1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 6, 3, byrow = TRUE))
  }
  par(mar = c(4, 5, 0, 0))

  # Grid for plotting the data state estimates
  yGrid <- seq(1, length(obs))
  xGrid <- seq(1, length(obs) - 1)

  #---------------------------------------------------------------------------
  # Observations
  #---------------------------------------------------------------------------
  plot(
    obs,
    col = plotColors[1],
    lwd = 1,
    type = "l",
    xlab = "time",
    ylab = "observations",
    ylim = yrange,
    xlim = c(0, ceiling(length(obs) / 10) * 10),
    bty = "n"
  )
  polygon(
    c(yGrid, rev(yGrid)),
    c(obs, rep(min(yrange), length(yGrid))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[1])) / 256, alpha = 0.25)
  )
  offset <- diff(1.2 * range(obs)) * 0.1
  x <- ceiling(length(obs) / 10) * 10
  y1 <- yrange[2] * 0.9
  y2 <- yrange[2] * 0.75
  y3 <- yrange[2] * 0.6

  text(x, y1, pos=2, labels=algorithm)
  text(x, y2, pos=2, labels=paste("acc. prob:", round(mean(result$accepted), 2)))
  text(x, y3, pos=2, labels=paste("frac. hess. corr:", round(result$no_hessians_corrected/settings$no_iters, 2)))

  #---------------------------------------------------------------------------
  # Log-volatility
  #---------------------------------------------------------------------------
  if (sum(statesEstMean) != 0) {
    statesEstUpperCI <- statesEstMean[-1] + 1.96 * statesEstStDev[-1]
    statesEstLowerCI <- statesEstMean[-1] - 1.96 * statesEstStDev[-1]
    plot(
      statesEstMean[-1],
      col = plotColors[2],
      lwd = 1.5,
      type = "l",
      xlab = "time",
      ylab = "state estimate",
      ylim = xrange,
      xlim = c(0, ceiling(length(statesEstMean[-1]) / 10) * 10),
      bty = "n"
    )

    polygon(
      c(xGrid, rev(xGrid)),
      c(statesEstUpperCI, rev(statesEstLowerCI)),
      border = NA,
      col = rgb(t(col2rgb(plotColors[2])) / 256, alpha = 0.25)
    )
  }
  #---------------------------------------------------------------------------
  # Parameter posteriors
  #---------------------------------------------------------------------------

    paramsScale <- matrix(paramsScale, nrow = length(paramsEstMean), ncol = 2, byrow = TRUE)
  iact <- c()

  for (k in 1:length(paramsEstMean)) {
    idx <- seq(1, min(memory_length*noItersToPlot, noIters), memory_length)
    grid <- seq(1, length(idx), 1)

    # Histogram of the posterior
    hist(
      paramsTrace[, k],
      breaks = floor(sqrt(noIters)),
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25),
      border = NA,
      xlab = paramsNames[k],
      ylab = "posterior estimate",
      main = "",
      xlim = paramsScale[k,],
      freq = FALSE
    )

    # Add lines for the kernel density estimate of the posterior
    kde <- density(paramsTrace[, k],
                   kernel = "e",
                   from = paramsScale[k, 1],
                   to = paramsScale[k, 2])
    lines(kde, lwd = 2, col = plotColors[k+2])

    # Plot the estimate of the posterior mean
    abline(v = paramsEstMean[k], lwd = 1, lty = "dotted")

    # Add lines for prior
    prior_grid <- seq(paramsScale[k, 1], paramsScale[k, 2], 0.01)
    if (k==1) {prior_values = dnorm(prior_grid, 0, 1)}
    if (k==2) {prior_values = dnorm(prior_grid, 0.5, 1)}
    if (k==3) {prior_values = dgamma(prior_grid, 2.0, 2.0)}
    if (k==4) {prior_values = dnorm(prior_grid, 0.0, 1.0)}
    lines(prior_grid, prior_values, col = "darkgrey")

    # Plot trace of the Markov chain
    plot(
      grid,
      paramsTrace[idx, k],
      col = plotColors[k+2],
      type = "l",
      xlab = "iteration",
      ylab = paramsNames[k],
      ylim = paramsScale[k,],
      bty = "n"
    )
    polygon(
      c(grid, rev(grid)),
      c(paramsTrace[idx, k], rep(paramsScale[k,1], length(grid))),
      border = NA,
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25)
    )
    abline(h = paramsEstMean[k], lwd = 1, lty = "dotted")

    if (length(result$iter_hessians_corrected) > 0) {
      rug(result$iter_hessians_corrected, ticksize = 0.1)
    }

    # Plot the autocorrelation function
    max_lag <- 100
    acf_res <- acf(paramsTrace[idx, k], plot = FALSE, lag.max = max_lag)
    plot(
      acf_res$lag,
      acf_res$acf,
      col = plotColors[k+2],
      type = "l",
      xlab = "iteration",
      ylab = paramsNamesACF[k],
      lwd = 2,
      ylim = c(-0.2, 1),
      bty = "n"
    )
    polygon(
      c(acf_res$lag, rev(acf_res$lag)),
      c(acf_res$acf, rep(0, length(acf_res$lag))),
      border = NA,
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25)
    )
    abline(h = 1.96 / sqrt(noIters), lty = "dotted")
    abline(h = -1.96 / sqrt(noIters), lty = "dotted")

    sig_acf_coef <- which(acf_res$acf < 0)[1]
    if (is.na(sig_acf_coef)) {sig_acf_coef <- max_lag}
    new_iact <- 1 + 2 * sum(acf_res$acf[1:sig_acf_coef])
    iact <- c(iact, new_iact)
    text(max_lag, 0.9, pos=2, labels=paste("IACT:", round(new_iact, 2)))
  }

  if (savePlotToFile) {dev.off()}

  iact
  }