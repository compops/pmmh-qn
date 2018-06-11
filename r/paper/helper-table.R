helper_table <- function(data, result, settings, memLength=1, offset=1) {

  paramsTrace <- result$params
  noIterations <- dim(paramsTrace)[1]
  noParameters <- dim(paramsTrace)[2]

  idx <- seq(offset, noIterations, memLength)
  iact <- rep(0, noParameters)

  for (k in 1:noParameters) {
    acf_res <- acf(paramsTrace[idx, k], plot = FALSE, lag.max = 250)
    iact[k] <- 1 + 2 * sum(acf_res$acf[-1])
  }

  acceptRate <- mean(result$accepted)

  if (exists('no_hessians_corrected', where=result)) {
    fracHessiansCorrected <- result$no_hessians_corrected / as.numeric(settings$no_iters)
  } else {
    fracHessiansCorrected <- 0
  }

  essPerSec <- result$time_per_iteration * memLength * mean(iact)

  c(settings$simulation_name, acceptRate, fracHessiansCorrected, mean(iact), result$time_per_iteration, essPerSec)
}