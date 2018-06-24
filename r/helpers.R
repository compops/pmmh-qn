# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

helper_table_example1 <- function(data, result, settings, memLength=1) {

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

helper_post_var <- function(result, memLength=1, offset=1) {

  paramsTrace <- result$params
  noIterations <- dim(paramsTrace)[1]
  noParameters <- dim(paramsTrace)[2]

  idx <- seq(offset, noIterations, memLength)
  post_var <- rep(0, noParameters)

  for (k in 1:noParameters) {
    post_var[k] <- var(paramsTrace[idx, k])
  }

  post_var
}