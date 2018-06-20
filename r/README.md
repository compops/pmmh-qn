# R code

This directory contains the code used to produce the plots and tables in the paper.

All the file paths are given for Linux system where the Git repo has been cloned and placed in the sub-directory `src/pmmh-qn` in the home directory. That is, the repo has been cloned to `~/src/pmmh-qn`. Hence, the file paths need to be changed for other systems.

Furthermore, the code requires a number of packages which can be installed using the command:

``` R
install.packages(c("jsonlite", "ggplot2", "cowplot", "akima", "RColorBrewer", "xtable"))
```

The header of each R file describes the exact plot or table created for the paper by running the code. Note that these scripts only work if the Python code has been executed for the specific experiment and the results are available in the sub-directory `results/`.