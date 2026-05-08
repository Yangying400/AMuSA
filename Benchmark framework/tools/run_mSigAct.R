
library(mSigAct)
library(dplyr)

sigs <- read.table("input_siganture.csv", sep = "\t", header = TRUE, row.names = 1,check.names = FALSE)
sigs <- as.matrix(sigs)
spectra <- read.csv("input_catatlog.csv",  sep = "\t", row.names = 1, header = TRUE)
spectra <- as.matrix(spectra)
stopifnot(all(rownames(sigs) == rownames(spectra)))

output_dir <- output_dir <- "your_output"

sparse.out <- SparseAssignActivity(
  spectra = spectra,
  sigs = sigs,
  output.dir = output_dir,
  max.level = min(3, ncol(sigs) - 1),   
  p.thresh = 0.05 / ncol(sigs),
  num.parallel.samples = 10,
  mc.cores.per.sample = 4,
  seed = 1,
  max.subsets = 10000                    
)

contrib <- sparse.out$proposed.solution$exposures

contrib_df <- as.data.frame(contrib) %>%mutate_if(is.numeric, round, digits = 4)
write.csv(contrib_df,file = file.path(output_dir, "your_path"))


