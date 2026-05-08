rm(list=ls())
getwd()
setwd("/home/yangying/SigFitTest-main/code")
library(mmsig)
library(dplyr)
library(reticulate)
use_python("~/anaconda3/bin/python", required = TRUE)
py_available()

source_python("/home/yangying/SigFitTest-main/code/MS_functions.py")
source_python("/home/yangying/SigFitTest-main/code/MS_run_and_evaluate.py")
result <- fit_external(input_catalog = '/home/yangying/sbs96/ground.truth.syn.catalog.SBS96.dat', 
                       catalog_GT = '/home/yangying/sbs96/ground.truth.syn.catalog.SBS96.dat', 
                       code_name = 'EXTSPSS')

print(result)

reticulate::py_last_error()




