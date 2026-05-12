---
title: "A Protocol for Signature Assignment Analysis Using AMuSA"
author: "yy"
date: "2026-05-09"
output:
  word_document:
    toc: true
    toc_depth: '3'
  html_document:
    toc: true
    toc_float: true
    toc_depth: 3
    number_sections: true
  pdf_document:
    latex_engine: pdflatex
    toc: true
    toc_depth: 3
---



```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  message = FALSE
)
```

# Introduction

## Background
Mutational signatures reflect the underlying biological processes that generate somatic mutations in genomes. These signatures arise from various mechanisms, including DNA replication errors, defects in DNA repair pathways, and exposure to environmental mutagens. Identifying these signatures can provide insights into disease etiology, cancer evolution, and therapeutic strategies.
Recent large-scale sequencing projects have cataloged numerous mutational signatures across different cancer types. Computational methods have therefore become essential tools for decomposing mutation profiles into combinations of mutational signatures.

## Overview of AMuSA
AMuSA (Autoencoder Mutational Signature Assignment) is a hybrid deep learning framework that integrates unsupervised autoencoder-based denoising with supervised classifier guidance, thereby overcoming the limitations of conventional linear refitting approaches. Furthermore, AMuSA incorporates a dynamic channel-weighting mechanism to reduce bias introduced by dominant signatures, enabling accurate and joint inference of single base substitution, insertion–deletion, and doublet base substitution signatures.
AMuSA (Autoencoder Mutational Signature Assignment) is a computational framework designed to identify active mutational signatures and estimate their quantitative contributions in genomic samples. Given a mutation count matrix derived from sequencing data, AMuSA predicts which mutational signatures are active in each sample and subsequently estimates their exposure levels.
The framework supports multiple mutation categories, including single base substitutions (SBS), doublet base substitutions (DBS), and small insertions and deletions (ID). The analysis can therefore be applied to different types of mutational signature datasets.


# Data Preprocessing

## Mutation Count Matrix
AMuSA requires a mutation count matrix as input.
Each row corresponds to a mutation type and each column corresponds to a sample.
**Example Count Matrix:**

| Type      | SP.Syn.Other::S.29| SP.Syn.Kidney::S.187|SP.Syn.Prostate::S.174  |     
|-----------|-------------------|---------------------|------------------------|  
| A[C>A]A   | 11                | 114                 | 94                     |      
| A[C>A]C   | 8                 | 69                  | 58                     |       
| A[C>A]G   | 1                 | 10                  | 18                     |      
| A[C>A]T   | 6                 | 75                  | 52                     |       
| C[C>A]A   | 4                 | 81                  | 54                     |       
| C[C>A]C   | 8                 | 93                  | 74                     |       
| ...       | ...               | ...                 | ...                    |       
| T[T>G]T   | 12                | 126                 | 101                    |       

## Mutation Context Format

The structure of the mutation count matrix depends on the mutation category used for analysis. AMuSA supports multiple mutation types, including single base substitutions (SBS), doublet base substitutions (DBS), and small insertions and deletions (ID).

- **SBS (Single Base Substitutions):**  
  The standard input format consists of 96 trinucleotide mutation contexts, defined by the substitution type and the immediate 5' and 3' flanking nucleotides.

- **DBS (Doublet Base Substitutions):**  
  The mutation matrix typically contains 78 contexts, representing dinucleotide substitutions classified according to established DBS definitions.

- **ID (Insertions and Deletions):**  
  The mutation matrix typically contains 83 contexts, representing indel events categorized by type, length, and sequence context.

## Signature Reference Preparation

### COSMIC Signature Database
AMuSA uses reference mutational signatures from the COSMIC Mutational Signatures Database. These signatures were identified from large-scale analyses of cancer genomes and represent mutational processes operating in human cancers. Reference signatures for single base substitutions (SBS), doublet base substitutions (DBS), and small insertions and deletions (ID) can be used for signature assignment.

### Reference Signature Matrix
The reference signature file should be provided as a matrix of mutation probabilities, where each row corresponds to a mutation type and each column corresponds to a mutational signature.

Each column represents the probability distribution of mutation types for a given signature, and therefore should sum to 1.


**Example Signature Matrix:**

The reference signature matrix should follow the format below:

| Type      | SBS1                 | SBS2                 | SBS3                | ... |
|-----------|--------------------- |--------------------- |---------------------|-----|
| A[C>A]A   | 0.000886157230877471 | 5.80016751463797e-07 | 0.0208083233293317  | ... |
| A[C>A]C   | 0.00228040461219034  | 0.000148004274511452 | 0.0165066026410564  | ... |
| A[C>A]G   | 0.000177031410683197 | 5.23015105199252e-05 | 0.00175070028011204 | ... |
| A[C>A]T   | 0.00128022715070335  | 9.78028246433782e-05 | 0.0122048819527811  | ... |
| C[C>A]A   | 0.000312055367983941 | 0.0002080060074215   | 0.0225090036014406  | ... |
| C[C>A]C   | 0.00179031765606171  | 9.53027524387929e-05 | 0.0253101240496198  | ... |
| ...       | ...                  | ...                  | ...                 | ... |
| T[T>G]T   | 2.23039573911599e-16 | 2.23006440649012e-16 | 0.0105042016806723  | ... |

# Software Requirements

## System Requirements

Python ≥ 3.10
CUDA 12.4 (recommended, for GPU acceleration)
PyTorch ≥ 2.5.1 (GPU version recommended)
GPU is recommended for training and large-scale signature analysis. CPU-only mode is supported but slower.

##Core Dependencies

numpy == 1.26.4
pandas == 2.2.3
scikit-learn == 1.5.2
matplotlib == 3.10.0
seaborn == 0.13.2

##Installation 

pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 matplotlib==3.10.0 seaborn==0.13.2

# Signature Assignment Analysis

The AMuSA framework performs mutational signature assignment through a two-stage procedure to identify active signatures and estimate their quantitative contributions in each sample.

In the first stage, AMuSA predicts the probability of each mutational signature being active using a trained ensemble model. These probabilities are subsequently converted into binary activation states via thresholding, yielding a set of candidate active signatures.

In the second stage, the contributions of the predicted active signatures are estimated using a context-weighted non-negative least squares (WNNLS) approach. This method incorporates mutation-type-specific weights to enhance robustness, particularly under conditions of low mutation counts or high noise levels.

By integrating probabilistic prediction with constrained optimization, AMuSA provides a robust and scalable framework for accurate mutational signature assignment across diverse datasets and mutation categories.

## Installing AMuSA

git clone git@github.com:Yangying400/AMuSA.git
cd AMuSA

## run AMuSA 

### Traning Models  Run this step only if you need to train a new model,If pretrained AMuSA models are available, you can skip this step and directly proceed to Prediction.

python -m AMuSA.optimize_parameters \
  --train_mutation data/example_train_catalog.csv \
  --train_exposure data/example_train_exposure.csv \
  --test_mutation data/example_catalog.csv \
  --test_exposure data/example_exposure.csv \
  --type SBS \
  --signature_file data/ground.truth.syn.sigs.SBS96.csv \
  --output_dir result

### Arguments

--train_mutation
Path to the training mutation catalog file.
Default: data/example_train_catalog.csv
--train_exposure
Path to the training exposure file.
Default: data/example_train_exposure.csv
--test_mutation
Path to the testing mutation catalog file.
Default: data/example_catalog.csv
--test_exposure
Path to the testing exposure file.
Default: data/example_exposure.csv
--type
Type of mutational signatures used in the analysis (e.g., SBS).
Default: SBS
--signature_file
Path to the ground truth or reference signature file.
Default: data/ground.truth.syn.sigs.SBS96.csv
--output_dir
Directory to save optimization results.
Default: results/

### output 

Trained/optimized model parameters for AMuSA pipeline
Predicted vs ground-truth evaluation results
Statistical metrics for performance assessment (e.g., MAE, Pearson correlation, F1-score)
Saved results in the specified output_dir

## Run AMuSA Pipeline (Run the main AMuSA pipeline for mutational signature analysis)

### bash usage

python -m AMuSA.main \
  --base_model_dir models \
  --mutation_file data/example_catalog.csv \
  --type SBS \
  --signature_file data/ground.truth.syn.sigs.SBS96.csv \
  --output_dir result

### Python usage (Run AMuSA directly in Python or Jupyter notebooks)

from AMuSA.main import run_pipeline
run_pipeline(
    base_model_dir="models",
    mutation_file="data/example_catalog.csv",
    mutation_type="SBS",
    signature_file="data/ground.truth.syn.sigs.SBS96.csv",
    output_dir="result"
)

### Input Arguments

Model and input settings
--base_model_dir
Path to the directory containing the pretrained AMuSA models.
Default: models/
--mutation_file
Path to the mutation catalog file for signature analysis.
Default: data/example_catalog.csv
--type
Type of mutational signatures used in the analysis (e.g., SBS).
Default: SBS
--signature_file
Path to the reference or ground-truth signature file.
Default: data/ground.truth.syn.sigs.SBS96.csv  
--cosine_threshold 
controls the cutoff for identifying low-confidence samples based on cosine similarity (default: 0.9)
--adaptive_threshold
is used during refinement to filter weak signature contributions (default: 0.02).

### Pipeline Overview
AMuSA performs mutational signature assignment through a multi-step pipeline. 
First, an ensemble of pretrained autoencoder-based models and signature classifiers is used to predict signature activation probabilities for each sample. These probabilities are then converted into binary activation states using learned signature-specific thresholds (Step 1). 
Next, active signatures are used to estimate exposure levels through a channel-weighted non-negative least squares (WNNLS) , and reconstruction quality is assessed using cosine similarity (Step 2). 
Samples with cosine similarity below 0.9 are considered low-confidence cases and are reprocessed in a refinement stage. In this step, activation thresholds are adjusted to allow more candidate signatures, and a biological linkage rule is applied to enforce co-activation among related signatures. Exposure levels are then re-estimated using a channel-weighted non-negative least squares (WNNLS) approach, followed by a 2% filtering step to remove weak contributions before final normalization and reintegration with high-confidence samples.(Step 3–4)
Finally, refined results from low-confidence samples are reintegrated with high-confidence results to generate the final exposure matrix (Step 5).

### output

AMuSA predicts the probability of each mutational signature being active using an ensemble model. These probabilities are then converted into binary states via signature-specific thresholds to identify active signatures.

Thresholds are derived from the trained models. During training, each signature’s threshold is optimized to distinguish active from inactive states. In the prediction stage, thresholds from multiple models are aggregated to ensure robust classification.

This step generates three types of outputs:

##### 1. Probabilities

Predicted probability of each mutational signature being active in each sample.

**Example:**

| Signature | SP.Syn.Other::S.295 | SP.Syn.Kidney::S.187 | SP.Syn.Prostate::S.174 |
|-----------|-------------------|---------------------|------------------------|
| SBS1      | 0.95252866        | 0.9790471           | 0.96011555             |
| SBS2      | 0.94629574        | 0.20303026          | 0.19648902             |
| SBS3      | 0.054253995       | 0.098799184         | 0.108573094            |
| SBS4      | 0.01777694        | 0.006597654         | 0.008848998            |
| SBS5      | 0.9346374         | 0.8638319           | 0.95538294             |
| ...       | ...               | ...                 | ...                    |
| SBS60     | 0.015175613       | 0.1035275           | 0.07222911             |

---

##### 2. Thresholds

Optimized thresholds used to convert probabilities into binary predictions.

**Example:**

| Signature | Threshold |
|-----------|----------|
| SBS1      | 0.4      |
| SBS2      | 0.4      |
| SBS3      | 0.4      |
| SBS4      | 0.4      |
| SBS5      | 0.4      |
| ...       | ...      |
| SBS60     | 0.3619   |

---

##### 3. Predictions

Binary assignment of signatures (active = 1, inactive = 0) based on thresholds.

**Example:**

| Signature | SP.Syn.Other::S.295 | SP.Syn.Kidney::S.187 | SP.Syn.Prostate::S.174 |
|-----------|-------------------|---------------------|------------------------|
| SBS1      | 1                 | 1                   | 1                      |
| SBS2      | 1                 | 0                   | 0                      |
| SBS3      | 0                 | 0                   | 0                      |
| SBS4      | 0                 | 0                   | 0                      |
| SBS5      | 1                 | 1                   | 1                      |
| ...       | ...               | ...                 | ...                    |
| SBS60     | 0                 | 0                   | 0                      |

These outputs are used as inputs for the subsequent exposure estimation step.

#### 4.initial_exposure exposure  (WNNLS)
initial_exposure refers to the raw signature exposure matrix estimated in Step 3 using WNNLS based on the predicted active signatures before any refinement.

**Example:**

| Signature | SP.Syn.Other::S.295| SP.Syn.Kidney::S.187| SP.Syn.Prostate::S.174 |
|-----------|------------------  |---------------------|------------------------|
| SBS1      | 754                | 316                 | 245                    |
| SBS2      | 1242               | 0                   | 0                      |
| SBS3      | 0                  | 0                   | 0                      |
| SBS4      | 0                  | 0                   | 0                      |
| SBS5      | 624                | 596                 | 3014                   |
| ...       | ...                | ...                 | ...                    |
| SBS60     | 0                  | 0                   | 0                      |

#### low_cosine_catalog_file  (Cosine Similarity) 
To evaluate reconstruction quality, cosine similarity is computed between the reconstructed mutation catalog and the ground-truth catalog. This score reflects the reliability of exposure estimation for each sample and provides a confidence measure for downstream refinement. Samples with low cosine similarity are selected for further reprocessing in subsequent steps.

#### Refinement (low_cosine_predictions)

To further improve robustness, AMuSA includes an optional refinement strategy for low-confidence samples. Samples with low reconstruction quality (cosine similarity < 0.9) are considered unstable and are selected for reprocessing. These samples undergo a second-pass prediction using the ensemble model to update signature activation probabilities under relaxed constraints.
#### Refinement (refined_exposure_df)

For low-confidence samples (cosine similarity < 0.9), exposure values are recalculated using updated signature predictions from the refinement stage. The refined exposures are obtained using a second-pass WNNLS estimation with adaptive filtering and are then normalized before being reintegrated with high-confidence results to form the final exposure matrix.
#### final_exposure

final_exposure is the final signature exposure matrix obtained by replacing the exposures of low-confidence samples with their refined estimates while keeping high-confidence samples unchanged, resulting in a complete and corrected exposure matrix across all samples.

# Visualization and Interpretation

##  Dot Plot of Signature Exposure
Each point represents the exposure level of a specific signature in a given sample. The y-axis shows the number of mutations per megabase (log10-scaled), while samples are grouped and displayed as facets. Horizontal dashed lines indicate the median exposure level within each sample.

### R code

```r 
library(ggplot2)    
library(grid)        
library(gtable)      
library(dplyr)      
library(scales)      
library(reshape2)    
library(readxl)      

# Load exposure matrix
raw_data <- read.csv("AMuSA/AMuSA_example_exposure.csv")

# Transpose data: samples as rows, signatures as columns
assignment <- as.data.frame(t(raw_data[-1]))
colnames(assignment) <- raw_data[[1]]
assignment$Signature <- rownames(assignment)

# Convert to long format
data <- melt(assignment, id.vars = "Signature")
colnames(data) <- c("Signature", "Sample", "Mutation")

# Create combined label for plotting
data$Name <- paste(data$Sample, data$Signature, sep = "-")

# Remove zero and NA values
data <- data[data$Mutation > 0, ]
data$X0 <- data$Mutation / 2800  

# Compute log-scale range
data_range <- range(data$X0[data$X0 > 0], na.rm = TRUE)
min_val <- 10^floor(log10(min(data_range)))
max_val <- 10^ceiling(log10(max(data_range)))

data <- data[data$X0 > 0 & !is.na(data$X0), ]

# Compute median exposure per sample
median_values <- data %>%
  group_by(Sample) %>%
  summarise(median_X0 = median(X0, na.rm = TRUE))

# Order samples by median exposure
facet_levels <- median_values$Sample[order(median_values$median_X0)]

data$Sample <- factor(data$Sample, levels = facet_levels)
median_values$Sample <- factor(median_values$Sample, levels = facet_levels)

# Sort within each sample group
data <- data %>%
  group_by(Sample) %>%
  arrange(X0, .by_group = TRUE) %>%
  mutate(Name = factor(Name, levels = Name)) %>%
  ungroup()

# Compute number of active signatures per sample
total_samples_per_group <- nrow(assignment)  
count_data <- data %>%
  group_by(Sample) %>%
  summarise(
    Active_Signatures = n(),  
    Total_Signatures = total_samples_per_group,  
    Fraction = paste0(Active_Signatures, "/", Total_Signatures),  
    midpoint = Name[floor(n() / 2) + 1]  
  ) %>%
  ungroup()

# Background shading for facets
facet_colors <- data.frame(
  Sample = facet_levels,
  fill = rep(c("white", "lightgrey"), length.out = length(facet_levels))
)
facet_colors$Sample <- factor(facet_colors$Sample, levels = facet_levels)

# Create dot plot
p <- ggplot(data = data, aes(x = Name, y = log10(X0))) +
 
  # Background panel shading
  geom_rect(data = facet_colors,
            aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, fill = fill),
            inherit.aes = FALSE, alpha = 0.2) +
 
  # Scatter points
  geom_point(size = 1.5, position = position_jitter(width = 0.15, height = 0), 
             alpha = 0.7, color = "black") +

  # Median line per sample
  geom_hline(data = median_values, aes(yintercept = log10(median_X0)), 
             color = "red", linewidth = 0.8, linetype = "dashed") +
 
  # Log-scale axis formatting
  scale_y_continuous(
    limits = log10(c(min_val/2, max_val*2)),
    breaks = log10(10^(floor(log10(min_val/2)):ceiling(log10(max_val*2)))),
    labels = function(x) format(10^x, scientific = FALSE, drop0trailing = TRUE),
    expand = expansion(mult = c(0.1, 0.1))
  ) +
  
  # Facet by sample
  facet_grid(. ~ Sample, scales = "free_x", space = "fixed") +
  
  # Display active signature counts
  geom_text(data = count_data,
            aes(x = midpoint, y = -Inf, label = Fraction),
            vjust = -0.001, size = 3, color = "black", fontface = "bold") +
  
  scale_fill_identity() + 
  labs(x = "", y = "Number of mutations per megabase") +  

  theme(
    text = element_text(family = "Arial"),
    panel.grid = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 9, color = "black"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor.y = element_blank(),
    panel.spacing.x = unit(0.5, "lines"),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 8, angle = 100,
                              margin = margin(t = 5, r = 0, b = 5, l = 0),
                              hjust = 0.5, vjust = 0.5),
    panel.background = element_blank(),
    plot.margin = unit(c(1, 1, 1.5, 1), "cm"),
    axis.title.y = element_text(size = 9, face = "bold", margin = margin(r = 15)),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 9)
  ) +
  ggtitle("Mutation Signature Exposure Across Samples")

print(p)

```
### Result

<p align="center">
  <img src="figure/SBS.png" width="900">
</p>

## Stacked Bar Chart
This proportional stacked bar plot shows the relative contributions of different mutation types across samples, allowing comparison of compositional differences between samples or groups.

### R code

```r 
library(ggplot2)    
library(grid)        
library(gtable)      
library(dplyr)      
library(scales)      
library(reshape2)    
library(readxl)      

# Load exposure matrix
raw_data <- read.csv("AMuSA/AMuSA_example_exposure.csv")

# Transpose matrix (samples as columns → rows)
assignment <- as.data.frame(t(raw_data[-1]))
colnames(assignment) <- raw_data[[1]]
assignment$Signature <- rownames(assignment)

# Convert to long format
data <- melt(assignment, id.vars = "Signature")
colnames(data) <- c("Signature", "Sample", "Mutation")

# Create combined identifier
data$Name <- paste(data$Sample, data$Signature, sep = "-")

# Filter non-zero values
data <- data[data$Mutation > 0, ]
data$X0 <- data$Mutation / 2800  

# Compute log-scale range
data_range <- range(data$X0[data$X0 > 0], na.rm = TRUE)
min_val <- 10^floor(log10(min(data_range)))
max_val <- 10^ceiling(log10(max(data_range)))

data <- data[data$X0 > 0 & !is.na(data$X0), ]

# Compute median exposure per sample
median_values <- data %>%
  group_by(Sample) %>%
  summarise(median_X0 = median(X0, na.rm = TRUE))

# Order samples by median exposure
facet_levels <- median_values$Sample[order(median_values$median_X0)]

data$Sample <- factor(data$Sample, levels = facet_levels)
median_values$Sample <- factor(median_values$Sample, levels = facet_levels)

# Sort values within each sample
data <- data %>%
  group_by(Sample) %>%
  arrange(X0, .by_group = TRUE) %>%
  mutate(Name = factor(Name, levels = Name)) %>%
  ungroup()

# Compute number of active signatures per sample
total_samples_per_group <- nrow(assignment)  
count_data <- data %>%
  group_by(Sample) %>%
  summarise(
    Active_Signatures = n(),  
    Total_Signatures = total_samples_per_group,  
    Fraction = paste0(Active_Signatures, "/", Total_Signatures),  
    midpoint = Name[floor(n() / 2) + 1]  
  ) %>%
  ungroup()

# Background color for facets
facet_colors <- data.frame(
  Sample = facet_levels,
  fill = rep(c("white", "lightgrey"), length.out = length(facet_levels))
)
facet_colors$Sample <- factor(facet_colors$Sample, levels = facet_levels)

# Create plot
p <- ggplot(data = data, aes(x = Name, y = log10(X0))) +
 
  # Background panels
  geom_rect(data = facet_colors,
            aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, fill = fill),
            inherit.aes = FALSE, alpha = 0.2) +
 
  # Scatter points
  geom_point(size = 1.5, position = position_jitter(width = 0.15, height = 0), 
             alpha = 0.7, color = "black") +

  # Median reference line
  geom_hline(data = median_values, aes(yintercept = log10(median_X0)), 
             color = "red", linewidth = 0.8, linetype = "dashed") +
 
  # Log-scale axis
  scale_y_continuous(
    limits = log10(c(min_val/2, max_val*2)),
    breaks = log10(10^(floor(log10(min_val/2)):ceiling(log10(max_val*2)))),
    labels = function(x) format(10^x, scientific = FALSE, drop0trailing = TRUE),
    expand = expansion(mult = c(0.1, 0.1))
  ) +
  
  # Faceting
  facet_grid(. ~ Sample, scales = "free_x", space = "fixed") +
  
  # Annotation
  geom_text(data = count_data,
            aes(x = midpoint, y = -Inf, label = Fraction),
            vjust = -0.001, size = 3, color = "black", fontface = "bold") +
  
  scale_fill_identity() + 
  labs(x = "", y = "Number of mutations per megabase") +  

  theme(
    text = element_text(family = "Arial"),
    panel.grid = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 9, color = "black"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor.y = element_blank(),
    panel.spacing.x = unit(0.5, "lines"),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 8, angle = 100,
                              margin = margin(t = 5, r = 0, b = 5, l = 0),
                              hjust = 0.5, vjust = 0.5),
    panel.background = element_blank(),
    plot.margin = unit(c(1, 1, 1.5, 1), "cm"),
    axis.title.y = element_text(size = 9, face = "bold", margin = margin(r = 15)),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 9)
  ) +
  ggtitle("Mutation Signature Exposure Across Samples")

print(p)
```
### Result

<p align="center">
  <img src="figure/Mutation_Signature_Proportion.png" width="900">
</p>



## Count Bar Plot
The stacked bar plot displays both the counts and relative proportions of categories across samples.

### R code

```r 
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)

# Load exposure matrix
data <- read.csv("AMuSA/AMuSA_example_exposure.csv", 
                 header = TRUE, 
                 row.names = 1,
                 check.names = FALSE)  

# Transpose matrix (samples as rows)
data_t <- as.data.frame(t(data))
data_t$samples <- rownames(data_t)

# Convert to long format
all_data_long <- melt(data_t, id.vars = "samples", 
                      variable.name = "Signature", 
                      value.name = "Count")

# Compute global ordering of samples by total mutation count
global_sample_order <- all_data_long %>%
  group_by(samples) %>%
  summarise(total = sum(Count)) %>%
  arrange(desc(total)) %>%
  pull(samples)

# Batch configuration for plotting
batch_size <- 100
total_samples <- length(global_sample_order)
num_batches <- ceiling(total_samples / batch_size)

# Define signature color palette
signature_colors <- c(
  "SBS1" = "#1f77b4", "SBS2" = "#ff7f0e", "SBS3" = "#2ca02c", "SBS4" = "#d62728",
  "SBS5" = "#9467bd", "SBS6" = "#8c564b", "SBS7a" = "#e377c2", "SBS7b" = "#7f7f7f",
  "SBS7c" = "#bcbd22", "SBS7d" = "#17becf", "SBS8" = "#aec7e8", "SBS9" = "#ffbb78",
  "SBS10a" = "#98df8a", "SBS10b" = "#ff9896", "SBS12" = "#c5b0d5", "SBS13" = "#c49c94",
  "SBS14" = "#f7b6d2", "SBS15" = "#c7c7c7", "SBS16" = "#dbdb8d", "SBS17a" = "#9edae5",
  "SBS17b" = "#ad494a", "SBS18" = "#8c6d31", "SBS19" = "#843c39", "SBS20" = "#6b6ecf",
  "SBS21" = "#e7ba52", "SBS22" = "#ce6dbd", "SBS23" = "#3182bd", "SBS24" = "#e6550d",
  "SBS26" = "#31a354", "SBS28" = "#756bb1", "SBS29" = "#636363", "SBS30" = "#bd9e39",
  "SBS31" = "#6baed6", "SBS32" = "#fd8d3c", "SBS33" = "#74c476", "SBS34" = "#9e9ac8",
  "SBS35" = "#969696", "SBS36" = "#5254a3", "SBS37" = "#bd9e39", "SBS38" = "#6baed6",
  "SBS39" = "#fd8d3c", "SBS40" = "#74c476", "SBS41" = "#9e9ac8", "SBS44" = "#969696",
  "SBS45" = "#5254a3", "SBS51" = "#3182bd", "SBS52" = "#e6550d", "SBS54" = "#31a354",
  "SBS56" = "#FF6B6B", "SBS58" = "#4ECDC4", "SBS60" = "#FFE66D"
)

# Loop over batches for visualization
for (batch in 1:num_batches) {
  
  # Define batch range
  start_idx <- (batch - 1) * batch_size + 1
  end_idx <- min(batch * batch_size, total_samples)
  
  # Select samples for current batch
  batch_samples <- global_sample_order[start_idx:end_idx]

  batch_data_long <- all_data_long %>% 
    filter(samples %in% batch_samples, Count > 0) %>%
    mutate(samples = factor(samples, levels = batch_samples))  

  # Identify signatures present in this batch
  existing_signatures <- unique(batch_data_long$Signature)
  batch_colors <- signature_colors[names(signature_colors) %in% existing_signatures]

  # Compute y-axis scaling
  batch_max_y <- max(batch_data_long %>% 
                       group_by(samples) %>% 
                       summarise(total = sum(Count)) %>% 
                       pull(total))

  batch_min_y <- min(batch_data_long %>% 
                       group_by(samples) %>% 
                       summarise(total = sum(Count)) %>% 
                       pull(total))

  y_breaks <- pretty(c(0, batch_max_y * 1.05), n = 8)

  # Create stacked bar plot
  p <- ggplot(batch_data_long, aes(x = samples, y = Count, fill = Signature)) +
    geom_bar(stat = "identity", position = "stack", width = 0.9, color = NA) +
    scale_fill_manual(values = batch_colors) +
    labs(
      title = paste("Number of mutations in each signature (Samples", start_idx, "-", end_idx, ")"),
      x = "Samples",
      y = "Number of mutations"
    ) +
    theme_classic() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8, color = "black"),
      axis.text.y = element_text(size = 8, color = "black"),
      axis.title = element_text(size = 9, face = "bold", color = "black"),
      plot.title = element_text(size = 9, face = "bold", hjust = 0.5, color = "black", margin = margin(b = 10)),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 9),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.5, "cm"),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      axis.line = element_line(color = "black", linewidth = 0.5)
    ) +
    scale_y_continuous(
      limits = c(0, max(y_breaks)),  
      breaks = y_breaks,             
      labels = scales::comma,        
      expand = expansion(mult = c(0, 0.05)) 
    ) +
    scale_x_discrete(expand = expansion(mult = c(0.01, 0.01))) +
    guides(fill = guide_legend(ncol = 1, keyheight = 0.4, keywidth = 0.4))

  # Print plot
  print(p)
}
```
### Result

<p align="center">
  <img src="figure/Mutation_siganture.png" width="900">
</p>

