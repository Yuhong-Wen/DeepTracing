# DeepTracing: Lineage-aware generative representation learning for single-cell lineage tracing data

`DeepTracing` is a PyTorch-based deep generative framework for integrating single-cell RNA sequencing (scRNA-seq) and lineage-tracing information. Built on a variational autoencoder (VAE), DeepTracing uses distance-informed Gaussian-process (GP) priors to model lineage-associated variation and can incorporate additional temporal or spatial measurements. The framework learns representations of transcriptional and lineage-associated variation and reconstructs gene expression through a probabilistic decoder.

\---

## Introduction

!\[DeepTracing framework](figs/DeepTracing.png)

**Overview of the DeepTracing framework.**  
**(A)** Deep generative modeling of scRNA-seq data with lineage tracing. Gene-expression measurements and pairwise lineage distances are used together with optional augmented measurements, such as sampling time or spatial location. A probabilistic encoder learns latent variables representing intrinsic, lineage-associated, and augmented variation, which are used by a probabilistic decoder to reconstruct gene expression.  
**(B)** The basic amortized variational autoencoder architecture. Encoder-derived latent distributions are combined with lineage-distance-informed GP kernel constraints, including sparse GP regression for lineage variables, before decoding and reconstruction.

\---

## Installation and Usage

**1. Clone this repository**

```bash
git clone https://github.com/Yuhong-Wen/DeepTracing.git
cd DeepTracing
```

The repository includes source code in `src/`, a runnable example in `example/`, and dataset-specific analyses in `experiments/`.

**2. Create the Conda environment**

An `environment.yml` file is provided in the repository root. Create the environment with:

```bash
conda env create -f environment.yml
```

**3. Activate the environment**

```bash
conda activate DeepTracing
```

**4. Run the example**

The `example/` directory provides an example of running DeepTracing on simulated lineage-tracing data:

```bash
cd example
python run\_DeepTracing\_TedSim.py
```

\---

## Experiments

The `experiments/` directory contains dataset-specific implementations and analyses:

|Directory|Description|
|-|-|
|[`experiments/TedSim/`](experiments/TedSim/)|Representation learning and evaluation on simulated lineage-tracing data.|
|[`experiments/Tumor/`](experiments/Tumor/)|Analysis of tumor lineage-tracing datasets.|
|[`experiments/eTracer/`](experiments/eTracer/)|Temporal analysis of tumor evolution and immune evasion using lineage-tracing and scRNA-seq data.|
|[`experiments/KPspatial/`](experiments/KPspatial/)|Integration and analysis of spatial measurements with lineage-tracing and transcriptional data.|

Each directory contains code for the corresponding dataset and analysis. Please refer to the scripts in each directory for dataset-specific processing and execution steps.

