# nSimplices: Orthogonal outlier detection and dimension estimation for improved MDS embedding of biological datasets

Conventional dimensionality reduction methods such as Multidimensional Scaling are prone to be sensitive to the presence of orthogonal outliers, leading to significant errors in the embedding. Here, we propose a robust MDS method, based on the geometry and statistics of simplices formed by data points, that allows to detect orthogonal outliers and subsequently reduce dimensionality.

# Installation

## Development and requirements

nSimplices has been developed using Python 3.8. 

# Code structure

nSimplices procedures are implemented in nsimplices.py. 
Experiment scripts are in exp_synthetic/, exp_cells/ and exp_hmp/. 
Experiment data is in data/.

# Commands to regenerate results, figures are saved in outputs/.

## Synthetic dataset

```
python3 exp_synthetic/test_synthetic_outlier_fraction.py
```
to generate Fig. 5, and 
```
python3 exp_synthetic/test_synthetic_datasets.py
```
to generate the rest


## Cell shape dataset
```
python3 exp_cells/test_cells_datasets.py 
```

## HMP dataset 
```
python3 exp_hmp/test_hmp_MDS_nSimplices.py 
```