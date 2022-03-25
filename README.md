[![DOI](https://zenodo.org/badge/465287085.svg)](https://zenodo.org/badge/latestdoi/465287085)
![Build Status](https://github.com/jameschapman19/multiviewdata/actions/workflows/python-package.yml/badge.svg)
[![downloads](https://img.shields.io/pypi/dm/multiviewdata)](https://pypi.org/project/multiviewdata/)
[![version](https://img.shields.io/pypi/v/multiviewdata)](https://pypi.org/project/multiviewdata/)
[![codecov](https://codecov.io/gh/jameschapman19/multiviewdata/branch/main/graph/badge.svg?token=kRD0CYuXsz)](https://codecov.io/gh/jameschapman19/multiviewdata)

# Multiview Data

* Experimental package to give easy access to key toy and simulated datasets from the (deep) multiview learning literature
* Feedback and contributions are welcome

## Getting Started

Datasets are imported and built with the following syntax:

```python
import os
from multiviewdata.torchdatasets import XRMB

my_dataset = XRMB(root=os.getcwd(),download=True)
```

Datasets have somewhat standardised batches. 

```python
my_dataset[0]['index'] # returns the index of the batch element
my_dataset[0]['views'] # returns a tuple/list of each view
```

Individual datasets may have additional information such as "label", "partial", and "userid".
For more information check the docs for each dataset.

## Roadmap

* option to convert torch datasets to dictionaries of numpy arrays to allow for batch methods
* additional datasets
* standardised plotting functions for each dataset?
* benchmarks?
* tensorflow versions?

## Sources

### XRMB
https://home.ttic.edu/~klivescu/XRMB_data/full/README

This directory contains data based on the University of Wisconsin X-ray Microbeam Database (referred to here as XRMB).

The original XRMB manual can be found here:  http://www.haskins.yale.edu/staff/gafos_downloads/ubdbman.pdf

We acknowledge John Westbury for providing the original data and for permitting this post-processed version to be redistributed.  The original data collection was supported (in part) by research grant number R01 DC 00820 from the National Institute of Deafness and Other Communicative Disorders, U.S. National Institutes of Health.

The post-processed data provided here was produced as part of work supported in part by NSF grant IIS-1321015.

Some of the original XRMB articulatory data was missing due to issues such as pellet tracking errors.  The data has been reconstructed in using the algorithm described in this paper:  

Wang, Arora, and Livescu, "Reconstruction of articulatory measurements with smoothed low-rank matrix completion," SLT 2014.
http://ttic.edu/livescu/papers/wang_SLT2014.pdf

The data provided here has been used for multi-view acoustic feature learning in this paper:

Wang, Arora, Livescu, and Bilmes, "Unsupervised learning of acoustic features via deep canonical correlation analysis," ICASSP 2015.
http://ttic.edu/livescu/papers/wang_ICASSP2015.pdf

If you use this version of the data, please cite the papers above.

### WIW
https://github.com/rotmanguy/DPCCA
MIT License

### Cars3d
https://github.com/llvqi/multiview_and_self-supervision
Apache License 2.0

### MNIST
https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis
Unlicensed

### MFeat


### Twitter
https://github.com/abenton/wgcca
MIT License

### CUB Image-Caption
https://github.com/iffsid/mmvae
We use Caltech-UCSD Birds (CUB) dataset, with the bird images and their captions serving as two modalities.
GNU General Public License v3.0

### MNIST-SVHN Dataset
https://github.com/iffsid/mmvae
We construct a dataset of pairs of MNIST and SVHN such that each pair depicts the same digit class. Each instance of a digit class in either dataset is randomly paired with 20 instances of the same digit class from the other dataset.
GNU General Public License v3.0
