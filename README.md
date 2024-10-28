[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14001254.svg)](https://doi.org/10.5281/zenodo.14001254)

# CoDAXS - Compositional Data Analysis from XRF scanning

A Jupyter Notebook to process and perform basic compositional data analysis on XRF-scan count data. 

This code should provide a state of the art data-handling and visualization of XRF core scanning data. Elemental counts are centred-log-ratio transformed. Singular value decomposition (SVD) is performed to reduce dimensions and provide deeper insights into the covariance between different elements. Additionally the automatic classification by clustering allows to identify scanned intervals with distinct geochemistry. The details about clr-transformation, SVD and how to interpret biplots of compositional data can be found in Aitchison and Greenacre, 2002. 


## Setup
The Jupyter Notebook runs on Python 3, with `pandas`, `numpy`, `matplotlib` and `pathlib`.

Additional requirements for the CoDAXRF_module: `scipy` and `scikit-learn`.

Install the required packages:
```
pip install -r requirements.txt
```

or 

```
pip install pandas numpy matplotlib scipy scikit-learn
```

A path for the input folder (containing the XRF count data as .csv) and output folder should be defined. An example file is provided in the input folder of this package.


## Usage
Make sure the input data went through quality control (i.e. determine noise levels/zero measurements) before you do any analysis with it and make interpretations using the CoDAXS package. 
Provide your XRF-scan data as positive count data and run the different cells of the CoDAXS jupyter notebook from top to bottom.
All figures and dataframes generated are stored in the output folder with name of the input file.

## Acknowlegedments
We would like to thank Katleen Wils, Tobias Schwestermann, Samuel Barrett, Steven Huang and Arne Ramisch for their input, which helped us to put together the CoDAXS package.


CoDAXS © 2024 by Markus Niederstätter, Marcel Ortler and Ariana Molenaar is licensed under CC BY 4.0 
