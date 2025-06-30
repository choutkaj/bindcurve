# Welcome to BindCurve

[![License: MIT](https://img.shields.io/github/license/choutkaj/bindcurve)](https://github.com/choutkaj/bindcurve/blob/main/LICENSE)
[![Pylint](https://github.com/choutkaj/bindcurve/actions/workflows/pylint.yml/badge.svg)](https://github.com/choutkaj/bindcurve/actions/workflows/pylint.yml)



This repository contains `bindcurve` - a lightweight Python package for fitting and plotting of binding curves (dose-response curves). It contains the logistic model for fitting $`\text{IC}_{50}`$ or $`\text{logIC}_{50}`$ and also  exact polynomial models for fitting $K_d$ from both direct and competitive binding experiments. Fixing lower and upper asymptotes of the models during fitting is supported, as well as fixing the slope in the logistic model. Additionally, $`\text{IC}_{50}`$ values can be converted to $K_d$ using conversion models.

`bindcurve` is intended as a simple tool for Python-based workflows in Jupyter notebooks or similar tools. Even if you have never used Python before, you can fit your binding curve in less than 5 lines of code. The results can be conveniently plotted with another few lines of code or simply reported in formatted output.



## Documentation
The `bindcurve` documentation can be found at https://choutkaj.github.io/bindcurve/.

## Installation
`bindcurve` is installed from pip using

```python
pip install bindcurve
```

If you want to upgrade to the latest version, use

```python
pip install --upgrade bindcurve
```

## Basic usage
`bindcurve` contains functions that are executed directly on Pandas DataFrames, which are used to store the data. The following example demonstrates the most basic usage. See the tutorials for more instructions and examples.

### Fitting
```python
# Import bindcurve
import bindcurve as bc

# Load data from csv file
input_data = bc.load_csv("path/to/your/file.csv")

# This DataFrame will now contain preprocessed input data
print(input_data)

# Fit IC50 from your data
IC50_results = bc.fit_50(input_data, model="IC50")
print(IC50_results)

# Fit Kd from your data
Kd_results = bc.fit_Kd_competition(input_data, model="comp_3st_specific", RT=0.05, LsT=0.005, Kds=0.0245)
print(Kd_results)
```
### Plotting
```python
# Import matplotlib
import matplotlib.pyplot as plt

# Initiate the plot
plt.figure(figsize=(6, 5))

# Plot your curves from the IC50_results dataframe
bc.plot(input_data, IC50_results)

# Just use matplotlib to set up and show the plot 
plt.xlabel("your x label")
plt.ylabel("your y label")
plt.xscale("log")
plt.legend()
plt.show()
```



## How to cite
No dedicated DOI is available yet. Please cite this repository.

## License

`bindcurve` is published under the MIT license. 
