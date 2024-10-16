# Welcome to BindCurve
This repository contains `bindcurve` - a lightweight Python package that allows fitting and plotting of binding curves. It contains classic logistic model for fitting IC50 and logIC50, from which pIC50 could obtained. It also contains exact polynomial models for directly fitting Kd from both direct and competitive binding experiments. Fixing minimal and maximal asymptotes during fitting is supported, as well as fixing the slope in logistic models. Additionally, IC50 values can be converted to Kd using conversion models.

`bindcurve` is intended as a simple tool ideally suited for work in Jupyter notebooks or similar tools. Even if you have never used Python before, you can learn `bindcurve` easily and fit your binding curve in less than 5 lines of code. The results can be conveniently plotted with another few lines of code by matplotlib-based functions, or simply reported in formatted output.

> [!WARNING]
> `bindcurve` is currently in beta version. Changes to API might happen momentarily without notice. If you encounter bugs, please report them as Issues. 


## Installation


## Basic usage
The following example demonstrates the most basic usage of `bindcurve`.

```python

# Import bindcurve
import bindcurve as bc

# Load data from csv file
input_data = bc.load_csv("path/to/your/file.csv")

# This DataFrame will now contain all the necessary information for subsequent fitting
print(input_data)

# Fit IC50 from your data
IC50_results = bc.fit_50(input_data, model="IC50")
print(IC50_results)

# Define experimental constants
RT = 0.05           # [R]T parameter
LsT = 0.005           # [L]*T parameter
Kds = 0.0245        # Kd of the probe

# Fit Kd from your data
Kd_results = bc.fit_Kd_competition(input_data, model="comp_3st_specific", RT=RT, LsT=LsT, Kds=Kds)
print(Kd_results)
```


## Documentation
The `bindcurve` documentation can be found at https://choutkaj.github.io/bindcurve/.


## How to cite

## License

