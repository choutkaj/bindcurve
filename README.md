# Welcome to BindCurve
This repository contains `bindcurve` - a lightweight Python package that allows fitting and plotting of binding curves. It contains classic logistic model for fitting IC50 and logIC50, from which pIC50 could obtained. It also contains exact polynomial models for directly fitting Kd from both direct and competitive binding experiments. Fixing minimal and maximal asymptotes during fitting is supported, as well as fixing the slope in logistic models. Additionally, IC50 values can be converted to Kd using conversion models.

`bindcurve` is intended as a simple tool ideally suited for work in Jupyter notebooks or similar tools. Even if you have never used Python before, you can learn `bindcurve` easily and fit your binding curve in less than 5 lines of code. The results can be conveniently plotted with another few lines of code by matplotlib-based functions, or simply reported in formatted output.

> [!WARNING]
> `bindcurve` is currently in Alpha version. Changes to API might happen momentarily without notice. If you encounter bugs, please report them as Issues. 


## Installation


## Basic usage
The following example demonstrates the most basic usage of `bindcurve`. For more instructions and examples see the tutorials.

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
### Plotting curves
```python
# Import matplotlib
import matplotlib.pyplot as plt

# Initiate the plot
plt.figure(figsize=(6, 5))

# Plot your curves from the IC50_results dataframe
bc.plot(input_data, IC50_results)

# Just use matplotlib to set up show the plot 
plt.xlabel("your x label")
plt.ylabel("your y label")
plt.xscale("log")
plt.legend()
plt.show()
```


## Documentation
The `bindcurve` documentation can be found at https://choutkaj.github.io/bindcurve/.


## How to cite

## License

