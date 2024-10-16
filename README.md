# BindCurve
This repository contains BindCurve. Bindcurve is a lightweight python package that allows for fitting and plotting of binding curves.

## Installation
work in progress


## Basic usage
The following example demonstrates the most basic usage of bindcurve.

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

