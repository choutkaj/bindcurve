import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import calculate
#import data
#import models

import bindcurve as bc



###
# Direct binding
###

input_data = bc.load_csv("examples/direct.csv")
print(input_data)


compounds = input_data["compound"].unique()
print("Detected compounds:", compounds)
print("No of compounds:", len(compounds))

min = 0
max = 1
LsT = 5       # [L]*T parameter

Ns = 0.5

dir_results_simple = bc.fit_Kd_direct(input_data, model="dir_simple", LsT=LsT, ci=True)
print(dir_results_simple)

dir_results_specific = bc.fit_Kd_direct(input_data, model="dir_specific", LsT=LsT, ci=True)
print(dir_results_specific)

dir_results_total = bc.fit_Kd_direct(input_data, model="dir_total", LsT=LsT, Ns=Ns, ci=True)
print(dir_results_total)


plt.figure(figsize=(6, 5))

bc.plot(input_data, dir_results_simple, single_color="red", single_label="dir_results_simple")
bc.plot(input_data, dir_results_specific, single_color="green", single_label="dir_results_specific")
bc.plot(input_data, dir_results_total, single_color="blue", single_label="dir_results_total")

bc.plot_asymptotes(dir_results_simple)
bc.plot_traces(dir_results_simple, value="Kds")
bc.plot_value(dir_results_simple, value="Kds", color="black", marker="o")

plt.xlabel("Log concentration")
plt.ylabel("Response")
plt.xscale("log")
plt.legend()
plt.show()






###
# competitive binding
###

input_data = bc.load_csv("examples/competitive.csv")
print(input_data)


compounds = input_data["compound"].unique()
print("Detected compounds:", compounds)
print("No of compounds:", len(compounds))


RT = 0.05           # [R]T parameter
LsT = 0.005           # [L]*T parameter
Kds = 0.0245        # Kd of the probe

N = 5


IC50_results = bc.fit_50(input_data, model="IC50")
#IC50_results = IC50_results.sort_values(by=['IC50'], ascending=False)
print(IC50_results)

logIC50_results = bc.fit_50(input_data, model="logIC50")
print(logIC50_results)

"""
Kd_results_3st_specific = bc.fit_Kd_competition(input_data, model="comp_3st_specific", RT=RT, LsT=LsT, Kds=Kds)
print(Kd_results_3st_specific)

Kd_results_3st_total = bc.fit_Kd_competition(input_data, model="comp_3st_total", RT=RT, LsT=LsT, Kds=Kds, N=N)
print(Kd_results_3st_total)

Kd_results_4st_specific = bc.fit_Kd_competition(input_data, model="comp_4st_specific", RT=RT, LsT=LsT, Kds=Kds, fix_ymin=25)
print(Kd_results_4st_specific)

Kd_results_4st_total = bc.fit_Kd_competition(input_data, model="comp_4st_total", RT=RT, LsT=LsT, Kds=Kds, N=N, fix_ymin=25)
print(Kd_results_4st_total)


plt.figure(figsize=(6, 5))

bc.plot(input_data, IC50_results, single_color="black", single_label="IC50_results")
#data.plot(input_data, logIC50_results, single_color="grey", single_label="logIC50_results")

bc.plot(input_data, Kd_results_3st_specific, single_color="red", single_label="Kd_results_3st_specific")
bc.plot(input_data, Kd_results_3st_total, single_color="green", single_label="Kd_results_3st_total")
bc.plot(input_data, Kd_results_4st_specific, single_color="blue", single_label="Kd_results_4st_specific")
bc.plot(input_data, Kd_results_4st_total, single_color="purple", single_label="Kd_results_4st_total")

plt.xlabel("Log concentration")
plt.ylabel("Response")
plt.xscale("log")
plt.legend()
plt.show()
"""





###
# plotting grid
###

bc.plot_grid(input_data, IC50_results, compound_sel=False, n_cols=3, figsize=(6,4), single_color="tomato", show_title=False, show_legend=False,
               show_all_data=True, show_medians=False, show_errorbars=False,
               markersize=2,
               hspace=0, wspace=0, show_inner_ticklabels=False)



exit()









"""
###
# plotting components - saturation
###

RT = 5           # [R]T parameter
LsT = 5           # [L]*T parameter
Kds = 100        # Kd of the probe

min = 0
max = 1

N = 10

x_curve = np.logspace(np.log10(0.001), np.log10(10000), 1000)

model = bc.dir_specific(x_curve, min, max, LsT, Kds)[0]
R = bc.dir_specific(x_curve, min, max, LsT, Kds)[1]
RLs = bc.dir_specific(x_curve, min, max, LsT, Kds)[2]
Ls = bc.dir_specific(x_curve, min, max, LsT, Kds)[3]

plt.axhline(y = LsT, label="LsT", linestyle = '--')
#plt.plot(x_curve, model, label="model", linestyle="-")
#plt.plot(x_curve, R, label="R", linestyle="-")
plt.plot(x_curve, RLs, label="RLs", linestyle="-")
plt.plot(x_curve, Ls, label="Ls", linestyle="-")
#plt.plot(x_curve, L, label="L", linestyle="-")

plt.xscale("log")
#plt.yscale("log")
plt.legend()
plt.show()



###
# plotting components - competition
###

RT = 0.5           # [R]T parameter
LsT = 5           # [L]*T parameter
Kds = 1000        # Kd of the probe

min = 0
max = 1

Kd = 1000

N = 0

x_curve = np.logspace(np.log10(0.001), np.log10(10000), 1000)

model = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[0]
R = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[1]
RLs = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[2]
RL = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[3]
Ls = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[4]
L = bc.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[5]

plt.axhline(y = LsT, label="LsT", linestyle = '--')
plt.axhline(y = RT, label="RT", linestyle = '--')
#plt.plot(x_curve, model, label="model", linestyle="-")
plt.plot(x_curve, R, label="R", linestyle="-")
plt.plot(x_curve, RLs, label="RLs", linestyle="-")
plt.plot(x_curve, RL, label="RL", linestyle="-")
plt.plot(x_curve, Ls, label="Ls", linestyle="-")
#plt.plot(x_curve, L, label="L", linestyle="-")

#plt.xscale("log")
#plt.yscale("log")
plt.legend()
plt.show()


exit()
"""



###
# plotting models
###
"""
min = 0
max = 1

Kd = 1
Kd3 = 0.1

N = 0.5
Ns = 0


x_curve = np.logspace(np.log10(0.001), np.log10(1000000), 1000)

y_curve_sat_simple = models.dir_simple(x_curve, min, max, Kds)
y_curve_sat_quadratic_specific = models.dir_specific(x_curve, min, max, LsT, Kds)
y_curve_sat_quadratic_total = models.dir_total(x_curve, min, max, LsT, Kds, Ns)


y_curve_cubic_specific = models.comp_3st_specific(x_curve, min, max, RT, LsT, Kds, Kd)[0]
y_curve_cubic_total = models.comp_3st_total(x_curve, min, max, RT, LsT, Kds, Kd, N)[0]

y_curve_quintic_specific = models.comp_4st_specific(x_curve, min, max, RT, LsT, Kds, Kd, Kd3)
y_curve_quintic_total = models.comp_4st_total(x_curve, min, max, RT, LsT, Kds, Kd, Kd3, N)


plt.plot(x_curve, y_curve_cubic_specific, label="y_curve_cubic_specific", linestyle="-")
plt.plot(x_curve, y_curve_cubic_total, label="y_curve_cubic_total", linestyle="--")

plt.plot(x_curve, y_curve_quintic_specific, label="y_curve_quintic_specific", linestyle="-")
plt.plot(x_curve, y_curve_quintic_total, label="y_curve_quintic_total", linestyle="--")

plt.xscale("log")
plt.legend()
plt.show()


exit()
"""














