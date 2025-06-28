import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from bindcurve import models


def load_csv(csvfile, c_scale=1):
    """Loads and preprocesses data from csv file.
    
    Parameters
    ----------
    csvfile : str
        Path to the csv file.
    c_scale : float or int, optional
        Factor for scaling concentration. Used for unit conversion.
    
    Returns
    -------
    DataFrame
        Pandas DataFrame containing preprocessed input data.
    """

    print("Loading data from", csvfile)
    
    # Loading input .csv file to pandas
    df = pd.read_csv(csvfile)
    
    # Renaming columns to standard names
    df.columns.values[0] = "compound"
    df.columns.values[1] = "c"
    
    df["c"] = df["c"] * c_scale
    
    # Adding a column log c  
    df.insert(loc=2, column='log c', value=np.log10(df['c']))
    # Adding a column n_replicates 
    df.insert(loc=3, column='n_reps', value=df.count(axis=1, numeric_only=True)-2)
    # Adding median, SD, and SEM columns
    df["median"] = df.iloc[:, 4:].median(numeric_only=True, axis=1)
    df["SD"] = df.iloc[:, 4:-1].std(numeric_only=True, axis=1)   
    df["SEM"] = df.SD/np.sqrt(df.n_reps)
    
    return df


def load_df(df, c_scale=1):
    """Loads and preprocesses data from existing DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame object with data.
    c_scale : float or int, optional
        Factor for scaling concentration. Used for unit conversion.
    
    Returns
    -------
    DataFrame
        Pandas DataFrame containing preprocessed input data.
    """
    
    print("Loading data from", df)
    
    # Renaming columns to standard names
    df.columns.values[0] = "compound"
    df.columns.values[1] = "c"
    
    df["c"] = df["c"] * c_scale
    
    # Adding a column log c  
    df.insert(loc=2, column='log c', value=np.log10(df['c']))
    # Adding a column n_replicates 
    df.insert(loc=3, column='n_reps', value=df.count(axis=1, numeric_only=True)-2)
    # Adding median, SD, and SEM columns
    df["median"] = df.iloc[:, 4:].median(numeric_only=True, axis=1)
    df["SD"] = df.iloc[:, 4:-1].std(numeric_only=True, axis=1)   
    df["SEM"] = df.SD/np.sqrt(df.n_reps)

    return df


def pool_data(df):
      
    # Creating empty list
    pooled_list = []

    # Iterating through df and pooling data into pooled_list
    for index, row in df.iterrows():
        for i in row.iloc[4:-3]:
            pooled_list.append([row.iloc[0], row.iloc[1], row.iloc[2], i])


    # Creating pandas dataframe from pooled_list
    pooled_df = pd.DataFrame(pooled_list, columns=['compound', 'c', 'log c', 'response'])
    
    # Removing rows with NaN 
    pooled_df = pooled_df.dropna(axis=0, how='any')
    
    return pooled_df



def fetch_pars(df, x_curve):
    pars = lmfit.Parameters()   
    
    if df['model'].iloc[0] == "IC50":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('IC50', value = float(df['IC50'].iloc[0]))
        pars.add('slope', value = float(df['slope'].iloc[0]))
        y_curve = models.IC50_lmfit(pars, x_curve)
    
    if df['model'].iloc[0] == "logIC50":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('logIC50', value = float(df['logIC50'].iloc[0]))
        pars.add('slope', value = float(df['slope'].iloc[0]))
        y_curve = models.logIC50_lmfit(pars, x_curve)
        
    if df['model'].iloc[0] == "dir_simple":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        y_curve = models.dir_simple_lmfit(pars, x_curve)
        
    if df['model'].iloc[0] == "dir_specific":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        y_curve = models.dir_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "dir_total":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Ns', value = float(df['Ns'].iloc[0]))
        y_curve = models.dir_total_lmfit(pars, x_curve)         
      
    if df['model'].iloc[0] == "comp_3st_specific":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        y_curve = models.comp_3st_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "comp_3st_total":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('N', value = float(df['N'].iloc[0]))
        y_curve = models.comp_3st_total_lmfit(pars, x_curve)               
                     
    if df['model'].iloc[0] == "comp_4st_specific":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('Kd3', value = float(df['Kd3'].iloc[0]))
        y_curve = models.comp_4st_specific_lmfit(pars, x_curve)         
        
    if df['model'].iloc[0] == "comp_4st_total":
        pars.add('ymin', value = float(df['ymin'].iloc[0]))
        pars.add('ymax', value = float(df['ymax'].iloc[0]))
        pars.add('RT', value = float(df['RT'].iloc[0]))
        pars.add('LsT', value = float(df['LsT'].iloc[0]))        
        pars.add('Kds', value = float(df['Kds'].iloc[0]))
        pars.add('Kd', value = float(df['Kd'].iloc[0]))
        pars.add('Kd3', value = float(df['Kd3'].iloc[0]))
        pars.add('N', value = float(df['N'].iloc[0]))
        y_curve = models.comp_4st_total_lmfit(pars, x_curve)              
                       
    return y_curve


def plot(input_df, results_df, compound_sel=False, xmin=False, xmax=False, 
         marker="o", 
         markersize=5, 
         linewidth=1, 
         linestyle="-",
         show_medians=True,
         show_all_data=False, 
         show_errorbars=True, 
         errorbars_kind="SD", 
         errorbar_linewidth = 1, 
         errorbar_capsize=3, 
         cmap="tab10", 
         cmap_min = 0, 
         cmap_max = 1, 
         custom_colors=False, 
         single_color=False, 
         custom_labels=False,
         single_label=False,
         no_labels=False):
    """Plots one or more curves into an initiated matplotlib plot.
    
    Parameters
    ----------
    input_df : DataFrame
        Pandas DataFrame containing the input data.
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds from the results_df will be used.
    xmin : float or int
        Manually set the minimum value on x axis for all curves. If set to False, it will be determined automatically for each curve.
    xmax : float or int
        Manually set the maximum value on x axis for all curves. If set to False, it will be determined automatically for each curve.
    marker : str
        Marker type. Any matplotlib syntax is accepted (see https://matplotlib.org/stable/api/markers_api.html).
    markersize : float or int
        Marker size.
    linewidth : float or int
        Line width of the curve.
    linestyle : str
        Line style of the curve.
    show_medians : bool
        Whether to show concentration median values.
    show_all_data : bool
        Whether to show all concentration points.
    show_errorbars : bool
        Whether to show errorbars.
    errorbars_kind : str
        What should be shown as errorbars, options are "SD" or "SEM".
    errorbar_linewidth : float or int
        Line width of the errorbars. 
    errorbar_capsize : float or int
        Size of the errorbar caps (upper and lower points of the bars).
    cmap : str
        What cmap to use for coloring the curves. Any matplotlib syntax is accepted (see https://matplotlib.org/stable/gallery/color/colormap_reference.html).
    cmap_min : float or int
        Minimum point of the cmap to use. Between 0 and 1.
    cmap_max : float or int
        Maximum point of the cmap to use. Between 0 and 1.
    custom_colors : list
        If you want to define custom colors for the curves, provide list. Length of the list should be the same as number of compounds.
    single_color : str
        Provide single color to color all ploted curves.
    custom_labels : list
        If you want to define custom labels for the curves, provide list. Length of the list should be the same as number of compounds.
    single_label : str
        Provide single label for all ploted curves.
    no_labels : bool
        If you do not want any labels, set this to true.
    """    

    # In compound selection is provided, than use it, otherwise plot all compounds
    if not compound_sel:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel

    # Setting up colors
    # By default, colors are set up as a cmap
    # cmap options: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = plt.colormaps[cmap](np.linspace(cmap_min, cmap_max, len(compounds)))
    if custom_colors:
        colors=custom_colors
    if single_color:
        colors=custom_colors
        colors = [single_color for _ in range(len(compounds))]
        
    # Setting up labels    
    labels = compounds
    if custom_labels:
        labels=custom_labels
    if single_label:
        labels = [single_label for _ in range(len(compounds))]
    if no_labels:
        labels = [None for _ in range(len(compounds))]

    
    # Iterate through compounds and plot them in matplotlib    
    for i, compound in enumerate(compounds):
        
        
        # This is a selection from the dataframe with the experimental data
        sel_data = input_df.loc[input_df['compound'] == compound]
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        

        if sel_results['model'].iloc[0] == "logIC50":
            conc = "log c"
        else:
            conc = "c"
        

        # Turning on/off plotting of errorbars, all data, or medians (defaults is medians with errorbars)
        if show_errorbars:
            plt.errorbar(sel_data[conc], sel_data["median"], yerr=sel_data[errorbars_kind], elinewidth=errorbar_linewidth, capthick=errorbar_linewidth, capsize=errorbar_capsize, linestyle="", marker="none", markersize=markersize, color=colors[i])
        if show_medians:
            plt.plot(sel_data[conc], sel_data["median"], marker=marker, markersize=markersize, linestyle="", color=colors[i])        
        if show_all_data:
            sel_data_pooled = pool_data(sel_data)
            plt.plot(sel_data_pooled[conc], sel_data_pooled["response"], marker=marker, markersize=markersize, linestyle="", color=colors[i])


        # Setting min and max on x axis for the curves
        if xmin:
            min_curve=xmin
        else:
            min_curve = min(sel_data[conc])
        if xmax:
            max_curve=xmax
        else:
            max_curve = max(sel_data[conc]) 

        
        # Setting up the x values for the curve
        
        if conc == "log c":
            x_curve = np.linspace(min_curve, max_curve, int((max_curve-min_curve)*100))
        else:
            x_curve = np.logspace(np.log10(min_curve), np.log10(max_curve), 1000)
        

        # Fetching parameters for a given model and getting y values
        y_curve = fetch_pars(sel_results, x_curve)

        # Plotting the curve
        plt.plot(x_curve, y_curve, color=colors[i], linestyle=linestyle, linewidth=linewidth)
        
        # Hidden plots just to make labels for the legend
        if not single_label:
            if show_medians and not show_all_data:
                plt.plot(sel_data[conc].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])
            if show_all_data:
                plt.plot(sel_data_pooled[conc].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])        
            if not show_medians and not show_all_data:
                plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=labels[i])
                
    if single_label:
        if show_medians and not show_all_data:
            plt.plot(sel_data[conc].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)
        if show_all_data:
            plt.plot(sel_data_pooled[conc].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=single_label)        
        if not show_medians and not show_all_data:
            plt.plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=single_label)
      

      
def plot_grid(input_df, results_df, compound_sel=False, xmin=False, xmax=False, 
         marker="o", 
         markersize=5, 
         linewidth=1, 
         linestyle="-",
         show_medians=True,
         show_all_data=False, 
         show_errorbars=True, 
         errorbars_kind="SD", 
         errorbar_linewidth = 1, 
         errorbar_capsize=3, 
         cmap="tab10", 
         cmap_min = 0, 
         cmap_max = 1, 
         custom_colors=False, 
         single_color=False, 
         custom_labels=False,
         single_label=False,
         no_labels=False,
         x_logscale=True,
         show_legend=False,
         show_title=True,
         figsize=(7, 5),
         n_cols=3,
         x_label="dose",
         y_label="response",         
         show_inner_ticklabels=False,
         sharex=True,
         sharey=True,
         hspace=0.3,
         wspace=0.3):
    """Plots a grid of binding curves.
    
    Parameters
    ----------
    input_df : DataFrame
        Pandas DataFrame containing the input data.
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds from the results_df will be used.
    xmin : float or int
        Manually set the minimum value on x axis for all curves. If set to False, it will be determined automatically for each curve.
    xmax : float or int
        Manually set the maximum value on x axis for all curves. If set to False, it will be determined automatically for each curve. 
    marker : str
        Marker type. Any matplotlib syntax is accepted (see https://matplotlib.org/stable/api/markers_api.html).
    markersize : float or int
        Marker size.
    linewidth : float or int
        Line width of the curve.
    linestyle : str
        Line style of the curve.
    show_medians : bool
        Whether to show concentration median values.
    show_all_data : bool
        Whether to show all concentration points.
    show_errorbars : bool
        Whether to show errorbars.
    errorbars_kind : str
        What should be shown as errorbars, options are "SD" or "SEM".
    errorbar_linewidth : float or int
        Line width of the errorbars. 
    errorbar_capsize : float or int
        Size of the errorbar caps (upper and lower points of the bars).
    cmap : str
        What cmap to use for coloring the curves. Any matplotlib syntax is accepted (see https://matplotlib.org/stable/gallery/color/colormap_reference.html).
    cmap_min : float or int
        Minimum point of the cmap to use. Between 0 and 1.
    cmap_max : float or int
        Maximum point of the cmap to use. Between 0 and 1.
    custom_colors : list
        If you want to define custom colors for the curves, provide list. Length of the list should be the same as number of compounds.
    single_color : str
        Provide single color to color all ploted curves.
    custom_labels : list
        If you want to define custom labels for the curves, provide list. Length of the list should be the same as number of compounds.
    single_label : str
        Provide single label for all ploted curves.
    no_labels : bool
        If you do not want any labels, set this to true.
    x_logscale : bool
        If set to True, the x axis will be plotted on a log scale.
    show_legend : bool
        Whether to show legend in each subplot.
    show_title : bool
        Whether to show names of the compounds in the title for each sunplot.
    figsize : tuple
        Tuple of (x, y) determining dimensions for the plot. This is passed into matplotlib figsize.
    n_cols : int
        Number of columns to plot. Number of rows is then determined automatically.
    x_label : str
        Axis label for the x axis.
    y_label : str
        Axis label for the y axis.
    show_inner_ticklabels : bool
        Whether to show ticklabels on the inner axes of the grid. 
    sharex : bool
        Whether to share (lock) the scales on the x axes for all subplots in the grid.
    sharey : bool
        Whether to share (lock) the scales on the y axes for all subplots in the grid.
    hspace : float or int
        Horizontal space between subplots.
    wspace : float or int
        Horizontal space between subplots.
    """
    

    # In compound selection is provided, than use it, otherwise plot all compounds
    if not compound_sel:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel

    # Setting up colors
    # By default, colors are set up as a cmap
    # cmap options: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colors = plt.colormaps[cmap](np.linspace(cmap_min, cmap_max, len(compounds)))
    if custom_colors:
        colors=custom_colors
    if single_color:
        colors=custom_colors
        colors = [single_color for _ in range(len(compounds))]
        
    # Setting up labels    
    labels = compounds
    if custom_labels:
        labels=custom_labels
    if single_label:
        labels = [single_label for _ in range(len(compounds))]
    if no_labels:
        labels = [None for _ in range(len(compounds))]

    
    #fig, axes = plt.subplots(2, 3, figsize=(7, 5))  # 2 rows, 2 columns
    
    # Figure out no of rows
    n_rows = int(len(compounds) / n_cols) + (len(compounds) % n_cols > 0)       # Just a fancy way of rounding up
    
    # Create a grid of subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=hspace, wspace=wspace)
    axes = gs.subplots(sharex=sharex, sharey=sharey)

    # Flatten the 2D axes array to iterate over it
    axes = axes.flatten()

    # Remove subplots if they would be empty
    n_remove = n_cols*n_rows - len(compounds)       # No of elements to temove

    for i in range(1, n_remove+1):
        fig.delaxes(axes[-i])  # Remove axes from the figure


    # Iterate through compounds and plot them in matplotlib    
    for i, compound in enumerate(compounds):
        
        # This is a selection from the dataframe with the experimental data
        sel_data = input_df.loc[input_df['compound'] == compound]
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]

        # Turning on/off plotting of errorbars, all data, or medians (defaults is medians with errorbars)
        if show_errorbars:
            axes[i].errorbar(sel_data["c"], sel_data["median"], yerr=sel_data[errorbars_kind], elinewidth=errorbar_linewidth, capthick=errorbar_linewidth, capsize=errorbar_capsize, linestyle="", marker="none", markersize=markersize, color=colors[i])
        if show_medians:
            axes[i].plot(sel_data["c"], sel_data["median"], marker=marker, markersize=markersize, linestyle="", color=colors[i])        
        if show_all_data:
            sel_data_pooled = pool_data(sel_data)
            axes[i].plot(sel_data_pooled["c"], sel_data_pooled["response"], marker=marker, markersize=markersize, linestyle="", color=colors[i])


        # Setting min and max on x axis for the curves
        if xmin:
            min_curve=xmin
        else:
            min_curve = min(sel_data["c"])
        if xmax:
            max_curve=xmax
        else:
            max_curve = max(sel_data["c"]) 

        # Setting up the x values for the curve
        x_curve = np.logspace(np.log10(min_curve), np.log10(max_curve), 1000)
        #x_curve = np.linspace(min_curve, max_curve, int((max_curve-min_curve)*100))

        # Fetching parameters for a given model and getting y values
        y_curve = fetch_pars(sel_results, x_curve)


        # Plotting the curve      
        axes[i].plot(x_curve, y_curve, color=colors[i], linestyle=linestyle, linewidth=linewidth)
        
        
        # Setting log scale for x axis
        if x_logscale:
            axes[i].set_xscale('log', base=10)
        
        # Show name in the title of subplots
        if show_title:
            axes[i].set_title(compound, fontsize=10)


        # Hidden plots just to make labels for the legend
        if show_medians and not show_all_data:
            axes[i].plot(sel_data['c'].iloc[0], sel_data['median'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])
        if show_all_data:
            axes[i].plot(sel_data_pooled['c'].iloc[0], sel_data_pooled['response'].iloc[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=labels[i])        
        if not show_medians and not show_all_data:
            axes[i].plot(x_curve[0], y_curve[0], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker="none", label=labels[i])
            
        # Setting up legend        
        if show_legend:        
            axes[i].legend()
                

    # Setting up axis labels 
    for ax in axes:
        ax.set(xlabel=x_label, ylabel=y_label)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes:
        ax.label_outer()
        if show_inner_ticklabels:
            ax.xaxis.set_tick_params(labelbottom=True)      # Put back x ticklabels 
            ax.yaxis.set_tick_params(labelbottom=True)      # Put back y ticklabels 

    
    plt.tight_layout()
    plt.show()






   
def plot_asymptotes(results_df, compound_sel=False, lower=True, upper=True, color="black", linewidth=1, linestyle="--"):
    """Plots lower and/or upper asymptote of the model as a horizontal line.
    
    Parameters
    ----------
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds from the results_df will be used.
    lower : bool
        Whether to plot the lower asymptote.
    upper : bool    
        Whether to plot the upper asymptote.  
    color : str
        Color for plotting the asymptotes. Any matplotlib syntax will be accepted.
    linewidth : floar or int
        Line width.
    linestyle : str
        Line style.
    """

    # If compound selection is provided, than use it, otherwise plot all compounds
    if not compound_sel:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
    
    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
             
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        if lower:
            plt.axhline(y = float(sel_results['ymin'].iloc[0]), color=color, linestyle=linestyle, linewidth=linewidth) 
        if upper:
            plt.axhline(y = float(sel_results['ymax'].iloc[0]), color=color, linestyle=linestyle, linewidth=linewidth)   




def plot_traces(results_df, value, compound_sel=False, kind="full", vtrace=True, htrace=True, color="black", linewidth=1, linestyle="--", label=None):
    """Plots traces to indicate a specific value on the curve.
    
    Parameters
    ----------
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    value : str
        What value to use for plotting the traces. This should be one of the column names in results_df. Usually "IC50", "Kd" or "Kds".
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds from the results_df will be used.
    kind : str
        What kind of trace should be plotted. Options are "full" or "partial".
    vtrace : bool
        Whether to plot the vertical trace.
    htrace : bool
        Whether to plot the horizontal trace.
    color : str
        Color for plotting the traces. Any matplotlib syntax will be accepted.
    linewidth : float or int
        Line width.
    linestyle : str
        Line style.
    label : str
        Label that will be used for the traces.
    """
    
    # If compound selection is provided, than use it, otherwise plot all compounds
    if not compound_sel:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
        

    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
        
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        x = float(sel_results[value].iloc[0])
        print("Plotting trace for x value", x)
        
        # Fetching parameters for a given model and getting y values
        y = fetch_pars(sel_results, x)

        
        # Plotting the traces
        if kind == "full":
            if vtrace:
                plt.axvline(x = x, color=color, linestyle=linestyle, linewidth=linewidth) 
            if htrace:
                plt.axhline(y = y, color=color, linestyle=linestyle, linewidth=linewidth)  
        
        if kind == "partial":
            if vtrace:
                plt.vlines(x, ymin=0, ymax=y, color=color, linestyle=linestyle, linewidth=linewidth)
            if htrace:
                plt.hlines(y, xmin=0, xmax=x, color=color, linestyle=linestyle, linewidth=linewidth)

         
        # Hidden plots just to make labels for the legend
        plt.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)





def plot_value(results_df, value, compound_sel=False, marker="o", markersize=5, color="black", label=None, show_annot=True, pre_text="", post_text="", decimals=2, xoffset=50, yoffset=0):
    """Plots a marker to indicate a specific value on the curve, optionally with text annotation.
    
    Parameters
    ----------
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    value : str
        What value to plot. This should be one of the column names in results_df. Usually "IC50", "Kd" or "Kds".
    compound_sel : list
        List of compounds to execute the function on. If set to False, all compounds from the results_df will be used.
    marker : str
        Marker type. Any matplotlib syntax is accepted (see https://matplotlib.org/stable/api/markers_api.html).
    markersize : float or int
        Marker size.
    color : str
        Color of the marker. Any matplotlib syntax will be accepted.
    label : str
        Label for the marker to show in legend.
    show_annot : bool
        Whether to show text annotation.
    pre_text : str
        Text to apear before the numerical annotation.
    post_text : str
        Text to apear after the numerical annotation.
    decimals : int
        Number of decimals to use for the numerical annotation.
    xoffset : float or int
        Offset of the annotation on x axis.
    yoffset : float or int
        Offset of the annotation on y axis.
    """
            
    # If compound selection is provided, than use it, otherwise plot all compounds
    if not compound_sel:
        compounds = results_df["compound"].unique()
    else:
        compounds = compound_sel
        

    # Iterate through compounds and plot them in matplotlib    
    for compound in compounds:
        
        # This is a selection from the dataframe with the fitting results
        sel_results = results_df.loc[results_df['compound'] == compound]
        
        x = float(sel_results[value].iloc[0])
        print("Plotting marker for x value", x)
        
        # Fetching parameters for a given model and getting y values
        y = fetch_pars(sel_results, x)

        
        # Plotting the marker
        plt.plot(x, y, marker=marker, markersize=markersize, color=color) 

         
        # Hidden plots just to make labels for the legend
        plt.plot(x, y, marker=marker, markersize=markersize, color=color, label=label, linestyle="")
        
        # Show annotation
        #plt.text(x, y, f"{pre_text}{x:.{decimals}f}{post_text}")
        if show_annot:
            plt.annotate(f"{pre_text}{x:.{decimals}f}{post_text}", (x, y), xytext=(x+xoffset, y+yoffset), horizontalalignment='center', verticalalignment='center')
        
        




def report(results_df, decimals=2):
    """Provides the results as a formatted report.
    
    Parameters
    ----------
    results_df : DataFrame
        Pandas DataFrame containing the fit results.
    decimals : int
        Number of decimals to use.
    
    Returns
    -------
    DataFrame
        Pandas DataFrame containing the report.
    """
    
    compounds = results_df["compound"].unique()

    # Initiating empty output_df
    output_df = pd.DataFrame(columns=['compound', 'Mean (95% CI)', 'Mean \u00B1 SE'])

    
    for compound in compounds:
        
        df_compound = results_df[results_df["compound"].isin([compound])]
        
        
        value_compound = df_compound.iloc[0, 2]
        loCL_compound = df_compound.iloc[0, 3]
        upCL_compound = df_compound.iloc[0, 4]
        SE_compound = df_compound.iloc[0, 5]
        
     
                
        # Creating new row for the output dataframe  
        if loCL_compound == "nd" and SE_compound != "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound}, {upCL_compound})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound:.{decimals}f}"]
        if SE_compound == "nd" and loCL_compound != "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound:.{decimals}f}, {upCL_compound:.{decimals}f})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound:}"]
        if loCL_compound == "nd" and SE_compound == "nd":
            new_row = [compound, 
                       f"{value_compound:.{decimals}f} ({loCL_compound}, {upCL_compound})", 
                       f"{value_compound:.{decimals}f} \u00B1 {SE_compound}"]            
        if loCL_compound != "nd" and SE_compound != "nd":
            new_row = [compound, 
                        f"{value_compound:.{decimals}f} ({loCL_compound:.{decimals}f}, {upCL_compound:.{decimals}f})", 
                        f"{value_compound:.{decimals}f} \u00B1 {SE_compound:.{decimals}f}"]
            

        # Adding new row to the output dataframe
        output_df.loc[len(output_df)] = new_row
    
    return output_df



    
    