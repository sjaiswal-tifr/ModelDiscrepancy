#!/usr/bin/env python3

# import sys, os
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
# from scipy.stats import beta
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# ============================================================================================

def setup_rc_params(fontsize=12, constrained_layout=True, usetex=True, dpi=400):

    black = "k"

    mpl.rcdefaults()  # Set to defaults
    x_minor_tick_size = y_minor_tick_size = 2.4
    x_major_tick_size = y_major_tick_size = 3.9

    # mpl.rc("text", usetex=True)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["text.usetex"] = usetex
    # mpl.rcParams["text.latex.preview"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.edgecolor"] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams["axes.labelcolor"] = black
    mpl.rcParams["axes.titlesize"] = fontsize

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["xtick.color"] = black
    mpl.rcParams["ytick.color"] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams["xtick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams["xtick.minor.size"] = x_minor_tick_size  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = y_minor_tick_size
    mpl.rcParams["xtick.major.size"] = x_major_tick_size  # Default 3.5
    mpl.rcParams["ytick.major.size"] = y_major_tick_size
    plt.rcParams["xtick.minor.visible"] =  True
    plt.rcParams["ytick.minor.visible"] =  True

    ppi = 72  # points per inch
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 0#3.0 / ppi
    mpl.rcParams["figure.constrained_layout.w_pad"] = 0#3.0 / ppi

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"

    mpl.rcParams["legend.title_fontsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams[
        "legend.edgecolor"
    ] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams["hatch.linewidth"] = 0.5
    
    return None

# ============================================================================================

def plot_corner_3compare(samples_1, samples_2, samples_3, color, plot_labels, param_ranges, truth, priors):
    """
    samples_1 : without MD (green)
    samples_2 : with MD Kernel I (blue)
    samples_3 : with MD Kernel II (red)
    
    This function creates a full pairplot (upper and lower triangles) with:
      - Lower triangle: KDE contours for samples_1 and samples_2
      - Upper triangle: KDE contours for samples_1 and samples_3
      - Diagonal: Normalized 1D KDEs for all three samples with median and ±68% CI text annotations.
    """
    n_params = len(plot_labels)
    colors = color

    # Convert samples to DataFrames and tag them
    df1 = pd.DataFrame(samples_1, columns=plot_labels)
    df1['Sample'] = 'samples_1'
    df2 = pd.DataFrame(samples_2, columns=plot_labels)
    df2['Sample'] = 'samples_2'
    df3 = pd.DataFrame(samples_3, columns=plot_labels)
    df3['Sample'] = 'samples_3'
    
    # Combine into a single DataFrame
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Create a full PairGrid
    g = sns.PairGrid(df_combined, vars=plot_labels, diag_sharey=False)

    # --------------------------------------------
    # Lower Triangle: KDE for `samples_1` and `samples_2`
    # --------------------------------------------
    def lower_kde_sample12(x, y, **kwargs):
        """
        Plot KDE for samples_1 and samples_2 in the lower triangle.
        """
        sample_data1 = df_combined[df_combined['Sample'] == 'samples_1']
        sample_data2 = df_combined[df_combined['Sample'] == 'samples_2']
        kwargs = {k: v for k, v in kwargs.items() if k != 'color'}
        sns.kdeplot(x=sample_data1[x.name], y=sample_data1[y.name],thresh=0.05, levels=5, color=colors[0], **kwargs,
                   fill=True, alpha=0.5, zorder=2
                   )
        sns.kdeplot(x=sample_data2[x.name], y=sample_data2[y.name],thresh=0.05, levels=5, color=colors[1], **kwargs,
                   fill=True, alpha=1, zorder=0
                   )

    # --------------------------------------------
    # Upper Triangle: KDE for `samples_1` and `samples_3`
    # --------------------------------------------
    def upper_kde_sample13(x, y, **kwargs):
        """
        Plot KDE for samples_1 and samples_3 in the upper triangle.
        """
        sample_data1 = df_combined[df_combined['Sample'] == 'samples_1']
        sample_data3 = df_combined[df_combined['Sample'] == 'samples_3']
        kwargs = {k: v for k, v in kwargs.items() if k != 'color'}
        sns.kdeplot(x=sample_data1[x.name], y=sample_data1[y.name],thresh=0.05, levels=5, color=colors[0], **kwargs,
                   fill=True, alpha=0.5, zorder=2
                   )
        sns.kdeplot(x=sample_data3[x.name], y=sample_data3[y.name],thresh=0.05, levels=5, color=colors[2], **kwargs,
                   fill=True, alpha=1, zorder=0
                   )
    
    # --------------------------------------------
    # Diagonal: Plot normalized KDEs with text annotations
    # --------------------------------------------
    def diag_density_unit_peak(x, **kwargs):
        ax = plt.gca()
        idx = plot_labels.index(x.name)
        
        # Extract data for the current parameter from each sample
        sample_woMD = df_combined.loc[df_combined['Sample'] == 'samples_1', x.name].dropna()
        sample_wMD1 = df_combined.loc[df_combined['Sample'] == 'samples_2', x.name].dropna()
        sample_wMD2 = df_combined.loc[df_combined['Sample'] == 'samples_3', x.name].dropna()

        # NEW: Compute and plot prior distribution
        xs_prior = np.linspace(param_ranges[idx][0], param_ranges[idx][1], 200)
        prior_density = np.exp([priors[idx](val) for val in xs_prior])  # Convert log-prob to linear
        
        # Normalize prior to match KDE peak height (max=1)
        if prior_density.max() > 0:
            prior_density_normalized = prior_density / prior_density.max() /2.0
        else:
            prior_density_normalized = prior_density

        
        # Plot filled prior (using zorder=0 to place behind other elements)
        ax.fill_between(
            xs_prior, 
            prior_density_normalized, 
            color='#800080',  # Purple color
            alpha=0.1, 
            zorder=0,
            label='Prior'
        )
        
        # Helper to compute KDE (normalized so that max=1) and stats
        def compute_kde_peak1_stats(data_array):
            if len(data_array) < 2:
                return None, None, None, None, None
                
            kde = gaussian_kde(data_array)
            xs = np.linspace(param_ranges[plot_labels.index(x.name)][0],
                             param_ranges[plot_labels.index(x.name)][1], 200)
            vals = kde(xs)
            peak = vals.max()
            if peak > 0:
                vals = vals / peak
            median = np.median(data_array)
            q16, q84 = np.percentile(data_array, [16, 84])
            plus = q84 - median
            minus = median - q16
            return xs, vals, median, plus, minus
        
        # For each sample, compute KDE and add both the line and text
        for data_array, color, (tx, ha) in [
            (sample_woMD, colors[0], (-0.08, 'left')),
            (sample_wMD1, colors[1],  (0.5, 'center')),
            (sample_wMD2, colors[2],   (1.08, 'right'))
        ]:
            xs, vals, median, plus, minus = compute_kde_peak1_stats(data_array)
            if xs is None:
                continue
            # Plot the KDE line
            ax.plot(xs, vals, color=color, linewidth=1, zorder=3)
            # Build a text string: e.g. "0.12^{+0.05}_{-0.03}"
            text_str = r"$%.2f^{+%.2f}_{-%.2f}$" % (median, plus, minus)
            # Place text above the plot (using axes coordinates, and clip_off so it is visible outside)
            ax.text(tx, 1.01, text_str,
                    transform=ax.transAxes,
                    ha=ha, va='bottom',
                    color=color,
                    fontsize=12,    # CHANGE TO ENSURE NO OVERLAP 
                    zorder=10,
                    clip_on=False)
            
        ax.set_ylim(0, 1.05)  # fix y-limit so the peak is 1 + a little
        ax.set_xlim(param_ranges[plot_labels.index(x.name)])
        ax.set_yticks([])  # Remove y-ticks on the diagonal

        # *** Draw the truth vertical line ***
        truth_value = truth[idx]
        ax.axvline(truth_value, color='gray', linestyle='--', linewidth=1, zorder=10)
    
    # Map the diagonals to the grid:
    g.map_lower(lower_kde_sample12)
    g.map_upper(upper_kde_sample13)
    g.map_diag(diag_density_unit_peak)

    # Remove ticks and labels from diagonals
    for i in range(n_params):
        diag_ax = g.axes[i, i]
        diag_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if i == 0:
            diag_ax.set_ylabel('')
    
    # Set axis ranges for off-diagonals based on param_ranges:
    for i in range(n_params):
        for j in range(n_params):
            if i != j:
                g.axes[i, j].set_xlim(param_ranges[j])
                g.axes[i, j].set_ylim(param_ranges[i])
    
    # Customize labels (only show tick labels on the bottom row and left column)
    for ax in g.axes[-1, :]:
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    for ax in g.axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    
    return g

# ============================================================================================
