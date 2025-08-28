#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

# ======================================================================================

def plot_obs_model(exp_data, truth, quantiles1, quantiles2, quantiles3, true_param, colors, obs_labels, legends, fig_name=None):

    t_exp = exp_data[:,0]
    num_bins_exp = len(exp_data[:,0])  # number of bins per observable
    
    t_truth = truth[:,0]
    num_bins_truth = len(truth[:,0])  # number of bins per observable
    mask = t_truth <= t_exp.max() + 0.07
    
    num_obs = 3  # number of observables
    obs_axes_indices = [0, 2, 3]  # Order in which to place observables
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
    axes = axes.flatten()
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.0)

    # ---------------------------------------------
    # get mean and std of experimental data in 1D arrays
    mean_exp_data = np.concatenate((exp_data[:,1], exp_data[:,3], exp_data[:,5]))
    std_exp_data = np.concatenate((exp_data[:,2], exp_data[:,4], exp_data[:,6]))

    # ---------------------------------------------
    
    for i in range(num_obs):
        ax = axes[obs_axes_indices[i]]
        start_idx_exp = i * num_bins_exp
        end_idx_exp = start_idx_exp + num_bins_exp

        start_idx_truth = i * num_bins_truth
        end_idx_truth = start_idx_truth + num_bins_truth
        
        # ---------------------------------------------
        # plot experimental observations
        mean_values = mean_exp_data[start_idx_exp:end_idx_exp]
        ci_95_percent = 1.96 * std_exp_data[start_idx_exp:end_idx_exp]  # 95% confidence interval

        # Plot the mean values with error bars
        ax.errorbar(t_exp, mean_values, yerr=[ci_95_percent, ci_95_percent], 
                    fmt='o', markersize=3, capsize=3, color='black', label=legends[0], zorder=8)

        # ---------------------------------------------
        # plot quantiles1
        model_median = quantiles1[0,start_idx_truth:end_idx_truth][mask]
        model_lower = quantiles1[1, start_idx_truth:end_idx_truth][mask]
        model_upper = quantiles1[2, start_idx_truth:end_idx_truth][mask]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth[mask], model_lower, model_upper, color=colors[0], alpha=0.3, label=legends[1], zorder=7)
        ax.plot(t_truth[mask], model_median, color=colors[0], zorder=7)
    
        # ---------------------------------------------
        # plot quantiles2
        model_median = quantiles2[0,start_idx_truth:end_idx_truth][mask]
        model_lower = quantiles2[1, start_idx_truth:end_idx_truth][mask]
        model_upper = quantiles2[2, start_idx_truth:end_idx_truth][mask]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth[mask], model_lower, model_upper, color=colors[1], alpha=0.3,label=legends[2], zorder=4)
        ax.plot(t_truth[mask], model_median, color=colors[1], zorder=4)

        # ---------------------------------------------
        # plot quantiles3
        model_median = quantiles3[0,start_idx_truth:end_idx_truth][mask]
        model_lower = quantiles3[1, start_idx_truth:end_idx_truth][mask]
        model_upper = quantiles3[2, start_idx_truth:end_idx_truth][mask]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth[mask], model_lower, model_upper, color=colors[2], alpha=0.3, label=legends[2], zorder=5)
        ax.plot(t_truth[mask], model_median, color=colors[2], zorder=5)

        # Set axis labels and titles
        ax.set_xlabel(r'time ($s$)', fontsize=16)
        ax.set_ylabel(obs_labels[i], fontsize=16)
        ax.tick_params(labelsize=16)
        ax.set_xticks([0,0.5,1])

    axes[0].set_ylim(52.5, 60.5)
    # Create a custom legend handles
    exp_handle = Line2D([], [], marker='o', color='black', markersize=3,
                        linestyle='none', label=r'``Experiment": 95\% CI')
    posterior_handles = [Patch(facecolor=col, label=lab) for col, lab in zip(colors, legends)]
    legend_handles = [exp_handle] + posterior_handles

    axes[1].cla()        # Clear any previous content
    axes[1].axis('off')  # Optionally remove the axis borders/ticks.
    axes[1].legend(handles=legend_handles, loc='center', bbox_to_anchor=(.36, 0.6), frameon=False, fontsize=14)

    plt.tight_layout(pad=0.7)
    
    # plt.grid(True)
    # plt.tight_layout()
    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()
    
# ======================================================================================

def plot_obs_modelPlusGP(exp_data, truth, quantiles1, quantiles2, quantiles3, true_param, colors, obs_labels, legends, fig_name=None):

    t_exp = exp_data[:,0]
    num_bins_exp = len(exp_data[:,0])  # number of bins per observable
    
    t_truth = truth[:,0]
    num_bins_truth = len(truth[:,0])  # number of bins per observable
    
    num_obs = int(quantiles3.shape[1]/num_bins_truth)  # number of observables
    
    # --- dynamic grid: max 4 columns; legend in last col of first row ---
    if num_obs ==1:
        nrows, ncols = 1, 2           # +1 slot for legend
        legend_ax_idx = 1             # last column of the row
        obs_axes_indices = [0]  
    elif num_obs <= 2:
        nrows, ncols = 1, 2           
        legend_ax_idx = None          
        obs_axes_indices = [0,1] 
    else:
        nrows, ncols = 2, 2                      # fixed 2x4 for up to 7 plots
        legend_ax_idx = 1                        # top-right of first row
        obs_axes_indices = [0, 2, 3]
    
    # safety (handles num_obs in 1..7)
    obs_axes_indices = obs_axes_indices[:num_obs]
    
    fig_w, fig_h = 2.75 * ncols, 2.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    # ---------------------------------------------
    # get mean and std of experimental data in 1D arrays
    mean_exp_data = np.concatenate((exp_data[:,1], exp_data[:,3], exp_data[:,5]))
    std_exp_data = np.concatenate((exp_data[:,2], exp_data[:,4], exp_data[:,6]))

    mean_truth = np.concatenate((truth[:,1], truth[:,2], truth[:,3]))
    # ---------------------------------------------
    
    for i in range(num_obs):
        ax = axes[obs_axes_indices[i]]
        start_idx_exp = i * num_bins_exp
        end_idx_exp = start_idx_exp + num_bins_exp

        start_idx_truth = i * num_bins_truth
        end_idx_truth = start_idx_truth + num_bins_truth
        
        # ---------------------------------------------
        truth_values = mean_truth[start_idx_truth:end_idx_truth]
        ax.plot(t_truth, truth_values, color='black', linestyle='-', zorder=9)

        # ---------------------------------------------
        # plot experimental observations
        mean_values = mean_exp_data[start_idx_exp:end_idx_exp]
        ci_95_percent = 1.96 * std_exp_data[start_idx_exp:end_idx_exp]  # 95% confidence interval

        # Plot the mean values with error bars
        ax.errorbar(t_exp, mean_values, yerr=[ci_95_percent, ci_95_percent], 
                    fmt='o', markersize=3, capsize=3, color='black', label=legends[0], zorder=8)

        # ---------------------------------------------
        # plot quantiles1
        model_median = quantiles1[0,start_idx_truth:end_idx_truth]
        model_lower = quantiles1[1, start_idx_truth:end_idx_truth]
        model_upper = quantiles1[2, start_idx_truth:end_idx_truth]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth, model_lower, model_upper, color=colors[0], alpha=0.3, label=legends[1], zorder=7)
        ax.plot(t_truth, model_median, color=colors[0], zorder=7)
    
        # ---------------------------------------------
        # plot quantiles2
        model_median = quantiles2[0,start_idx_truth:end_idx_truth]
        model_lower = quantiles2[1, start_idx_truth:end_idx_truth]
        model_upper = quantiles2[2, start_idx_truth:end_idx_truth]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth, model_lower, model_upper, color=colors[1], alpha=0.3,label=legends[2], zorder=4)
        ax.plot(t_truth, model_median, color=colors[1], zorder=4)

        # ---------------------------------------------
        # plot quantiles3
        model_median = quantiles3[0,start_idx_truth:end_idx_truth]
        model_lower = quantiles3[1, start_idx_truth:end_idx_truth]
        model_upper = quantiles3[2, start_idx_truth:end_idx_truth]

        # Plot the median with error bars representing the 95% confidence interval
        ax.fill_between(t_truth, model_lower, model_upper, color=colors[2],label=legends[2], zorder=5, 
                        alpha=0.8, facecolor='none', hatch='///')
        ax.plot(t_truth, model_median, color=colors[2], zorder=5)

        # Set axis labels and titles
        ax.set_xlabel(r'time ($s$)', fontsize=16)
        ax.set_ylabel(obs_labels[i], fontsize=16)
        ax.tick_params(labelsize=16)
        ax.set_xticks([0,0.5,1])

    axes[0].set_ylim(52.5, 60.5)
    # Create a custom legend handles
    truth_handle = Line2D([], [],  color='black', markersize=3,
                    linestyle='-', label=r'Truth')
    exp_handle = Line2D([], [], marker='o', color='black', markersize=3,
                        linestyle='none', label=r'``Experiment": 95\% CI')
    posterior_handles1 = [Patch(facecolor=col, label=lab) for col, lab in zip(colors[:2], legends[:2])]
    posterior_handles2 = [Patch(facecolor='none', label=legends[2], edgecolor=colors[2], hatch='///')]
    legend_handles = [truth_handle] + [exp_handle] + posterior_handles1 + posterior_handles2

    if legend_ax_idx != None:
        axes[legend_ax_idx].cla()        # Clear any previous content
        axes[legend_ax_idx].axis('off')  # Optionally remove the axis borders/ticks.
        axes[legend_ax_idx].legend(handles=legend_handles, loc='center', bbox_to_anchor=(.36, 0.6), frameon=False, fontsize=14)

    plt.tight_layout(pad=0.7)
    
    # plt.grid(True)
    # plt.tight_layout()
    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()
    