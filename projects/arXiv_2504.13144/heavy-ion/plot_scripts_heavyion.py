#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ======================================================================================

def plot_obs_model(exp_data, emu, quantiles1, quantiles2, quantiles3, true_param, colors, obs_labels, legends, fig_name=None):

    num_bins = len(exp_data[:,0])  # number of bins per observable
    bin_centers = exp_data[:,0]  # bin centers
    bin_width = 0.2
    bin_errors = np.full(len(bin_centers), bin_width/2)  # bin widths

    num_obs = 7  # number of observables

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    
    # ---------------------------------------------
    # get mean and std of experimental data in 1D arrays
    mean_exp_data = np.concatenate((exp_data[:,1], exp_data[:,3], exp_data[:,5], exp_data[:,7], exp_data[:,9], exp_data[:,11], exp_data[:,13]))
    std_exp_data = np.concatenate((exp_data[:,2], exp_data[:,4], exp_data[:,6], exp_data[:,8], exp_data[:,10], exp_data[:,12], exp_data[:,14]))

    # ---------------------------------------------
    # get mean and std of model prediction with true parameter values
    true_param = true_param.reshape(1,-1)
    true_mean, true_std = emu.predict(true_param)
    true_mean, true_std = true_mean.flatten(), true_std.flatten()
    # ---------------------------------------------

    # print(mean_exp_data.shape, true_mean.shape)

    obs_axes_indices = [0, 1, 2, 4, 5, 6, 7]  # Order in which to place observables
    
    for i in range(num_obs):
        ax = axes[obs_axes_indices[i]]
        start_idx = i * num_bins
        end_idx = start_idx + num_bins

        # ---------------------------------------------
        # plot experimental observations
        mean_values = mean_exp_data[start_idx:end_idx]
        ci_95_percent = 1.96 * std_exp_data[start_idx:end_idx]  # 95% confidence interval

        # Plot the mean values with error bars
        ax.errorbar(bin_centers, mean_values, xerr=bin_errors, markersize=1, fmt='o', color='black',
                    label=legends[0])

        # Add rectangles for the error regions
        for bin_cen, y, yerr in zip(bin_centers, mean_values, ci_95_percent):
            rect = Rectangle(
                (bin_cen - bin_width/2, y - yerr),  # Bottom-left corner of the rectangle
                bin_width,  # Rectangle width
                2 * yerr,  # Rectangle height
                edgecolor='none', 
                facecolor='black', 
                alpha=0.3,
                zorder=1 
            )
            ax.add_patch(rect)
        # ---------------------------------------------
        # plot quantiles1
        model_median = quantiles1[0,start_idx:end_idx]
        model_lower = quantiles1[1, start_idx:end_idx]
        model_upper = quantiles1[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers-bin_width/6, model_median, yerr=ci_errors, fmt='o', color=colors[0], 
                    label=legends[1], lw=1, markersize=2, capsize=2, zorder=3 )

        # ---------------------------------------------
        # plot quantiles2
        model_median = quantiles2[0,start_idx:end_idx]
        model_lower = quantiles2[1, start_idx:end_idx]
        model_upper = quantiles2[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers, model_median, yerr=ci_errors, fmt='o', color=colors[1], 
                    label=legends[2], lw=1, markersize=2, capsize=2, zorder=4 )

        # ---------------------------------------------
        # plot quantiles3
        model_median = quantiles3[0,start_idx:end_idx]
        model_lower = quantiles3[1, start_idx:end_idx]
        model_upper = quantiles3[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers+bin_width/6, model_median, yerr=ci_errors, fmt='o', color=colors[2], 
                    label=legends[3], lw=1, markersize=2, capsize=2, zorder=5 )

        # ---------------------------------------------
        # plot model with true param
        true_m= true_mean[start_idx:end_idx]
        true_ci_95= 1.96 * true_std[start_idx:end_idx]  # 95% confidence interval

        # Plot the mean values with error bars
        ax.errorbar(bin_centers, true_m, xerr=bin_errors, markersize=1, fmt='o', color='gray',
                    label=legends[4])

        # Add rectangles for the error regions
        for bin_cen, y, yerr in zip(bin_centers, true_m, true_ci_95):
            rect = Rectangle(
                (bin_cen - bin_width/2, y - yerr),  # Bottom-left corner of the rectangle
                bin_width,  # Rectangle width
                2 * yerr,  # Rectangle height
                edgecolor='gray', 
                facecolor='none', 
                alpha=0.8,
                zorder=0 
            )
            ax.add_patch(rect)


        # Set axis labels and titles
        ax.set_xlabel(r'$p_T\ \mathrm{(GeV)}$', fontsize=16)
        ax.set_ylabel(obs_labels[i], fontsize=16)
        ax.tick_params(labelsize=16)

        lastn = int(np.ceil(bin_centers[-1]))
        ax.set_xticks(np.linspace(0, lastn, num=int(lastn / 0.5) + 1))

    legend_handles, _ = axes[0].get_legend_handles_labels()
    # Use the empty subplot to display the legend.
    legend_ax = axes[3]
    # Clear any previous content (optional).
    legend_ax.cla()
    # Place the legend in the center of this axis.
    legend_ax.legend(handles=legend_handles, labels=legends, loc='center', fontsize=15)
    # Optionally remove the axis borders/ticks.
    legend_ax.axis('off')

    plt.tight_layout(pad=0.7)
    
    # plt.grid(True)
    # plt.tight_layout()
    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()

# ======================================================================================

def plot_obs_modelPlusGP(exp_data, emu, quantiles1, quantiles2, quantiles3, true_param, colors, obs_labels, legends, fig_name=None):

    num_bins = len(exp_data[:,0])  # number of bins per observable
    bin_centers = exp_data[:,0]  # bin centers
    bin_width = 0.2
    bin_errors = np.full(len(bin_centers), bin_width/2)  # bin widths

    num_obs = int(quantiles3.shape[1]/num_bins)  # number of observables
    
    # --- dynamic grid: max 4 columns; legend in last col of first row ---
    if num_obs <= 3:
        nrows, ncols = 1, num_obs + 1            # +1 slot for legend
        legend_ax_idx = ncols - 1                # last column of the row
        obs_axes_indices = list(range(num_obs))  # 0..num_obs-1
    else:
        nrows, ncols = 2, 4                      # fixed 2x4 for up to 7 plots
        legend_ax_idx = 3                        # top-right of first row
        # First 3 on row 1 (cols 0,1,2), rest on row 2 (indices 4..7)
        obs_axes_indices = [0, 1, 2] + list(range(4, 4 + (num_obs - 3)))
    
    # safety (handles num_obs in 1..7)
    obs_axes_indices = obs_axes_indices[:num_obs]
    
    fig_w, fig_h = 4 * ncols, 3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()
    
    # ---------------------------------------------
    # get mean and std of experimental data in 1D arrays
    mean_exp_data = np.concatenate((exp_data[:,1], exp_data[:,3], exp_data[:,5], exp_data[:,7], exp_data[:,9], exp_data[:,11], exp_data[:,13]))
    std_exp_data = np.concatenate((exp_data[:,2], exp_data[:,4], exp_data[:,6], exp_data[:,8], exp_data[:,10], exp_data[:,12], exp_data[:,14]))

    # ---------------------------------------------
    # get mean and std of model prediction with true parameter values
    true_param = true_param.reshape(1,-1)
    true_mean, true_std = emu.predict(true_param)
    true_mean, true_std = true_mean.flatten(), true_std.flatten()
    # ---------------------------------------------

    # print(mean_exp_data.shape, true_mean.shape)

    obs_axes_indices = [0, 1, 2, 4, 5, 6, 7]  # Order in which to place observables
    
    for i in range(num_obs):
        ax = axes[obs_axes_indices[i]]
        start_idx = i * num_bins
        end_idx = start_idx + num_bins

        # ---------------------------------------------
        # plot experimental observations
        mean_values = mean_exp_data[start_idx:end_idx]
        ci_95_percent = 1.96 * std_exp_data[start_idx:end_idx]  # 95% confidence interval

        # Plot the mean values with error bars
        ax.errorbar(bin_centers, mean_values, xerr=bin_errors, markersize=1, fmt='o', color='black',
                    label=legends[0])

        # Add rectangles for the error regions
        for bin_cen, y, yerr in zip(bin_centers, mean_values, ci_95_percent):
            rect = Rectangle(
                (bin_cen - bin_width/2, y - yerr),  # Bottom-left corner of the rectangle
                bin_width,  # Rectangle width
                2 * yerr,  # Rectangle height
                edgecolor='none', 
                facecolor='black', 
                alpha=0.3,
                zorder=1 
            )
            ax.add_patch(rect)
        # ---------------------------------------------
        # plot quantiles1
        model_median = quantiles1[0,start_idx:end_idx]
        model_lower = quantiles1[1, start_idx:end_idx]
        model_upper = quantiles1[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers-bin_width/6, model_median, yerr=ci_errors, fmt='o', color=colors[0], 
                    label=legends[1], lw=1, markersize=2, capsize=2, zorder=3 )

        # ---------------------------------------------
        # plot quantiles2
        model_median = quantiles2[0,start_idx:end_idx]
        model_lower = quantiles2[1, start_idx:end_idx]
        model_upper = quantiles2[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers, model_median, yerr=ci_errors, fmt='o', color=colors[1], 
                    label=legends[2], lw=1, markersize=2, capsize=2, zorder=4 )

        # ---------------------------------------------
        # plot quantiles3
        model_median = quantiles3[0,start_idx:end_idx]
        model_lower = quantiles3[1, start_idx:end_idx]
        model_upper = quantiles3[2, start_idx:end_idx]

        # Plot the median with error bars representing the 95% confidence interval
        ci_errors = [model_median - model_lower, model_upper - model_median]  # Asymmetric error bars
        ax.errorbar(bin_centers+bin_width/6, model_median, yerr=ci_errors, fmt='o', color=colors[2], 
                    label=legends[3], lw=1, markersize=2, capsize=2, zorder=5 )

        # ---------------------------------------------
        # # plot model with true param
        # true_m= true_mean[start_idx:end_idx]
        # true_ci_95= 1.96 * true_std[start_idx:end_idx]  # 95% confidence interval

        # # Plot the mean values with error bars
        # ax.errorbar(bin_centers, true_m, xerr=bin_errors, markersize=1, fmt='o', color='gray',
        #             label=legends[4])

        # # Add rectangles for the error regions
        # for bin_cen, y, yerr in zip(bin_centers, true_m, true_ci_95):
        #     rect = Rectangle(
        #         (bin_cen - bin_width/2, y - yerr),  # Bottom-left corner of the rectangle
        #         bin_width,  # Rectangle width
        #         2 * yerr,  # Rectangle height
        #         edgecolor='gray', 
        #         facecolor='none', 
        #         alpha=0.8,
        #         zorder=0 
        #     )
        #     ax.add_patch(rect)


        # Set axis labels and titles
        ax.set_xlabel(r'$p_T\ \mathrm{(GeV)}$', fontsize=16)
        ax.set_ylabel(obs_labels[i], fontsize=16)
        ax.tick_params(labelsize=16)

        lastn = int(np.ceil(bin_centers[-1]))
        ax.set_xticks(np.linspace(0, lastn, num=int(lastn / 0.5) + 1))

    # Collect labels from the first plotted axis
    legend_handles, _ = axes[obs_axes_indices[0]].get_legend_handles_labels()
    
    legend_ax = axes[legend_ax_idx]
    legend_ax.cla()
    legend_ax.legend(handles=legend_handles, labels=legends, loc='center', fontsize=15)
    legend_ax.axis('off')


    plt.tight_layout(pad=0.7)
    
    # plt.grid(True)
    # plt.tight_layout()
    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()

# ======================================================================================

def plot_etabys(samples1, samples2, samples3, eta_over_s, true_param, model_param_bounds, colors, fig_name=None):
    
    T_temp = np.linspace(0.14, 0.242, num=100)

    # For samples1
    results1 = np.array([[eta_over_s(T, param) for T in T_temp] for param in samples1])
    # For samples2
    results2 = np.array([[eta_over_s(T, param) for T in T_temp] for param in samples2])
    # For samples3
    results3 = np.array([[eta_over_s(T, param) for T in T_temp] for param in samples3])

    # Calculate mean, std, and quantiles or samples 1
    MAP1 = results1.mean(axis=0)
    quantiles1 = np.percentile(results1, [2.5, 50, 97.5], axis=0)  # 2.5th, 50th (median), 97.5th percentiles
    # ==========================================================

    # Calculate mean, std, and quantiles for samples 2
    MAP2 = results2.mean(axis=0)
    quantiles2 = np.percentile(results2, [2.5, 50, 97.5], axis=0)  # 2.5th, 50th (median), 97.5th percentiles
    # ==========================================================

    # Calculate mean, std, and quantiles for samples 3
    MAP3 = results3.mean(axis=0)
    quantiles3 = np.percentile(results3, [2.5, 50, 97.5], axis=0)  # 2.5th, 50th (median), 97.5th percentiles
    # ==========================================================

    # =========== Prior =============================
    # Sample parameters from flat priors
    n_samples = 5000
    prior_param_samples = np.random.uniform(
        low=model_param_bounds[:, 0], 
        high=model_param_bounds[:, 1], 
        size=(n_samples, model_param_bounds.shape[0])
    )

    # Compute eta/s for each sample and T
    eta_s_realizations = np.array([
        [eta_over_s(T, param) for T in T_temp] for param in prior_param_samples
    ])  # Shape: [n_samples, n_T]

    # Compute percentiles across all realizations for each T
    quantiles_prior = np.percentile(eta_s_realizations, [2.5, 50, 97.5], axis=0)
    # ==========================================================
    
    # Plot mean, quantiles, and error bands
    plt.figure(figsize=(5, 3.4))
    
    # plt.plot(T_temp, quantiles_prior[1], color='#800080', linestyle='-', lw=2, alpha=0.5)
    
    # Plot 95% quantile bands (2.5th to 97.5th percentile)
    plt.fill_between(
        T_temp,
        quantiles_prior[0],  # 2.5th percentile
        quantiles_prior[2],  # 97.5th percentile
        color='#800080',
        facecolor='none',
        edgecolor='#800080',
        alpha=0.5,
        zorder=0,
        label=r"$\mathrm{Prior:}\  95\%\ \mathrm{CI}$"
    )

    # ================ w/o MD =======================

    # Plot samples1 results
    plt.plot(T_temp, quantiles1[1], color=colors[0], linestyle='-', lw=2)
    
    # Plot 95% quantile bands (2.5th to 97.5th percentile)
    plt.fill_between(
        T_temp,
        quantiles1[0],  # 2.5th percentile
        quantiles1[2],  # 97.5th percentile
        color=colors[0],
        facecolor=colors[0],#'none',
        edgecolor=colors[0],
        # hatch='||',
        alpha=0.3,
        zorder=1,
        label=r"$\mathrm{Posterior: w/o \ MD}$"
    )

    # ==================== w/ MD kernel I ========================
    
    # Plot samples2 results
    plt.plot(T_temp, quantiles2[1], color=colors[1], linestyle='-', lw=2)
    
    # Plot 95% quantile bands (2.5th to 97.5th percentile)
    plt.fill_between(
        T_temp,
        quantiles2[0],  # 2.5th percentile
        quantiles2[2],  # 97.5th percentile
        color=colors[1],
        facecolor=colors[1], #'none', 
        edgecolor=colors[1],
        # hatch='\\\\',
        alpha=.3,
        zorder=2,
        label=r"$\mathrm{Posterior: MD\, Kernel\, I}$"
    )

    # ==================== w/ MD kernel II ========================
    
    # Plot samples3 results
    plt.plot(T_temp, quantiles3[1], color=colors[2], linestyle='-', lw=2)
    
    # Plot 95% quantile bands (2.5th to 97.5th percentile)
    plt.fill_between(
        T_temp,
        quantiles3[0],  # 2.5th percentile
        quantiles3[2],  # 97.5th percentile
        color=colors[2],
        facecolor='none',
        edgecolor=colors[2],
        hatch='||',
        alpha=1,
        zorder=3,
        label=r"$\mathrm{Posterior: MD\, Kernel\, II}$"
    )
    
    # ==================== Truth ==============================
    true_shear = np.array([eta_over_s(T, true_param) for T in T_temp])
    plt.plot(T_temp, true_shear, color='black', linestyle='--', 
               label=r"$\mathrm{True\ value}$",
               zorder=4)
    
    # Labels and grid
    plt.xlabel(r'$T\ \mathrm{(GeV)}$', fontsize=15)
    plt.ylabel(r'$\eta/s$', fontsize=17)
    plt.tick_params(labelsize=15)
    
    plt.legend(fontsize=12, loc="upper left",
              borderaxespad=.3   # padding between legend and axes
              )
    
    # plt.grid(True)

    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    
    plt.show()

# ============================================================================================


