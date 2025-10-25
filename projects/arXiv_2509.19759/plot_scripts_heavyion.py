#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ======================================================================================

def plot_eta_zeta(samples_eta, samples_zeta, eta_s, zeta_s, 
                  param_bounds_eta, param_bounds_zeta, 
                  colors, labels, linestyle, fig_name=None, quant = [5, 20, 50, 80, 95]): #[20, 80] for 60% CI [5, 95] for 90% CI. 50 for median

    T_temp = np.linspace(0.145, 0.36, num=100)

    # quantiles for eta/s per tempertaure gird
    quantiles = []
    for s in samples_eta:
        res = np.array([[eta_s(T, param) for T in T_temp] for param in s])
        quantiles.append(np.percentile(res, quant, axis=0))

    quantiles1_eta, quantiles2_eta = quantiles
    # ==========================================================
    
    # quantiles for zeta/s per tempertaure gird
    quantiles = []
    for s in samples_zeta:
        res = np.array([[zeta_s(T, param) for T in T_temp] for param in s])
        quantiles.append(np.percentile(res, quant, axis=0))

    quantiles1_zeta, quantiles2_zeta = quantiles
    del quantiles, res
    # ==========================================================

    # =========== Prior quantiles =============================
    # Sample parameters from flat priors for eta/s
    prior_param_samples = np.random.uniform(
        low=param_bounds_eta[:, 0], high=param_bounds_eta[:, 1], 
        size=(20000, param_bounds_eta.shape[0])
    )
    prior_realizations = np.array([[eta_s(T, param) for T in T_temp] for param in prior_param_samples])
    quantiles_prior_eta = np.percentile(prior_realizations, quant, axis=0)

    # Sample parameters from flat priors for zeta/s
    prior_param_samples = np.random.uniform(
        low=param_bounds_zeta[:, 0], high=param_bounds_zeta[:, 1], 
        size=(20000, param_bounds_zeta.shape[0])
    )
    prior_realizations = np.array([[zeta_s(T, param) for T in T_temp] for param in prior_param_samples])
    quantiles_prior_zeta = np.percentile(prior_realizations, quant, axis=0)
    
    del prior_param_samples, prior_realizations
    # ==========================================================
    
    def plot_helper(T, quantiles, color, facecolor='none', alpha=0.3, linestyle='-', zorder=1, label=None, hatch=None, showmedian=False):
    
        if showmedian==True:
            plt.plot(T, quantiles[1], color=color, linestyle=linestyle, lw=1.5)
        
        # Plot 95% quantile bands
        plt.fill_between(
            T, quantiles[0], quantiles[2],
            color=color, facecolor=facecolor, 
            alpha=alpha, linestyle=linestyle, lw=1.5,
            edgecolor=color if facecolor == 'none' else 'none', 
            hatch=hatch,
            zorder=zorder, label=label
        )

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharex=False, gridspec_kw={'wspace': 0.27})
    
    # ---------- zeta/s subplot ----------
    plt.sca(axes[0])
    plot_helper(T=T_temp, quantiles=quantiles_prior_zeta[[0,2,4], :],
                color = 'gray',  facecolor = 'gray',
                linestyle='-', zorder=0, alpha=0.3, hatch=None, showmedian=False)
    
    plot_helper(T=T_temp, quantiles=quantiles1_zeta[[1,2,3], :],
                color=colors[0], facecolor = colors[0], alpha=.3, linestyle='None', zorder=1)
    
    plot_helper(T=T_temp, quantiles=quantiles1_zeta[[0,2,4], :],
                color=colors[0], linestyle=linestyle[0], alpha=1, zorder=1)
    
    plot_helper(T=T_temp, quantiles=quantiles2_zeta[[1,2,3], :],
                color=colors[1], facecolor = colors[1], alpha=.3, linestyle='None', zorder=2)
    
    plot_helper(T=T_temp, quantiles=quantiles2_zeta[[0,2,4], :],
                color=colors[1], linestyle=linestyle[1], alpha=1, zorder=2)

    axes[0].set_xlabel(r'$T\ \mathrm{(GeV)}$', labelpad=0)
    axes[0].set_ylabel(r"$\zeta/s$", labelpad=0)
    axes[0].set_ylim(None, 0.2) #  0.185
    axes[0].set_xticks( [0.15, 0.2, 0.25, 0.3, 0.35]) #)np.linspace(min(T_temp), max(T_temp), num=int(max(T_temp)/10) + 1))
    
    # ---------- eta/s subplot ----------
    plt.sca(axes[1])
    plot_helper(T=T_temp, quantiles=quantiles_prior_eta[[0,2,4], :],
                color = 'gray',  facecolor = 'gray', label=r"$\mathrm{Prior:}\ 90\%\ \mathrm{CI}$",
                linestyle='-', zorder=0, alpha=0.3, hatch=None, showmedian=False)
    
    plot_helper(T=T_temp, quantiles=quantiles1_eta[[1,2,3], :], label=labels[0], 
                color=colors[0], facecolor = colors[0], alpha=.3, linestyle='None', zorder=1)

    plot_helper(T=T_temp, quantiles=quantiles2_eta[[1,2,3], :], label=labels[1], 
                color=colors[1], facecolor = colors[1], alpha=.3, linestyle='None', zorder=2)

    
    plot_helper(T=T_temp, quantiles=quantiles1_eta[[0,2,4], :], label=labels[2], 
                color=colors[0], linestyle=linestyle[0], alpha=1, zorder=1)

    plot_helper(T=T_temp, quantiles=quantiles2_eta[[0,2,4], :], label=labels[3], 
                color=colors[1], linestyle=linestyle[1], alpha=1, zorder=2)
    
    axes[1].set_xlabel(r'$T\ \mathrm{(GeV)}$', labelpad=0)
    axes[1].set_ylabel(r"$\eta/s$", labelpad=0)
    axes[1].set_ylim(None, 0.52) 
    axes[1].set_xticks( [0.15, 0.2, 0.25, 0.3, 0.35])
    # ---------------------------------
    
    # put legend only on the eta/s subplot
    handles, labels = axes[1].get_legend_handles_labels()

    leg = axes[1].legend(
        fontsize=12, loc="upper left", frameon=False,
        borderaxespad=0.3, handletextpad=0.3, labelspacing=0.2, columnspacing=0.6
    )
    # for h in leg.legendHandles:
    #     if hasattr(h, "set_alpha"):
    #         h.set_alpha(1.0)

    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.grid(True)

    if fig_name != None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()

# ============================================================================================

def plot_model_param(samples, mod_param_index, model_param_bounds, param_lab,
                     colors, linestyle, labels, alpha = 0.3, fig_name=None,
                     pos = [0.7, 0.6, 0.65, 0.6, 0.7, 0.65, 0.05, 0.7, 0.6]):
    
    samp1, samp2 = samples[0][:, mod_param_index], samples[1][:, mod_param_index]
    mod_param_bonds = model_param_bounds[mod_param_index]
    ndim = len(mod_param_index)

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 5.7), gridspec_kw={"wspace": 0.05, "hspace": 0.35})
    axes = axes.ravel()

    def tex_ci(m, q20, q80):
        dm60_lo, dm60_hi = (m - q20), (q80 - m)
        return rf"${m:.3g}^{{+{dm60_hi:.3g}}}_{{-{dm60_lo:.3g}}}$"
        
    for i in range(9):
        ax = axes[i]
        lo, hi = np.min(mod_param_bonds[i]), np.max(mod_param_bonds[i])
        xs = np.linspace(lo, hi, 600)

        # plot prior
        ax.axvspan(lo, hi, ymin=0.0, ymax=0.5, facecolor='gray', edgecolor='none', alpha=alpha, zorder=0)

        # plot posteriors
        d1 = samp1[:, i]
        d2 = samp2[:, i]

        kde1 = gaussian_kde(d1)
        kde2 = gaussian_kde(d2)
        ax.plot(xs, kde1(xs), color=colors[0], linestyle=linestyle[0], lw=1.2)
        ax.plot(xs, kde2(xs), color=colors[1], linestyle=linestyle[1], lw=1.2)

        # percentiles: 60% CI -> [20, 80], 90% CI -> [5, 95]
        q1_05, q1_20, q1_50, q1_80, q1_95 = np.percentile(d1, [5, 20, 50, 80, 95])
        q2_05, q2_20, q2_50, q2_80, q2_95 = np.percentile(d2, [5, 20, 50, 80, 95])

        # Shaded spans and median lines for set 1
        # ax.axvspan(q1_05, q1_95, color=colors[0], alpha=alpha/5, lw=0, zorder=0)  # 90%
        ax.axvspan(q1_20, q1_80, color=colors[0], alpha=alpha, lw=0, zorder=1)  # 60%
        ax.axvline(q1_50, color=colors[0], lw=1, linestyle=":", zorder=3)

        # Shaded spans and median lines for set 2
        # ax.axvspan(q2_05, q2_95, color=colors[1], alpha=alpha/5, lw=0, zorder=0)
        ax.axvspan(q2_20, q2_80, color=colors[1], alpha=alpha, lw=0, zorder=1)
        ax.axvline(q2_50, color=colors[1], lw=1, linestyle=":", zorder=3)
        
        # two lines of MathText with your requested format
        line1 = tex_ci(q1_50, q1_20, q1_80)
        line2 = tex_ci(q2_50, q2_20, q2_80)

        ax.text(pos[i], 0.95, line1, transform=ax.transAxes, ha="left", va="top", fontsize=9, color=colors[0])
        ax.text(pos[i], 0.75, line2, transform=ax.transAxes, ha="left", va="top", fontsize=9, color=colors[1])
        
        ax.tick_params(axis='x', pad=2)          # distance between tick marks and their numbers
        ax.set_xlabel(param_lab[i], labelpad=0)  # distance between numbers and the x-axis label

        ax.set_yticks([])
        # top = ax.get_ylim()[1]
        # ax.set_ylim(top=top * 1.25) 

    labels_txt = [r"Grad: $60\%$ CI", r"CE: $60\%$ CI"]
    dummy = [Line2D([], [], linestyle='none', linewidth=0) for _ in labels_txt]
    
    leg = axes[0].legend(dummy, labels_txt, loc="upper left", frameon=False,
                         handlelength=0, handletextpad=0.0, fontsize=9,
                         borderaxespad=0.0, labelspacing=0.2, columnspacing=0.6)
    
    for txt, col in zip(leg.get_texts(), [colors[0], colors[1]]):
        txt.set_color(col)

    axes[0].add_artist(leg)

    prior_patch = Patch(facecolor='gray', edgecolor='none', alpha=alpha, label='Prior')
    axes[0].legend(handles=[prior_patch], loc='upper left', frameon=False,
                   bbox_to_anchor=(0.0, 0.77),
                   handlelength=2, handletextpad=0.4, fontsize=9, 
                   borderaxespad=0.0, labelspacing=0.2, columnspacing=0.6)
    
    plt.tight_layout()# (rect=[0, 0, 1, 0.95])
    if fig_name is not None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()

# ============================================================================================

def plot_obs(experimental_data, obs_bins, quantiles_list, colors, obs_labels, legends, fig_name=None, 
             error = 'show', error_all = 'All', eps=1e-12):

    observables = list(obs_bins.keys())
    slices = []
    start_idx = 0
    for i, obs in enumerate(observables):
        n_bins_bayes = len(obs_bins[obs])
        end_idx = start_idx + n_bins_bayes
        slices.append((start_idx, end_idx))
        start_idx = end_idx
    
    num_rows, num_cols = 3, 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12),
                             sharex=False, gridspec_kw={'wspace': 0.25, 'hspace': 0.2})
    axes = axes.flatten()
    
    for i, obs in enumerate(observables):
        ax = axes[i]
        bin_centers = np.array([np.mean(interval) for interval in obs_bins[obs]])
        bin_width  = np.array([y - interval[0] for y, interval in zip(bin_centers, obs_bins[obs])])  # half-width
        
        slc = slice(slices[i][0], slices[i][1])
    
        # plot experimental data ------------------------------------------------>
        exp_mean = experimental_data[slc, 0]
        exp_std  = experimental_data[slc, 1]
        ax.errorbar(bin_centers, exp_mean, xerr=bin_width, markersize=1, fmt='o',
                    color='black', label=legends[0])

        # shaded boxes for exp errors
        for (x_left, x_right), y, yerr in zip(obs_bins[obs], exp_mean, exp_std):
            rect = Rectangle((x_left, y - yerr), x_right - x_left, 2 * yerr,
                             edgecolor='black', facecolor='black', alpha=0.2)
            ax.add_patch(rect)
        # ------------------------------------------------

        sep = 1.0/(len(quantiles_list)+1)  # fraction of local bin width for spacing
        if error == 'show':
            # --- add sub-axis for error plot that shares x with main ax ---
            divider = make_axes_locatable(ax)
            ax_gp = divider.append_axes("bottom", size="30%", pad=0.05, sharex=ax)
            ax.tick_params(axis='x', labelbottom=False)     # x labels only on GP panel
            ax_gp.axhline(0, color="0.5", lw=0.8, zorder=0)
        # ------------------------------------------------
        
        # plot model predictions and error plot ---------------------------------->
        m = len(quantiles_list)            # 4 series total
        bin_widths = obs_bins[obs][:, 1] - obs_bins[obs][:, 0]  # full width (for jitter)
        
        for j, (color, q_model) in enumerate(zip(colors[:m], quantiles_list)):
            q_model = np.asarray(q_model)
            median, lo, hi = q_model[0, slc], q_model[3, slc], q_model[4, slc]
            yerr = np.vstack([median - lo, hi - median])
            x_shift = (j - (m - 1) / 2) * sep * bin_widths
            xj = bin_centers + x_shift
    
            # model on main axis
            ax.errorbar(xj, median, yerr=yerr, fmt='o', color=color, lw=1, ms=2,
                        capsize=2, elinewidth=0.8, zorder=3, label=legends[j+1])

            if error == 'show':
                # bottom axis: residuals = experiment - model
                resid = (exp_mean - median)/(exp_mean+eps)  # add 'eps' to avoid division by 0
                resid_yerr = np.vstack([np.sqrt((median - lo)**2 + exp_std**2),
                                        np.sqrt((hi - median)**2 + exp_std**2),
                                       ])/(exp_mean+eps)
                
                if error_all == 'All':
                    ax_gp.errorbar(xj, resid, yerr=resid_yerr, fmt='o', color=color, lw=1, ms=2,
                                   capsize=2, elinewidth=0.8, zorder=2)
                    
                else:
                    # Show only the last `N` (= error_all) quantile results
                    N = max(0, min(int(error_all), m))  # clamp to [0, m]
                    if j >= m - N:
                        ax_gp.errorbar(xj, resid, yerr=resid_yerr, fmt='o', color=color, lw=1, ms=2,
                                       capsize=2, elinewidth=0.8, zorder=2)
    
        # labels/ticks (x on GP panel)
        ax.set_ylabel(obs_labels[obs], labelpad=3)
        ax.tick_params(axis='both', which='major')
        lastn = int(obs_bins[obs][-1][-1])
        if error == 'show':
            ax_gp.set_xticks(np.linspace(0, lastn, num=int(lastn/10) + 1))
            ax_gp.set_xlabel(r'\textrm{Centrality}', labelpad=0)
        else:
            ax.set_xticks(np.linspace(0, lastn, num=int(lastn/10) + 1))
            ax.set_xlabel(r'\textrm{Centrality}', labelpad=0)
    
        if i == 0:
            ax.legend(fontsize=12, loc="upper right", frameon=False,
                      handlelength=1.0, handletextpad=0.2,   # tighter text next to handle
                      labelspacing=0.3,                      # more vertical space between entries
                      borderaxespad=0.2, columnspacing=0.6)
    
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
    plt.show()
    
# ============================================================================================

# def plot_obs_error(experimental_data, obs_bins, quantiles_list, quantiles_error, colors, obs_labels, legends, 
#              fig_name=None, which = 'model'):

#     observables = list(obs_bins.keys())
#     slices = []
#     start_idx = 0
#     for i, obs in enumerate(observables):
#         n_bins_bayes = len(obs_bins[obs])
#         end_idx = start_idx + n_bins_bayes
#         slices.append((start_idx, end_idx))
#         start_idx = end_idx
    
#     num_rows, num_cols = 3, 4
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12),
#                              sharex=False, gridspec_kw={'wspace': 0.25, 'hspace': 0.2})
#     axes = axes.flatten()
    
#     for i, obs in enumerate(observables):
#         ax = axes[i]
#         bin_centers = np.array([np.mean(interval) for interval in obs_bins[obs]])
#         bin_width  = np.array([y - interval[0] for y, interval in zip(bin_centers, obs_bins[obs])])  # half-width
        
#         slc = slice(slices[i][0], slices[i][1])
    
#         # plot experimental data ------------------------------------------------>
#         exp_mean = experimental_data[slc, 0]
#         exp_std  = experimental_data[slc, 1]
#         ax.errorbar(bin_centers, exp_mean, xerr=bin_width, markersize=1, fmt='o',
#                     color='black', label=legends[0])

#         # shaded boxes for exp errors
#         for (x_left, x_right), y, yerr in zip(obs_bins[obs], exp_mean, exp_std):
#             rect = Rectangle((x_left, y - yerr), x_right - x_left, 2 * yerr,
#                              edgecolor='black', facecolor='black', alpha=0.2)
#             ax.add_patch(rect)
#         # ------------------------------------------------

#         if quantiles_error != None:
#             # --- add sub-axis for error plot that shares x with main ax ---
#             divider = make_axes_locatable(ax)
#             ax_gp = divider.append_axes("bottom", size="30%", pad=0.05, sharex=ax)
#             ax.tick_params(axis='x', labelbottom=False)     # x labels only on GP panel
    
#             if which == 'model':
#                 ax_gp.axhline(0, color="0.5", lw=0.8, zorder=0)
#                 sep = 0.20 # fraction of local bin width for spacing
#             elif which == 'modelPlusGP':
#                 ax_gp.axhline(1, color="0.5", lw=0.8, zorder=0)
#                 sep = 0.30 # fraction of local bin width for spacing
#             # ax_gp.set_yticks([])
#             # ax_gp.set_ylabel(r'$\delta_\textrm{GP}/|\textrm{data}|$', labelpad=0.1)
#         elif quantiles_error == None:
#             sep = 0.20
#         # ------------------------------------------------
        
#         # plot model predictions and error plot ---------------------------------->
#         m = len(quantiles_list)            # 4 series total
#         bin_widths = obs_bins[obs][:, 1] - obs_bins[obs][:, 0]  # full width (for jitter)
        
#         for j, (color, q_model) in enumerate(zip(colors[:m], quantiles_list)):
#             q_model = np.asarray(q_model)
#             median, lo, hi = q_model[0, slc], q_model[3, slc], q_model[4, slc]
#             yerr = np.vstack([median - lo, hi - median])
#             x_shift = (j - (m - 1) / 2) * sep * bin_widths
#             xj = bin_centers + x_shift
    
#             # model on main axis
#             ax.errorbar(xj, median, yerr=yerr, fmt='o', color=color, lw=1, ms=2,
#                         capsize=2, elinewidth=0.8, zorder=3, label=legends[j+1])

#             if quantiles_error != None:
#                 # GP on sub-axis for the last two series (match colors & jitter)
#                 if j >= m - len(quantiles_error):
#                     qgp = np.asarray(quantiles_error[j - (m - len(quantiles_error))])
#                     gmed, glo, ghi = qgp[0, slc], qgp[3, slc], qgp[4, slc]
#                     gyerr = np.vstack([gmed - glo, ghi - gmed])
#                     ax_gp.errorbar(xj, gmed, yerr=gyerr, fmt='o', color=color, lw=1, ms=2,
#                                    capsize=2, elinewidth=0.8, zorder=2)
    
#         # labels/ticks (x on GP panel)
#         ax.set_ylabel(obs_labels[obs], labelpad=3)
#         ax.tick_params(axis='both', which='major')
#         lastn = int(obs_bins[obs][-1][-1])
#         if quantiles_error != None:
#             ax_gp.set_xticks(np.linspace(0, lastn, num=int(lastn/10) + 1))
#             ax_gp.set_xlabel(r'\textrm{Centrality}', labelpad=0)
#         elif quantiles_error == None:
#             ax.set_xticks(np.linspace(0, lastn, num=int(lastn/10) + 1))
#             ax.set_xlabel(r'\textrm{Centrality}', labelpad=0)
    
#         if i == 0:
#             ax.legend(fontsize=12, loc="upper right", frameon=False,
#                       handlelength=1.0, handletextpad=0.2,   # tighter text next to handle
#                       labelspacing=0.3,                      # more vertical space between entries
#                       borderaxespad=0.2, columnspacing=0.6)
    
#     plt.tight_layout()
#     if fig_name is not None:
#         plt.savefig(fig_name, format="pdf", dpi=400, bbox_inches="tight")
#     plt.show()
    
# ============================================================================================




