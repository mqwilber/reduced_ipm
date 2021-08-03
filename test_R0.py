import numpy as np
import full_model as full
import reduced_model as red
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import scipy.stats as stats
importlib.reload(full)
importlib.reload(red)


"""
Test whether R0 equations in main text reflect invasion thresholds for
reduced IPM.  You can adjust parameters params_red or params_full to examine
whether different values for R0 correspond to the expected dynamic behavior
(invade or fail to invade) of the full and reduced IPM model.
"""


def sim_full_ipm(mod, steps):
    """ Simulate the full IPM model """

    sim = np.empty((len(mod.species[0].y) + 2, steps))
    sim[:, 0] = mod.species[0].density_vect
    for step in range(1, steps):
        mod.update_deterministic()
        sim[:, step] = mod.species[0].density_vect

    return(sim)


def summarize_sim(sim, mod):
    """ Extract summary statistics from simulation """

    loads = sim[1:-1, :]
    load_dists = loads / loads.sum(axis=0)
    y = mod.species[0].y
    mean_loads = (load_dists * y[:, np.newaxis]).sum(axis=0)
    var_loads = (load_dists * y[:, np.newaxis]**2).sum(axis=0) - mean_loads**2
    k = mean_loads**2 / var_loads

    sum_dict = {
        'loads': loads,
        'load_dists': load_dists,
        'y': y,
        'mean': mean_loads,
        'var': var_loads,
        'k': k
    }
    return(sum_dict)


if __name__ == '__main__':

    # Base reduced model parameters.
    # ADJUST THESE PARAMETERS TO EXPLORE R0 AS AN INVASION THRESHOLD

    # If True, look at full and reduced IPM with constant infected survival
    # If False, look at full and reduced IPM with load-dependent survival prob
    constant_surv = True

    # Parameter definitions
    # ---------------------
    # s0: uninfected survival prob (between 0 - 1)
    # sI: infected survival prob (between 0 - 1)
    # lI: Loss of infection prob (between 0 - 1)
    # trans_beta: Transmission parameter (between 0 - infinity)
    # sigma_full : Standard deviation of log load distribution (between 0 - infinity)
    # mu_s: LD50 of log-load dependent survival curve (between -infty - infty)
    # sigma_s: Shape parameter of the log-load dependent survival curve (smaller is steeper, between 0 and infinity)
    # a : log pathogen growth rate (-infty to infty)
    # b : Density-dependent pathogen growth (0 - 1, 1 is exponential growth)
    # lam : Pathogen shedding rate (0 to infinity)
    # max_shed : Maximum shedding rate of Type II shedding function
    # sigmaG : Standard deviation of within-host pathogen growth
    # sigma0 : Standard deviation of log initial infection load
    params_red = {
        's0': 1,
        'sI': 1,
        'lI': 0.3,
        'trans_beta': 0.003,
        'sigma_full': 1,
        'mu_s': 3,
        'sigma_s': 0.5,
        'gamma_s': 0,
        'gamma_l': 0.3,
        'a': 1.5,
        'b': 0.5,
        'k': 100,
        'kG': 10,
        'k0': 5,
        'lam': 1,
        'max_shed': 7.38,
        'r': 0,
        'K': 0.1,
        'nu': 0.1,
        'mu0': 0,
        'repro_time': 1,
        'sigmaG': 1,
        'sigma0': 1,
        'kappa': 0
    }

    model_params = {'time_step': 7}

    # Base full IPM model.
    # The only parameter you might want to change here is 'constant_surv'
    # `constant_surv` Determines whether the full IPM assumes the survival of
    # infected hosts is constant with load or not. Otherwise, all parameters
    # are matched to those defined in the reduced IPM.
    params_full = {
        'surv_sus': params_red['s0'],
        'growth_fxn_inter': params_red['a'],
        'growth_fxn_slope': params_red['b'],
        'growth_fxn_sigma': params_red['sigmaG'],
        'growth_fxn_k': params_red['kG'],
        'loss_fxn_inter': params_red['lI'],
        'loss_fxn_slope': 0,
        'loss_gamma': params_red['gamma_l'],
        'init_inf_fxn_inter': params_red['mu0'],
        'init_inf_fxn_sigma': params_red['sigma0'],
        'init_inf_fxn_k': params_red['k0'],
        'surv_fxn_inter': params_red['mu_s'],
        'surv_fxn_slope': params_red['sigma_s'],
        'surv_gamma': params_red['gamma_s'],
        'trans_fxn_zpool': params_red['trans_beta'],
        'max_load': 1000,
        'min_load': -1000,
        'shedding_prop': params_red['lam'],
        'repro_time': params_red['repro_time'],
        'fec': params_red['r'],
        'K': params_red['K'],
        'nu': params_red['nu'],
        'constant_surv': constant_surv,  # Set to False to include load-dependent survival
        'constant_loss': True,
        'sI': params_red['sI'],
        'lI': params_red['lI']
    }

    ipm_params = {
        'min_size': -10,
        'max_size': 20,
        'bins': 300,
        'time_step': 7
    }

    Sinit = 10
    steps = 100

    ### Models on log scale ###

    ### Set up and simulate the Full IPM
    init_dens_full = np.r_[Sinit, np.zeros(ipm_params['bins']), 1]
    comm_params = {'time': 7,
                   'species': {'spp1': params_full},
                   'density': {'spp1': init_dens_full}}
    full_mod_log = full.Community('log', comm_params, ipm_params,
                                  pathogen_scale="log")
    Rmat_log, R0log = full_mod_log.species[0].model_R0(init_dens_full[0])
    sim_full_log = sim_full_ipm(full_mod_log, steps)
    full_log_sum = summarize_sim(sim_full_log, full_mod_log)

    ### Initialize and simulate reduced model on the log scale
    if constant_surv:
        model_nm = "1mom_log"
    else:
        model_nm = "1mom_log_sl"

    params_red['sigma_full'] = np.sqrt(full_log_sum['var'][-1])*1
    red_modlog = red.ReducedModel(params_red, np.array([Sinit - 1e-5, 1e-5, 0, 1]),
                                  model_nm=model_nm)
    sim_modlog = red_modlog.simulate(steps)

    if constant_surv:
        logmean = red.muF_log(params_red)
        R0_redlog = red.calc_R0_log(Sinit, params_red, logmean)
    else:
        logmean = red.muF_log_sl(params_red)
        R0_redlog = red.calc_R0_log_sl(Sinit, params_red, logmean)

    # Does R0 act as an invasion threshold? Look at the predicted plot to
    # determine if Z, P, and I are increasing or decreasing. Decreasing should
    # correspond with R0 < 1 and increasing should correspond to R0 > 1.
    fig, axes = plt.subplots(1, 1)
    axes = [axes]
    full_mods = [full_mod_log]
    full_sims = [sim_full_log]
    red_sims = [sim_modlog]
    time = np.arange(steps)*model_params['time_step']
    titles = ['log']

    for i in range(len(axes)):
        ax = axes[i]

        fres = full_sims[i]
        rres = red_sims[i]

        # Plot full IPM dynamics
        y = full_mods[i].species[0].y
        Iipms = fres[1:-1, :].sum(axis=0)
        Pipms = (fres[1:-1, :] * y[:, np.newaxis]).sum(axis=0)

        ax.semilogy(time, Iipms, label="I, full", color=sns.color_palette()[0])
        ax.semilogy(time, fres[0, :], label="S, full", color=sns.color_palette()[1])
        ax.semilogy(time, Pipms, label="P, full", color=sns.color_palette()[2])
        ax.semilogy(time, Pipms / Iipms, label="mean, full", color=sns.color_palette()[3])

        # Plot reduced IPM dynamics
        ax.semilogy(time, rres[1, :], ':', label="I, red", color=sns.color_palette()[0])
        ax.semilogy(time, rres[0, :], ':', label="S, red", color=sns.color_palette()[1])
        ax.semilogy(time, rres[2, :], ':', label="P, red", color=sns.color_palette()[2])
        ax.semilogy(time, rres[2, :] / rres[1, :], ':', label="mean, red", color=sns.color_palette()[3])

        ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("Log density/load")

    ax.set_title("Full IPM R0: {0:.3}, Red. IPM R0: {1:.3}".format(np.abs(R0log), R0_redlog))

    # Uncomment to look at the equilibrium load distributions for reduced
    # and full IPMs
    # fig, ax = plt.subplots(1, 1)
    # for i in np.arange(40, 50, step=2):
    #     dist = full_log_sum['load_dists'][:, i]
    #     y = full_log_sum['y']
    #     ax.plot(y, dist)

    #     tmean = logmean
    #     sd = np.sqrt(full_log_sum['var'][i])
    #     dy = y[1] - y[0]
    #     norm = stats.norm(loc=tmean, scale=sd)
    #     ax.plot(y, norm.pdf(y)*dy, color='black')

    plt.show()
