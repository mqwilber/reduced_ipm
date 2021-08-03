import numpy as np
from scipy.special import loggamma
import scipy.stats as stats
from scipy.optimize import brentq, fsolve

"""
Classes and function for simulating and exploring reduced host-parasite IPMs

Author : Mark Wilber
"""


class ReducedModel(object):
    """
    A reduced IPM model
    """

    def __init__(self, params, init, model_nm="1mom_nat"):
        """
        Parameters
        ----------
        params : dict
            Dictionary with model parameters
        init : array-like
            Initial values for model
        model_nm : str
            The type of model to fit
            `1mom_nat`: First moment IPM with load on the natural scale
            `1mom_log`: First moment IPM with load on the log scale
            `2mom_nat`: Second moment IPM with load on the natural scale
            `2mom_log`: Second moment IPM with load on the log scale
            `1mom_nat_sl`: First moment IPM with load on the natural and survival and loss being load-dependent
            `1mom_log_sl`: First moment IPM with load on the log scale and survival being load-dependent

        Notes
        -----
        Parameters to include params

        Shared parameters
            s0: Survival probability of uninfected host
            sI: Survival probability of infected host
            lI: Loss of infection probabilit
            trans_beta: Transmission parameter
            a: Either the log pathogen growth rate or natural scale pathogen growth rate
            b: Density-dependent pathogen growth parameter
            mu0: Mean initial infection intensity
            lam: Pathogen shedding rate
            nu: Zoospore survival probability in a time-ste
            max_shed: Maximum shedding rate for Type II shedding function

        Log load parameters
            mu_s: LD50 for load-dependent host survival on the log scale
            sigma_s: Inverse measure of steepness of host survival curve
            sigmaG: Standard deviation in log pathogen growth
            sigma0: Standard deviation in initial infection of pathogen growth
            sigma_full: Fixed variance of the reduced IPM

        Natural load parameters
            gamma_s: Rate parameter for load-dependent mortality on the natural scale
            gamma_l: Rate parameter for load-dependent loss of infection on the natural scale
            k: Fixed aggregation parameter for parasite load distribution on the natural scale
            kG: Fixed aggregation parameter for parasite growth on natural scale
            k0: Fixed aggregation parameter for initial infection on the natural scale

        """

        self.params = params
        self.init = np.array(init)
        self.model_nm = model_nm

    def simulate(self, steps):
        """
        Simulate the reduced IPM model
        """

        time = np.arange(steps)
        n = len(self.init)
        res = np.zeros((n, len(time)))
        res[:, 0] = np.array(self.init)

        for t in time[1:]:
            val_now = res[:, (t - 1)]

            # Build K matrix
            if self.model_nm == "1mom_nat":
                K = self.build_K_1mom_natural(val_now)
            elif self.model_nm == "1mom_log":
                K = self.build_K_1mom_log(val_now)
            elif self.model_nm == "2mom_log":
                K = self.build_K_2mom_log(val_now)
            elif self.model_nm == "2mom_nat":
                K = self.build_K_2mom_natural(val_now)
            elif self.model_nm == '1mom_nat_sl':
                K = self.build_K_1mom_natural_sl(val_now)
            elif self.model_nm == '1mom_log_sl':
                K = self.build_K_1mom_log_sl(val_now)
            elif self.model_nm == '1mom_log_sl_shed':
                K = self.build_K_1mom_log_sl_shed(val_now)

            res[:, t] = np.dot(K, val_now)

        return(res)

    def build_K_1mom_natural(self, svar):
        """
        Projection matrix for first moment reduced IPM on the natural scale

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Z] at time t

        Returns
        -------
        : 4 by 4 projection matrix
        """
        p = self.params
        S, I, P, Z = svar

        phi = 1 - np.exp(-p['trans_beta']*Z)
        sl = p['sI']*(1 - p['lI'])
        Icontrib1 = (P / I)**p['b'] * p['a'] * sl
        Icontrib2 = np.exp(loggamma(p['b'] + p['k']) - (loggamma(p['k']) + p['b']*np.log(p['k'])))

        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*phi*p['mu0'],
                         0])
        colI = np.array([p['sI']*p['lI'],
                         sl,
                         Icontrib1*Icontrib2,
                         0])
        colP = np.array([0,
                         0,
                         0,
                         p['lam']])
        colZ = np.array([0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colZ]).T
        return(K)

    def build_K_1mom_natural_sl(self, svar):
        """
        Projection matrix for first moment reduced IPM on the natural scale.
        Survival and and loss of infection are assumed to increase exponentially
        with load.

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Z] at time t

        Returns
        -------
        : 4 by 4 projection matrix
        """
        p = self.params
        S, I, P, Z = svar

        phi = 1 - np.exp(-p['trans_beta']*Z)
        sl = h_fxn(p['gamma_s'], P / I, p['k']) - h_fxn(p['gamma_s'] + p['gamma_l'], P / I, p['k'])
        Icontrib = p['a']*(g_fxn(p['gamma_s'], P / I, p['k'], p['b']) -
                           g_fxn(p['gamma_s'] + p['gamma_l'], P / I, p['k'], p['b']))

        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*phi*p['mu0'],
                         0])
        colI = np.array([h_fxn(p['gamma_s'] + p['gamma_l'], P / I, p['k']),
                         sl,
                         Icontrib,
                         0])
        colP = np.array([0,
                         0,
                         0,
                         p['lam']])
        colZ = np.array([0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colZ]).T
        return(K)

    def build_K_1mom_log(self, svar):
        """
        Projection matrix for first moment reduced IPM on the log scale

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Z] at time t

        Returns
        -------
        : 4 by 4 projection matrix
        """

        p = self.params
        S, I, P, Z = svar

        phi = 1 - np.exp(-p['trans_beta']*Z)
        sl = p['sI']*(1 - p['lI'])

        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*phi*p['mu0'],
                         0])
        colI = np.array([p['sI']*p['lI'],
                         sl,
                         sl*p['a'],
                         p['lam']*np.exp(P / I)*np.exp(p['sigma_full']**2 / 2)])
        colP = np.array([0,
                         0,
                         sl*p['b'],
                         0])
        colZ = np.array([0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colZ]).T
        return(K)

    def build_K_1mom_log_sl(self, svar):
        """
        Projection matrix for first moment reduced IPM on the log scale with
        non-linear survival function.

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Z] at time t

        Returns
        -------
        : 4 by 4 projection matrix
        """

        p = self.params
        S, I, P, Z = svar

        phi = 1 - np.exp(-p['trans_beta']*Z)

        talpha = ((P / I) - p['mu_s']) / p['sigma_s']
        tzeta = p['sigma_full'] / p['sigma_s']
        arg1 = talpha / np.sqrt(1 + tzeta**2)
        sI_prime = 1 - stats.norm.cdf(arg1)
        # print(sI_prime)

        bcoef1 = (p['sigma_full']*(tzeta / np.sqrt(1 + tzeta**2))*stats.norm.pdf(arg1))
        bcoef2 = (P / I)*(1 - sI_prime)
        bcoef = (P / I) - (bcoef1 + bcoef2)


        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*p['mu0']*phi,
                         0])

        # Alternative formulation
        colI = np.array([sI_prime*p['lI'],
                         sI_prime*(1 - p['lI']),
                         p['a']*(1 - p['lI'])*sI_prime + p['b']*(1 - p['lI'])*bcoef,
                         p['lam']*np.exp(P / I)*np.exp(p['sigma_full']**2 / 2)])

        colP = np.array([0,
                         0,
                         0,
                         0])

        colZ = np.array([0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colZ]).T
        return(K)

    def build_K_1mom_log_sl_shed(self, svar):
        """
        Projection matrix for first moment reduced IPM on the log scale with
        non-linear survival function and non-proportional shedding.

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Z] at time t

        Returns
        -------
        : 4 by 4 projection matrix
        """

        p = self.params
        S, I, P, Z = svar

        phi = 1 - np.exp(-p['trans_beta']*Z)

        talpha = ((P / I) - p['mu_s']) / p['sigma_s']
        tzeta = p['sigma_full'] / p['sigma_s']
        arg1 = talpha / np.sqrt(1 + tzeta**2)
        sI_prime = 1 - stats.norm.cdf(arg1)
        # print(sI_prime)

        bcoef1 = (p['sigma_full']*(tzeta / np.sqrt(1 + tzeta**2))*stats.norm.pdf(arg1))
        bcoef2 = (P / I)*(1 - sI_prime)
        bcoef = (P / I) - (bcoef1 + bcoef2)

        shed_coef = typeII_approx(P / I, p['sigma_full'], p['max_shed'])


        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*p['mu0']*phi,
                         0])

        # Alternative formulation
        colI = np.array([sI_prime*p['lI'],
                         sI_prime*(1 - p['lI']),
                         p['a']*(1 - p['lI'])*sI_prime + p['b']*(1 - p['lI'])*bcoef,
                         shed_coef])

        colP = np.array([0,
                         0,
                         0,
                         0])

        colZ = np.array([0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colZ]).T
        return(K)

    def build_K_2mom_log(self, svar):
        """
        Projection matrix for the second moment reduced IPM on the log scale

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, Plog, Vlog, Z] at time t

        Returns
        -------
        : 6 by 6 projection matrix

        Notes
        -----

        Interesting.  Modeling P (not only logP) is actually pretty important for
        capturing the correct dynamics. Really helps get Z right, which helps
        get prevalence right.

        Note: 'a' is on the log scale when passed in

        """
        p = self.params
        S, I, P, Plog, Vlog, Z = svar
        mulog = Plog / I
        logv = (Vlog / I) - (Plog / I)**2

        phi = 1 - np.exp(-p['trans_beta']*Z)
        sl = p['sI']*(1 - p['lI'])

        sigma_contrib = (logv*p['b']**2 + p['sigmaG']**2) / 2

        Icontrib = (np.exp(Plog / I)**p['b'] *
                    sl *
                    np.exp(p['a']) *
                    np.exp(sigma_contrib))

        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*np.exp(p['mu0'])*np.exp(p['sigma0']**2 * 0.5)*phi,
                         p['s0']*phi*p['mu0'],
                         p['s0']*phi*(p['sigma0']**2 + p['mu0']**2),
                         0])
        colI = np.array([p['sI']*p['lI'],
                         sl,
                         Icontrib,
                         sl*p['a'],
                         sl*(p['sigmaG']**2 + p['a']**2),
                         0])

        colP = np.array([0,
                         0,
                         0,
                         0,
                         0,
                         p['lam']])
        colPlog = np.array([0,
                            0,
                            0,
                            sl*p['b'],
                            sl*2*p['a']*p['b'],
                            0])
        colVlog = np.array([0,
                            0,
                            0,
                            0,
                            sl*p['b']**2,
                            0])
        colZ = np.array([0,
                         0,
                         0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colPlog, colVlog, colZ]).T
        return(K)

    def build_K_2mom_natural(self, svar):
        """
        Projection matrix for the second moment reduced IPM on the natural scale

        Parameters
        ----------
        svar : array-like
            Array of state variable values [S, I, P, V, Z] at time t

        Returns
        -------
        : 5 by 5 projection matrix

        Notes
        -----


        """
        p = self.params
        S, I, P, V, Z = svar
        v = (V / I) - (P / I)**2
        if v <= 0:
            v = 0.01

        phi = 1 - np.exp(-p['trans_beta']*Z)
        sl = p['sI']*(1 - p['lI'])
        k = (P / I)**2 / v
        theta = k / (P / I)

        Icontrib1 = (p['a']*sl*np.exp(loggamma(p['b'] + k) - (loggamma(k) + p['b']*np.log(theta))))
        Icontrib2 = (p['a']**2 * sl * (1 + (1 / p['kG'])) *
                     np.exp(loggamma(2*p['b'] + k) - (loggamma(k) + 2*p['b']*np.log(theta)))
                     )

        colS = np.array([p['s0']*(1 - phi),
                         p['s0']*phi,
                         p['s0']*phi*p['mu0'],
                         p['s0']*phi*p['mu0']**2 * (1 + (1 / p['k0'])),
                         0])
        colI = np.array([p['sI']*p['lI'],
                         sl,
                         Icontrib1,
                         Icontrib2,
                         0])
        colP = np.array([0,
                         0,
                         0,
                         0,
                         p['lam']])
        colV = np.array([0,
                         0,
                         0,
                         0,
                         0])
        colZ = np.array([0,
                         0,
                         0,
                         0,
                         p['nu']])

        K = np.vstack([colS, colI, colP, colV, colZ]).T
        return(K)


def typeII_approx(x, sigma, lam):
    """ Helper function for type II approximation with shedding """

    first = (lam * np.exp(x)) / (lam + np.exp(x))
    second = (lam**2 * np.exp(x) * (lam - np.exp(x))) / (lam + np.exp(x))**3

    # Second order Taylor expansion
    res = first + (second / 2) * sigma**2
    return(res)


def aftersurv_approx(x, sigmaF, sigmaS, muS, lam):
    """
    Taylor expansion approximation for

    lambda * e^x * [1 - Phi((x - muS) / sigmaS)]

    """

    tsd = np.sqrt(sigmaS**2)# + sigmaF**2)
    surv = stats.norm(loc=muS, scale=tsd)
    l1 = lam * np.exp(x) * surv.sf(x)
    l2 = lam * np.exp(x) * (1 - surv.cdf(x) - 2*surv.pdf(x) + ((x - muS) / (tsd**2)) * surv.pdf(x))
    approx2 = l1 + l2*(sigmaF**2 / 2)
    return(approx2)


def aftersurv_exact(x, sigmaF, sigmaS, muS, lam, lower=-10, upper=10, num=100000):
    """
    Evaluation of the integral for shedding after survival
    """
    y = np.linspace(lower, upper, num=num)
    dy = y[1] - y[0]
    shed = lam * np.sum(np.exp(y) *
                        stats.norm(loc=muS,
                                   scale=sigmaS).sf(y) *
                        stats.norm(loc=x, scale=sigmaF).pdf(y) * dy)
    return(shed)


def h_fxn(γ, μ, k):
    """ Helper function """

    num1 = (k*(np.log(k) - np.log(μ)))
    denom1 = k*np.log(k*(1 / μ) + γ)
    return(np.exp(num1 - denom1))


def g_fxn(γ, μ, k, b):
    """ Helper function """

    num1 = (k*(np.log(k) - np.log(μ)))
    denom1 = (k + b)*np.log(k*(1 / μ) + γ)
    p1 = (num1 - denom1)
    p2 = loggamma(k + b) - loggamma(k)
    return(np.exp(p1 + p2))


def calc_R0_natural(Sinit, params, muF):
    """
    R0 for reduced IPM with gamma-distributed natural scale load

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model

    Returns
    -------
    : R0
    """

    sl = params['sI']*(1 - params['lI'])

    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - params['nu'])
    ratio2 = (params['lam'] * muF) / (1 - sl)

    R0 = ratio1*ratio2

    return(R0)


def muF_fxn_nat(muF, params):
    """ Implicit function for mean pathogen load on natural scale"""

    sl = params['sI']*(1 - params['lI'])
    k = params['k']
    b = params['b']
    a = params['a']
    Kon = np.exp(loggamma(b + k) - (loggamma(k) + b*np.log(k)))
    return(sl*Kon*(a*muF**b) + (1 - sl)*params['mu0'] - muF)


def muF_natural(params):
    """ Equilibrium mean for reduced IPM on natural scale given params """

    lower = params['mu0']
    upper = params['a']**(1 / (1 - params['b']))

    if np.round(lower, 2) != np.round(upper, 2):
        natmean = brentq(muF_fxn_nat, lower, upper, args=(params,))
    else:
        natmean = lower

    return(natmean)


def calc_R0_log(Sinit, params, mu_init, sigma_init=None):
    """
    R0 for reduced IPM with normal-distributed, log scale load.

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model
    mu_init : float
        Initial mean load on the log scale
    sigma_init : float
        Initial variance in load on the log scale. Defaults to sigma_full in
        params if None is passed.

    Returns
    -------
    : R0

    """

    # Convert to probability scale
    lI = params['lI']
    sl = params['sI']*(1 - lI)

    muF = mu_init

    if sigma_init is None:
        sigma2 = np.exp(params['sigma_full']**2 / 2)
    else:
        sigma2 = np.exp(sigma_init**2 / 2)

    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - sl)
    ratio2 = (sigma2*np.exp(muF)*params['lam']) / (1 - params['nu'])

    R0 = ratio1*ratio2

    return(R0)


def muF_log(params):
    """ Equilibrium mean for reduced IPM on log scale """

    sl = params['sI']*(1 - params['lI'])

    logmean = ((sl*params['a'] + (1 - sl)*params['mu0']) /
               (1 - params['b']*sl))

    return(logmean)


def varF_log(params):
    """
    Equilibrium, initial variance for reduced IPM on log scale.

    Notes
    -----
    Note that ``Reduced IPM'' assumes a FIXED variance.
    This equation calculates equilibrium variance when variance is dynamic.
    """

    muF = muF_log(params)

    a = params['a']
    b = params['b']
    mu0 = params['mu0']
    sigma0 = params['sigma0']
    sigmaG = params['sigmaG']
    sl = params['sI']*(1 - params['lI'])
    num = sl*(sigmaG**2 + a**2 + 2*a*b*muF) + (1 - sl)*(sigma0**2 + mu0**2)
    denom = (1 - sl*b**2)
    second_mom = num / denom
    var_est = second_mom - muF**2

    return(var_est)


def calc_R0_log_sl(Sinit, params, mu_init, sigma_init=None):
    """
    R0 for reduced IPM with normal-distributed, log scale load, and
    load-dependent host survival

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model
    mu_init : float
        The equilibrium mean pathogen load
    sigma_init : float
        Initial variance in load on the log scale. Defaults to sigma_full in
        params if None is passed.

    Returns
    -------
    : R0

    """

    # Convert to probability scale
    if sigma_init is None:
        sigmaF = params['sigma_full']
    else:
        sigmaF = sigma_init

    lI = params['lI']
    alpha = (mu_init - params['mu_s']) / np.sqrt(params['sigma_s']**2 + sigmaF**2)

    sI = stats.norm(loc=0, scale=1).sf(alpha)
    sl = sI*(1 - lI)

    muF = mu_init

    sigma2 = np.exp(sigmaF**2 / 2)

    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - sl)
    ratio2 = (sigma2*np.exp(muF)*params['lam']) / (1 - params['nu'])

    R0 = ratio1*ratio2

    return(R0)


def calc_R0_log_sl_shed(Sinit, params, mu_init):
    """
    R0 for reduced IPM with normal-distributed, log scale load, non-proportional
    shedding, and load-dependent host survival.

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model
    mu_init : float
        The equilibrium mean pathogen load

    Returns
    -------
    : R0

    """

    # Convert to probability scale
    lI = params['lI']
    alpha = (mu_init - params['mu_s']) / params['sigma_s']
    zeta = params['sigma_full'] / params['sigma_s']
    arg1 = alpha / (np.sqrt(1 + zeta**2))

    sI = stats.norm(loc=0, scale=1).sf(arg1)
    sl = sI*(1 - lI)

    muF = mu_init
    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - sl)

    lam_approx = typeII_approx(muF, params['sigma_full'], params['max_shed'])
    ratio2 = (lam_approx) / (1 - params['nu'])

    R0 = ratio1*ratio2

    return(R0)


def calc_R0_log_sl_after(Sinit, params, mu_init, approx=True):
    """
    R0 for reduced IPM with normal-distributed, log scale load, shedding that
    occurs after survival, and load-dependent host survival

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model
    mu_init : float
        The equilibrium mean pathogen load
    approx : bool
        If True, use Taylor expansion approximation for the integral. Otherwise,
        directly evaluate the integral for shedding.

    Returns
    -------
    : R0

    """

    # Convert to probability scale
    lI = params['lI']
    alpha = (mu_init - params['mu_s']) / np.sqrt(params['sigma_s']**2 + params['sigma_full']**2)
    sI = stats.norm(loc=0, scale=1).sf(alpha)
    sl = sI*(1 - lI)

    muF = mu_init
    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - sl)


    if approx:
        lam = aftersurv_approx(muF, params['sigma_full'], params['sigma_s'],
                               params['mu_s'], params['lam'])
    else:
        lam = aftersurv_exact(muF, params['sigma_full'], params['sigma_s'],
                              params['mu_s'], params['lam'])


    ratio2 = (lam) / (1 - params['nu'])

    R0 = ratio1*ratio2

    return(R0)


def calc_R0_nat_sl(Sinit, params, mu_init):
    """
    R0 for reduced IPM with gamma-distributed, natural scale load, and
    load-dependent loss and survival

    Parameters
    ----------
    Sinit: float
        Initial density of susceptible hosts
    params : dict
        Parameters that go into the model

    Returns
    -------
    : R0

    """

    muF = mu_init
    k = params['k']
    γ_s = params['gamma_s']
    γ_l = params['gamma_l']

    sl = h_fxn(γ_s, muF, k) - h_fxn(γ_s + γ_l, muF, k)
    ratio1 = (Sinit*params['trans_beta']*params['s0']) / (1 - sl)
    ratio2 = (muF*params['lam']) / (1 - params['nu'])

    R0 = ratio1*ratio2

    return(R0)


def surv_prob(params, mu_init):
    """
    Average infected host survival probability when survival is load-dependent
    """

    # Convert to probability scale
    alpha = (mu_init - params['mu_s']) / params['sigma_s']
    zeta = params['sigma_full'] / params['sigma_s']
    arg1 = alpha / (np.sqrt(1 + zeta**2))

    sI = stats.norm(loc=0, scale=1).sf(arg1)
    return(sI)


def muF_nat_implicit(M, params):
    """
    Function for implicitly solving for mean pathogen load when survival
    and loss of infection depend on pathogen load
    """

    a = params['a']
    b = params['b']
    k = params['k']
    γ_s = params['gamma_s']
    γ_l = params['gamma_l']

    delta_muF = a*(g_fxn(γ_s, M, k, b) - g_fxn(γ_s + γ_l, M, k, b))
    sl = h_fxn(γ_s, M, k) - h_fxn(γ_s + γ_l, M, k)

    return(delta_muF + (1 - sl)*params['mu0'] - M)


def muF_nat_sl(params, upper=10e5):
    """
    Predicted individual-level mean when survival depends on pathogen load
    """

    a = params['a']
    b = params['b']
    lower = params['mu0']

    if b != 1:
        upper = a**(1 / (1 - b))

    equil = brentq(muF_nat_implicit, 1e-4, upper, args=(params))
    return(equil)


def sl_nat(M, params):
    """
    Survival and no loss of infection for natural scale load
    """
    k = params['k']
    γ_s = params['gamma_s']
    γ_l = params['gamma_l']

    sl = h_fxn(γ_s, M, k) - h_fxn(γ_s + γ_l, M, k)
    return(sl)


def muF_log_implicit(M, params):
    """
    Function for implicitly solving for mean pathogen load when survival
    depends on pathogen load (log scale)
    """

    a = params['a']
    b = params['b']
    mu0 = params['mu0']
    muS = params['mu_s']
    sigmaS = params['sigma_s']
    sigmaF = params['sigma_full']
    lI = params['lI']
    zeta = sigmaF / sigmaS
    alpha = (M - muS) / sigmaS
    gamma = np.sqrt(1 + zeta**2)
    arg1 = alpha / gamma
    arg2 = zeta / np.sqrt(1 + zeta**2)
    norm = stats.norm(loc=0, scale=1)
    sI = norm.sf(arg1)

    return(sI*(1 - lI)*(a + b*M) - (1 - lI)*b*sigmaF*arg2*norm.pdf(arg1) + (1 - sI*(1 - lI))*mu0 - M)


def muF_log_sl(params):
    """
    Predicted individual-level mean when survival depends on pathogen load
    """

    a = params['a']
    b = params['b']
    lower = params['mu0']

    if b != 1:
        upper = a / (1 - b)
    else:
        upper = 100*params['mu_s']

    if lower == upper:
        lower = lower - 0.5
        upper = upper + 0.5

    try:
        equil = brentq(muF_log_implicit, lower, upper, args=(params))
    except ValueError:
        equil = brentq(muF_log_implicit, lower - 2.0, upper, args=(params))

    return(equil)


def muF_varF_log_sl(params):
    """
    Get the mean and variance of the load distribution when survival depends
    on pathogen load

    Parameters
    ----------
    params : Model

    Returns
    -------
    : tuple
        Mean and variance of load distribution
    """

    tmuF = muF_log_sl(params)
    muF, varF = fsolve(muF_varF_log_implicit, (tmuF, tmuF), args=(params,))
    return((muF, varF))


def muF_varF_log_implicit(mv, params):
    """
    Function for implicitly solving for mean pathogen load and variance when
    depends on pathogen load (log scale)

    Parameters
    ----------
    mv : tuple
        Mean and variance
    params : dict
        Model parameters
    """

    M, v = mv  # Mean and variance
    V = v + M**2  # Second moment
    s_init = np.sqrt(v)
    p = params

    alpha = (M - p['mu_s']) / np.sqrt(p['sigma_s']**2 + s_init**2)
    zeta = s_init**2 / np.sqrt(p['sigma_s']**2 + s_init**2)
    sI = 1 - stats.norm.cdf(alpha)
    sIx = M - (zeta*stats.norm.pdf(alpha) + M*stats.norm.cdf(alpha))
    dp = double_probit_approx(M - p['mu_s'],
                              mu_param(s_init),
                              s_param(s_init, p['sigma_s']))
    sIx2 = (s_init**2 + M**2) - (s_init**2*dp + 2*M*zeta*stats.norm.pdf(alpha) + M**2 * stats.norm.cdf(alpha))

    Meq = ((1 - p['lI'])*sI*(p['a'] + p['b']*(M - zeta*stats.norm.pdf(alpha) / sI)) + (1 - (1 - p['lI'])*sI)*p['mu0']) - M

    Veq = ((1 - p['lI'])*(p['sigmaG']**2 * sI + p['a']**2 * sI + 2*p['a']*p['b']*sIx + p['b']**2 * sIx2) +
           (p['sigma0']**2 + p['mu0']**2)*(1 - (1 - p['lI'])*sI)) - V

    return((Meq, Veq))


def mu_param(sigma_f):
    """
    Mean parameter of double probit as a function of sigma_f

    Parameters
    ----------
    sigma_f : Variance of the distribution
    """
    return(1.566*sigma_f)


def s_param(sigma_f, sigma_s):
    """
    Shape parameter of double probit as a fxn of s

    Parameters
    ----------
    sigma_f : Variance of the distribution
    sigma_s : Slope of the probit survival fxn
    """
    return(0.99*sigma_s + 0.5927*sigma_f + -0.0846*sigma_f*sigma_s)


def double_probit_approx(delta, mu, s):
    """
    Approximation to the integral

    int_-infty^infty x**2 Phi(a + bx) phi(x) dx

    where Phi is the cdf of a standard normal, phi is the pdf of a standard
    normal, a = (muF - muS) / sigma_s and b = (sigma_f / sigma_s)
    """
    p1 = stats.norm(loc=-mu, scale=s).cdf(delta) * 0.5
    p2 = stats.norm(loc=mu, scale=s).cdf(delta) * 0.5
    approx = p1 + p2
    return(approx)


def simulate_base_ipm(params, steps):
    """
    Simulate the individual load distribution from an IPM with load-independent
    survival. This is the distribution whose mean we are estimating with
    M_1^*.

    Parameters
    ----------
    params_red : dict
        Parameters of the reduced IPM
    steps : int
        Number of simulation steps to run. Used to approximate the load distribution.

    Returns
    -------
    : array
        Distribution of loads
    """

    loads = np.empty(steps + 1)
    loads[:] = np.nan
    infected = np.empty(steps + 1)
    infected[0] = 0
    norm0 = stats.norm(loc=params['mu0'], scale=params['sigma0'])

    for i in range(1, steps + 1):

        # Uninfected
        if infected[i - 1] == 0:
            infected[i] = 1  # Gain infection immediately.  Transmission doesn't matter, just loss of infection and survival

            # Initial infection load
            if infected[i] == 1:
                loads[i] = norm0.rvs()
        else:
            # Survive and don't lose infection
            infected[i] = stats.binom(1, params['sI']*(1 - params['lI'])).rvs()

            # Pathogen growth
            if infected[i]:
                loads[i] = stats.norm.rvs(loc=params['a'] + params['b']*loads[i - 1], scale=params['sigmaG'])

    return(loads[~np.isnan(loads)])


def simulate_surv_ipm(params, steps):
    """
    Simulate the individual load distribution from an IPM with load-dependent
    survival. This is the load distribution whose mean we are estimating with
    M_2^*

    Parameters
    ----------
    params_red : dict
        Parameters of the reduced IPM
    steps : int
        Number of simulation steps to run. Used to approximate the load distribution.

    Returns
    -------
    : array
        Distribution of loads
    """
    loads = np.empty(steps + 1)
    loads[:] = np.nan
    infected = np.empty(steps + 1)
    infected[0] = 0
    norm0 = stats.norm(loc=params['mu0'], scale=params['sigma0'])

    for i in range(1, steps + 1):

        # Uninfected
        if infected[i - 1] == 0:
            infected[i] = 1  # Gain infected immediately. Transmission doesn't matter, just loss of infection and survival

            # Initial infection load
            if infected[i] == 1:
                loads[i] = norm0.rvs()
        else:
            # Survive and don't lose infection
            sI = stats.norm(loc=params['mu_s'], scale=params['sigma_s']).sf(loads[i - 1])
            infected[i] = stats.binom(1, sI*(1 - params['lI'])).rvs()

            # Pathogen growth
            if infected[i]:
                loads[i] = stats.norm.rvs(loc=params['a'] + params['b']*loads[i - 1],
                                          scale=params['sigmaG'])

    return(loads[~np.isnan(loads)])

