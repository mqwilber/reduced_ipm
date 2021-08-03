import numpy as np
import scipy.stats as stats

"""Classes and functions of simulating and exploring the full host-parasite IPM"""


class Community:
    """
    Community class with host species.  Allows for the inclusion of multiple
    host species in the model.
    """

    def __init__(self, name, comm_params, ipm_params, pathogen_scale="log"):
        """
        Parameters
        ----------
        comm_params :
            'time': int, the starting time. 1 is a good default.
            'species': dict, keywords are species names and each keyword
                       looks up a dictionary with species specific parameters
                       needed for the Species class.
            'density': dict, keywords are species and each looks up a vector
                       of initial densities for that species.
        ipm_params : dict
            'min_size': The lower bound of the IPM integral
            'max_size': The upper bound of the IPM integral
            'bins': Number of discretized classes in the IPM integral
            'time_step': The number of days in a time step
        pathogen_scale: str
            "log":  Model pathogen load on the log scale. Otherwise,
                    model pathogen load on the natural scale assuming a
                    gamma distribution.
        """

        self.species = []
        for name, spp_params in comm_params['species'].items():
            self.species.append(Species(name, spp_params, ipm_params,
                                        comm_params['density'][name],
                                        pathogen_scale))

        self.time = comm_params['time']
        self.update_zoospore_pool()
        self.ipm_params = ipm_params

    def update_zoospore_pool(self):
        """ Calculate the zoospore pool """
        self.zpool = np.sum([spp.density_vect[-1] for spp in self.species])

    def __str__(self):
        """ String representation of community object """
        return("\n".join([spp.__str__() for spp in self.species]))

    def update_time(self):
        self.time = self.time + self.ipm_params['time_step']

    def update_deterministic(self):
        """ Update all species one time-step, deterministically """

        [spp.update_deterministic(self.zpool, self.time)
            for spp in self.species]

        self.update_zoospore_pool()
        self.update_time()

    def update_stochastic(self):
        """ Update all species one time-step, stochastically """

        [spp.update_stochastic(self.zpool, self.time)
            for spp in self.species]

        self.update_zoospore_pool()
        self.update_time()

    def simulate(self, steps, stochastic=False):
        """
        Simulate the community for steps time steps either stochastically or
        deterministically

        Returns
        -------
        : dict
            Key word for each species in the community looking on S x steps + 1
            matrix with density vector at each time step.
        """

        ts = self.ipm_params['time_step']
        time_vals = np.arange(0, steps*ts + ts, step=ts)

        # Initial arrays to hold results
        species_res = {}
        for spp in self.species:

            all_res = np.empty((spp.ipm_params['bins'] + 2, steps + 1))
            all_res[:, 0] = np.copy(spp.density_vect)  # Initialize
            species_res[spp.name] = all_res

        if stochastic:
            for i in range(steps):
                self.update_stochastic()

                # This could be sped up I am guessing
                for spp in self.species:
                    species_res[spp.name][:, i + 1] = spp.density_vect
        else:
            for i in range(steps):
                self.update_deterministic()

                # This could be sped up I am guessing
                for spp in self.species:
                    species_res[spp.name][:, i + 1] = spp.density_vect

        return((time_vals, species_res))


class Species:
    """
    Host species object that builds an IPM for a particular host species
    """

    def __init__(self, name, spp_params, ipm_params, init_density_vect,
                 pathogen_scale="log"):
        """
        Parameters
        ----------

        spp_params : dict
            'growth_fxn_*':
                - 'inter': The intercept of the growth function
                - 'slope': The effect of previous Bd load on current Bd load
                - 'sigma': The standard deviation of the Bd growth function
            'loss_fxn_*':
                - 'inter': The intercept of the Bd loss probability function
                - 'slope': The effect of current Bd load on log odds loss
            'init_inf_fxn_*':
                - 'inter': The intercept of the initial infection function
                - 'sigma': The standard deviation in the initial infection load
            'surv_fxn_*': dict('inter', 'slope', 'temp')
                - 'inter': The LD50 of the survival function
                - 'slope': Inverse measure of the steepness of the survival function
            'constant_surv': If True, infected survival probability is constant
                             with probability s_I
            'constant_loss': If True, loss of infection is constant with probability
                             lI
            'shedding_prop': Proportionality constant for zoospore shedding from
                             infected individual.
            'trans_fxn_*':
                - 'zpool': The transmission coefficient between zoospore pool
                           density and infection probability in a time step.
            'nu': Zoospore survival probability

        ipm_params : dict
            'min_size': The lower bound of the IPM integral
            'max_size': The upper bound of the IPM integral
            'bins': Number of discretized classes in the IPM integral
        """

        self.name = name
        self.spp_params = spp_params  # Holds parameters related to species biology
        self.ipm_params = ipm_params  # Holds parameters related to IPM implementation
        self.density_vect = init_density_vect

        # Make ipm parameters
        ipm_params = set_discretized_values(ipm_params['min_size'],
                                            ipm_params['max_size'],
                                            ipm_params['bins'])

        self.y = ipm_params['y']  # midpoints
        self.h = ipm_params['h']  # Width of interval
        self.bnd = ipm_params['bnd']
        self.matrix_error = False
        self.pathogen_scale = pathogen_scale

    def build_ipm_matrix(self):
        """
        Function to build the IPM portion of the host-parasite IPM
        """
        #
        X, Y = np.meshgrid(self.y, self.y)
        _, Y_upper = np.meshgrid(self.y, self.bnd[1:])
        _, Y_lower = np.meshgrid(self.y, self.bnd[:-1])
        G = _growth_fxn(Y_lower, Y_upper, X, self.spp_params, self.pathogen_scale)
        S = _survival_fxn(self.y, self.spp_params, self.pathogen_scale)
        L = _loss_fxn(self.y, self.spp_params, self.pathogen_scale)

        P = np.dot(G, np.diagflat(S*(1 - L)))

        # All column sums should be less than 1
        if np.any(P.sum(axis=0) > 1):
            self.matrix_error = True

        # Save the intermediate kernels results
        self.S = S
        self.L = L
        self.G = G
        self.P = P

    def build_full_matrix(self, zpool, time):
        """
        The full transition matrix that includes susceptible hosts and
        the zoospore pool. Used to update the model one-time step within a
        season.

        Parameters
        ----------
        zpool: float
            The density of zoospores in the pool
        """

        self.build_ipm_matrix()

        inf_prob = _trans_fxn(zpool, self.spp_params, self.pathogen_scale)

        y_lower = self.bnd[:-1]
        y_upper = self.bnd[1:]
        init_inf = _init_inf_fxn(y_lower, y_upper,
                                 self.spp_params, self.pathogen_scale)

        # Susceptible -> infected
        col0 = np.r_[self.spp_params['surv_sus']*(1 - inf_prob),
                     self.spp_params['surv_sus']*inf_prob*init_inf]

        # Infected -> susceptible
        row0 = np.r_[self.spp_params['surv_sus']*(1 - inf_prob),
                     self.S*self.L]

        # Zoospore production and death
        zsurv = self.spp_params['nu']

        if self.pathogen_scale == 'log':
            rowminus1 = np.r_[0, self.spp_params['shedding_prop']*np.exp(self.y), zsurv]
        else:
            rowminus1 = np.r_[0, self.spp_params['shedding_prop']*self.y, zsurv]

        colminus1 = np.r_[np.repeat(0, len(self.P) + 1), zsurv]

        # Add Susceptible and zoospore vectors to matrix
        T = np.empty(np.array(self.P.shape) + 2)
        T[:-1, 0] = col0
        T[0, :-1] = row0
        T[1:-1, 1:-1] = self.P
        T[-1, :] = rowminus1
        T[:, -1] = colminus1

        self.T = T

        # Add seasonal reproduction.
        self.F = np.zeros(T.shape)
        year_time = (time % 365)
        lower = year_time - self.ipm_params['time_step']
        if ((self.spp_params['repro_time'] > lower) &
           (self.spp_params['repro_time'] <= year_time)):

            n = np.sum(self.density_vect[:-1])
            repro = self.spp_params['fec']*np.exp(-self.spp_params['K']*n)
            full_repro = np.r_[np.repeat(repro, T.shape[0] - 1), 0]
            self.F[0, :] = full_repro

    def model_R0(self, Sinit):
        """
        Calculate R0 for the full IPM mode

        Parameters
        ----------
        Sinit : float
            Number of susceptible individuals at disease-free equilibrium

        Returns
        -------
        : tuple
            (next generation matrix, R0)
        """

        # U matrix
        self.build_ipm_matrix()
        Ured = np.copy(self.P)

        if self.pathogen_scale == "log":
            row1 = np.exp(self.y)*self.spp_params['shedding_prop']
        elif self.pathogen_scale == "natural":
            row1 = self.y*self.spp_params['shedding_prop']

        U1 = np.vstack((Ured, row1))
        col1 = np.r_[np.zeros(len(row1)), self.spp_params['nu']][:, np.newaxis]
        U = np.hstack((U1, col1))

        # Build reproduction matrix
        F = np.zeros(U.shape)
        y_lower = self.bnd[:-1]
        y_upper = self.bnd[1:]
        init_inf = _init_inf_fxn(y_lower, y_upper, self.spp_params, self.pathogen_scale)
        init_col = Sinit*self.spp_params['surv_sus']*self.spp_params['trans_fxn_zpool']*init_inf
        F[:-1, -1] = init_col

        minusUinv = np.linalg.inv(np.eye(len(U)) - U)
        Rmat = np.dot(F, minusUinv)

        return((Rmat, np.max(np.linalg.eigvals(Rmat))))

    def update_deterministic(self, zpool, time):
        """
        One time-step update, deterministic
        """

        self.build_full_matrix(zpool, time)
        self.density_vect = np.dot(self.F + self.T, self.density_vect)

    def update_stochastic(self, zpool, time):
        """
        One time-step update, stochastic
        """

        # Create
        self.build_full_matrix(zpool, time)

        T = self.T.copy()
        T[-1, :-1] = 0  # Remove zoospore fertility but keep survival
        death_probs = 1 - T.sum(axis=0)
        death_probs[death_probs < 0] = 0
        Taug = np.vstack([T, death_probs])  # Add death class
        Taug = Taug / Taug.sum(axis=0)

        # Production of zoospores. Using the fact that a sum of Poissons
        # is Poisson.
        Zrepro = self.T[-1, 1:-1]
        inf_hosts = np.ceil(self.density_vect[1:-1]).astype(np.int)
        gained_z = stats.poisson(np.sum(Zrepro*inf_hosts)).rvs(size=1)

        # Moving to new infection, non-infected classes

        updated = np.array([np.random.multinomial(n, p) for n, p in
                            zip(self.density_vect.astype(np.int), Taug.T)])
        new_density = updated.sum(axis=0)[:-1]  # Don't track dead individuals
        new_density[-1] = new_density[-1] + gained_z

        # Add new susceptible individuals from reproduction
        n = self.density_vect[:-1]
        repro = stats.poisson(np.sum(self.F[0, :-1]*n)).rvs(size=1)
        new_density[0] = new_density[0] + repro

        self.density_vect = new_density

    def get_mean_intensity(self):
        """ Get mean log Bd intensity from species """

        load_dist = self.density_vect[1:-1]
        load_dist = load_dist / np.sum(load_dist)
        mean_log_load = np.sum(self.y * load_dist)
        return(mean_log_load)

    def get_prevalence(self):
        """ Get the Bd prevalence """

        uninf = self.density_vect[0]
        inf = np.sum(self.density_vect[1:-1])
        return(inf / (inf + uninf))

    def __str__(self):
        return("Species name: {0}\nCurrent density: {1}".format(
               self.name, np.sum(self.density_vect[:-1])))


def _growth_fxn(x_next_lower, x_next_upper, x_now, params, scale):
    """ The Bd growth function """

    max_load = params['max_load']
    min_load = params['min_load']
    x_now[x_now > max_load] = max_load
    x_now[x_now < min_load] = min_load

    if scale == "log":
        μ = params['growth_fxn_inter'] + params['growth_fxn_slope']*x_now
        σ = params['growth_fxn_sigma']
        norm_fxn = stats.norm(loc=μ, scale=σ)
        prob = norm_fxn.cdf(x_next_upper) - norm_fxn.cdf(x_next_lower)
    elif scale == 'natural':
        μ = params['growth_fxn_inter']*x_now**(params['growth_fxn_slope'])
        k = params['growth_fxn_k']
        gamma_fxn = stats.gamma(k, scale=μ / k)
        prob = gamma_fxn.cdf(x_next_upper) - gamma_fxn.cdf(x_next_lower)
    else:
        raise KeyError("Scale {0} not recognized: use 'log' or 'natural'".format(scale))

    return(prob)


def _survival_fxn(x_now, params, scale):
    """ The host survival function """

    if not params['constant_surv']:

        if scale == 'log':
            u = params['surv_fxn_inter']
            prob = stats.norm(loc=u, scale=params['surv_fxn_slope']).sf(x_now)
        elif scale == 'natural':
            prob = np.exp(-params['surv_gamma']*x_now)
    else:
        prob = np.repeat(params['sI'], len(x_now))

    return(prob)


def _loss_fxn(x_now, params, scale):
    """ The loss of infection function """

    if not params['constant_loss']:

        if scale == 'log':
            u = params['loss_fxn_inter'] + params['loss_fxn_slope']*x_now
            prob = 1 / (1 + np.exp(-u))
        elif scale == 'natural':
            prob = np.exp(-params['loss_gamma']*x_now)
    else:
        prob = np.repeat(params['lI'], len(x_now))

    return(prob)


def _trans_fxn(zpool, params, scale):
    """ The transmission function """

    beta = params['trans_fxn_zpool']
    return(1 - np.exp(-(beta*zpool)))


def _init_inf_fxn(x_next_lower, x_next_upper, params, scale):
    """ The initial infection function """

    if scale == 'log':
        μ = params['init_inf_fxn_inter']
        σ = params['init_inf_fxn_sigma']
        norm_fxn = stats.norm(loc=μ, scale=σ)
        prob = norm_fxn.cdf(x_next_upper) - norm_fxn.cdf(x_next_lower)
    elif scale == 'natural':
        μ = params['init_inf_fxn_inter']  # Natural scale
        k = params['init_inf_fxn_k']
        gamma_fxn = stats.gamma(k, scale=μ / k)
        prob = gamma_fxn.cdf(x_next_upper) - gamma_fxn.cdf(x_next_lower)
    else:
        raise KeyError("Scale {0} not recognized: use 'log' or 'natural'".format(scale))

    return(prob)


def set_discretized_values(min_size, max_size, bins):
    """
    Calculates the necessary parameters to use the midpoint rule to evaluate
    the IPM model

    Parameters
    ----------
    min_size : The lower bound of the integral
    max_size : The upper bound of the integral
    bins : The number of bins in the discretized matrix

    Returns
    -------
    dict: min_size, max_size, bins, bnd (edges of discretized kernel), y (midpoints),
    h (width of cells)
    """

    # Set the edges of the discretized kernel
    bnd = min_size + np.arange(bins + 1)*(max_size-min_size) / bins

    # Set the midpoints of the discretizing kernel. Using midpoint rule for evaluation
    y = 0.5 * (bnd[:bins] + bnd[1:(bins + 1)])

    # Width of cells
    h = y[2] - y[1]

    return(dict(min_size=min_size,
                max_size=max_size,
                bins=bins, bnd=bnd, y=y,
                h=h))




