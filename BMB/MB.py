#!/usr/bin/env python3
# coding: utf-8

# scipy stack
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.stats.mstats as mstats
import rtnorm
import itertools

# misc
from abc import ABC, abstractmethod
from BMB import util

import gig

# debugging
import logging
import sys
import os


def block_cholesky_BWD(B, W, D):
    """Returns upper cholesky factor L^t of the Matrix C=kron(W,B)+diag(D).
    """
    K = W.shape[1]
    n_K = B.shape[0]
    L_d = n_K * K
    L_t = np.zeros((L_d, L_d))

    for start in range(K):
        A = W[start, start] * B + np.diag(D[start, ])

        L_start = np.linalg.cholesky(A)
        L_row_t = L_start.T
        block_unweighted = linalg.solve_triangular(L_start, B, lower=True)
        if((start+1) < K):
            l = np.empty(
                (K-(start), block_unweighted.shape[0], block_unweighted.shape[1]))
            l[0, :, :] = L_row_t
            for j in range(1, l.shape[0]):
                l[j, :, :] = W[start, start+j] * block_unweighted
            L_row_t = np.concatenate(l, axis=1)
        L_t[start*n_K:(start+1)*n_K, start*n_K:L_d] = L_row_t
    return(L_t)


class BMB(ABC):
    def __init__(self, X, p, q,
                 lambda_,
                 burnin,  NSCAN,
                 DO_COPULA=True, plugin_threshold=40,
                 TRUE_INVCOV=None, gig_seed=4321, use_w=False,
                 Z_save_each = -1):
        """Set options, data and initial values for the BMB.

        Arguments:
            X {ndarray} -- Data
            p {int} -- Number of query variables
            q {int} -- number of non-query variables
            burnin {int} -- Burnin of MCMC sampler
            lambda_ {float} -- Sparsity hyperparameter

        Keyword Arguments:
            NSCAN {int} -- Number of MCMC iterations (default: {1000})
            plugin_threshold {int} --  (default: {40})
            TRUE_INVCOV {ndarray} -- If available, the true inverse covariance matrix. Used for diagnostics. (default: {None})
            DO_COPULA {bool} -- Wrap semi-parametric copula around the Gibbs sampler (default: {False})

        """
        assert(np.shape(X)[1] == p+q)
        assert(q > 0 and p > 0 and lambda_ > 0)
        assert(burnin > 0 and NSCAN > 0)

        self.DO_COPULA = DO_COPULA

        self.TRUE_INVCOV = TRUE_INVCOV
        self.p = p
        self.q = q
        self.num_vars = self.p + self.q
        self.n = np.shape(X)[0]
        self.use_w = use_w

        self.lambda_ = lambda_
        self.NSCAN = NSCAN
        self.burnin = burnin
        self.X = X

        self.gig_seed = gig_seed
        self.GIG = gig.GIG(gig_seed)

        # initialize S, W and T.inv
        self.T_inv = np.random.normal(size=p*q).reshape(p, q)

        if(DO_COPULA):
            self.Z, self.S = util.cov_from_normscores(
                X, scaled=False, fill_na=True)
        else:
            self.Z, self.S = util.cov_from_normscores(
                X, scaled=True, fill_na=False)
            self.S = self.S * self.n

        self.W0 = np.eye(p+q) * self.lambda_
        if(self.n < self.S.shape[0]):
            a = self.S.shape[0] - self.n + 1
        else:
            a = 0
        self.W = stats.wishart.rvs(
            df=self.n+a, scale=util.cholesky_invert(self.S+self.W0))

        # keep track of intermediate results
        self.W_list = np.zeros((NSCAN, self.W.shape[0], self.W.shape[1]))
        self.W_list_burnin = self.W_list[0:burnin, :, :]
        self.W_list_acc = self.W_list[burnin:NSCAN, :, :]
        self.T_inv_list = np.zeros((NSCAN, p, q))
        self.T_inv_list_acc = self.T_inv_list[burnin:NSCAN, :, :]

        self.Z_save_each = Z_save_each
        if(self.Z_save_each>0):
            self.Z_saves = np.zeros((np.floor_divide((self.NSCAN-self.burnin-1), self.Z_save_each), self.Z.shape[0], self.Z.shape[1]))
            
        self.Z_mean = np.zeros((self.Z.shape[0], self.Z.shape[1]))
        self.Z_ssd = np.zeros((self.Z.shape[0], self.Z.shape[1]))
        if(DO_COPULA):
            # get the levels and number of levels
            # for the copula
            self.R = BMB.calc_R(self.X)
            self.Rlevels = np.amax(self.R, axis=0)

            self.plugin_marginal = np.apply_along_axis(
                lambda col: np.unique(col).shape[0] > plugin_threshold, 0, X)*1

    @staticmethod
    def calc_R(X):
        """ Calculate row-wise ranks of X.

        Arguments:
            X {[type]} -- [description]
        """
        R = None
        for j in range(X.shape[1]):
            # np.unique "Returns the sorted unique elements of an array."
            _uqs, inv_idx = np.unique(X[:, j], return_inverse=True)
            # NAs encoded as minus one, as R[i,j] >=0 forall i,j
            # inv_idx = inv_idx.astype('float')
            inv_idx[np.argwhere(np.isnan(X[:, j]))] = -1
            if (R is None):
                R = inv_idx
            else:
                R = np.column_stack((R, inv_idx.astype(np.int)))
        return(R)

    @staticmethod
    def _draw_MGIG_inv(n_, A, B, num_its=10):
        """Returns the inverse of a MGIG draw via a continued fraction of Wishart draws.
        (X~MGIG(n'=0.5(n+n0+2p-2), W12*(S22+I)W21, S11+I) and W11 = X^-1)

        Arguments:
            n_ {int} -- [Degrees of Freedom]
            A {ndarray} -- []
            B {ndarray} -- []

        Keyword Arguments:
            num_its {int} -- Number of iterations (default: {20})
        """
        p = A.shape[0]
        B_inv = util.cholesky_invert(B)
        eigvals = np.linalg.eigvalsh(A)
        if(np.sum(eigvals>1e-5)<p):
            A = A + (np.min(np.abs(eigvals))+0.01)*np.identity(p)
        A_inv = util.cholesky_invert(A)
        # draw from W(df = n + n_0 + 3p - 3, (W_12(S_22 + I)W_21)^-1
        # As we are only given n' of the MGIG for W_11^-1, we need to express it in terms of this with
        # df = 2*n' + p - 1
        df = 2*n_+p-1
        # MGIG dfs my version
        # df = 2*n_
        X = np.zeros(shape=np.shape(A))

        Y1 = stats.wishart.rvs(df=df,
                               scale=A_inv,
                               size=num_its)
                               
        Y2 = stats.wishart.rvs(df=df,
                               scale=B_inv,
                               size=num_its)

        for i in range(0, num_its):
            # This would converge to W_11^-1, but we need W_11, so don't do the last inversion
            # (see notes at 6.2, "Sampling from the MGIG")
            if(i > 0):
                X = util.cholesky_invert(X)
            X = Y1[i] + util.cholesky_invert(Y2[i] + X)
        return(X)

    def _draw_latent_scores(self, Z, R, Rlevels, W, n, plugin_marginal):
        """Draw scores from the latent variables of the copula.

        Arguments:
            Z {np.ndarray[np.double_t, ndim=2]} -- [description]
            R {np.ndarray[np.int_t, ndim=2]} -- [description]
            Rlevels {np.ndarray[np.int_t, ndim=1]} -- [description]

        Returns:
            [type] -- [description]
        """

        idx1 = np.zeros((Z.shape[1],), dtype=bool)
        perm = np.random.permutation(Z.shape[1])

        sampler = rtnorm.rtnorm()

        # NOTE/REMINDER: nas are encoded as -1 in self.R
        has_nans = np.any(R == -1)
        var = 1/W.diagonal()
        sd = np.sqrt(var)
        for _a in range(perm.shape[0]):
            i = perm[_a]

            idx1[i] = 1
            mu = Z[:, ~idx1] @(W[~idx1, idx1] * (-var[i]))
            idx1[i] = 0

            if (not plugin_marginal[i]):
                _ir_allr = [R[:,i] == r for r in range(Rlevels[i])] + [()]
                for r in range(Rlevels[i]):

                    # set lower bound to maximum Z score of values corresponding to the next lower rank
                    _ir = _ir_allr[r-1]#(R[:, i] == r-1) & (R[:, i] != -1)
                    if(not np.any(_ir)):
                        lb = -np.inf
                    else:
                        lb = np.max(Z[_ir, i])

                    # set upper bound to minimum Z score of values corresponding to the next higher rank
                    _ir = _ir_allr[r+1]#(R[:, i] == r+1) & (R[:, i] != -1)
                    if(not np.any(_ir)):
                        ub = np.inf
                    else:
                        ub = np.min(Z[_ir, i])

                    # set the scores of the values corresponding to the current rank to a draw from a truncated normal, bounded as shown above
                    _ir = _ir_allr[r]#(R[:, i] == r) & (R[:, i] != -1)
                    # a, b = (lb - mu[_ir]) / sd[i], (ub - mu[_ir]) / sd[i]
                    # Z[_ir, i] = stats.truncnorm.rvs(a, b)
                    Z[_ir, i] = sampler.cyrtnormClass.sample(
                        a=np.array([lb]), b=np.array([ub]), mu=mu[_ir], sigma=np.array([sd[i]]))

            if(has_nans):
                _ir = (R[:, i] == -1)
                Z[_ir, i] = np.random.normal(loc=mu[_ir], scale=sd[i])
            # update the scores..
            ranks = mstats.rankdata(Z[:, i])
            Z[:, i] = stats.norm.ppf(ranks/(n+1))

        return Z


    def _draw_T_inv(self, w12, p, q, lambda_, GIG, T=1):
        """ Draw IG distributed sparsity hyperparameters via GIG.

        Returns:
            [Matrix] -- [(p x q) T matrix of the hyperparameters.]
        """

        mus = lambda_/np.abs(w12.flatten())
        lambdas = np.square(lambda_)*np.ones(p*q)

        # T_inv = np.random.wald(mus, lambdas)

        a = (lambdas/np.square(mus))/T
        b = lambdas/T
        p_ = -1.5/T + 1 * np.ones(p*q)

        T_inv = GIG.sample(lambda_=p_, chi=b, psi=a)
        return (T_inv.reshape(p, q))

    def _draw_W11(self, S, W, p, q, n, lambda_, T=1):
        """Draw from the W11 posterior conditional using an MGIG distribution.

        Arguments:

            S {ndarray} -- Sample Covariance / Covariance of scores
            W {ndarray} -- Most current instance of W.
            p {int} -- Number of query variables.
            q {int} -- Number of non-query variables
            n {int} -- Number of samples.
            lambda_ {[type]} -- [description]
        """
        identity11 = np.eye(N=p, M=p)
        identity22 = np.eye(N=q, M=q)
        #n_ = (n+p+1)/2
        A = (W[0:p, p:(p+q)]@(S[p:(p+q), p:(p+q)] +
                              lambda_*identity22)@W[0:p, p:(p+q)].T)
        B = S[0:p, 0:p]+lambda_*identity11

        W11 = BMB._draw_MGIG_inv(
            # notes
            n_=0.5*n/T+p,
            # n_=0.5*(n + n0 + 2*p - 2),
            # MGIG my ver
            # n_=(n_+(p+1)/2)/T - (p+1)/2,
            A=A/T,
            B=B/T
        )
        return(W11)

    def _draw_W11_Wishart(self, S, W, p, q, n, lambda_, T=1):
        """Draw from W11 Posterior scaled by the Temperature T: P(W11|W21,T_inv,...)^(T)

        DEPRECATED

        Arguments:
            S {[type]} -- [description]
            W {[type]} -- [description]
            p {[type]} -- [description]
            q {[type]} -- [description]
            n {[type]} -- [description]
            lambda_ {[type]} -- [description]

        Keyword Arguments:
            T {int} -- [description] (default: {1})
        """
        s11 = S[0:p, 0:p]
        s12 = S[0:p, p:(p+q)]
        s22 = S[p:(p+q), p:(p+q)]
        s22_lambda_inv = util.cholesky_invert(s22 + np.eye(N=q, M=q)*lambda_)

        n_ = (3*n-2*q-1)/3
        Sigma = s11-s12 @ s22_lambda_inv @ s12.T + np.eye(p)
        # updated df due to SA cooling
        scale = T*Sigma
        df = (n_-scale.shape[0]-1)/T + scale.shape[0] + 1

        W11 = stats.wishart.rvs(df=df, scale=scale)
        return(W11)

    def _draw_W12(self, S, W, p, q, T_inv, lambda_, T=1):
        """Draw from W12 Posterior scaled by the Temperature T: P(W12|W11,T_inv,...)^(T)

        Arguments:
            T_inv {[type]} -- [description]
            w11 {[type]} -- [description]
            s12 {[type]} -- [description]
            s22 {[type]} -- [description]
            p {[type]} -- [description]
            q {[type]} -- [description]
            T {float} -- SA cooling parameter
        """
        w11 = W[0:p, 0:p]
        s12 = S[0:p, p:(p+q)]
        s22 = S[p:(p+q), p:(p+q)]

        w11_inv = util.cholesky_invert(w11)

        identity = np.eye(N=q, M=q)

        # block cholesky decomposition
        L_T = block_cholesky_BWD(
            s22+lambda_ * identity, w11_inv, T_inv)
        L = L_T.T

        # method as in Notes, in my opinion a bit clearer than in initial R-code(both are ultimately similar)
        # NOTE: flatten is column major in numpy, so no need to transpose s12 prior to flattening.
        v = (s12).flatten()
        y = linalg.solve_triangular(a=L, b=-v, lower=True)
        mu = linalg.solve_triangular(a=L_T, b=y)
        r = sp.random.standard_normal(size=(p*q))
        b = linalg.solve_triangular(a=L_T, b=r)
        # cooling parameter
        b = b * np.sqrt(T)

        w12 = (mu + b).reshape(p, q)
        return(w12)

    def run(self, debug_ratio=5):
        """Start the sampling procedure to infer the Markov Blanket.

        Returns:
            [BMB] -- The BMB object it is applied on.
        """
        p = self.p
        q = self.q
        n = self.n

        lambda_ = self.lambda_
        NSCAN = self.NSCAN

        W = self.W
        S = self.S
        T_inv = self.T_inv
        Z = self.Z

        DO_COPULA = self.DO_COPULA
        update_scores = False

        if(DO_COPULA):
            update_scores = (np.sum(np.logical_not(self.plugin_marginal)) > 0
                             or np.any(np.isnan(self.X)))
            R = self.R
            Rlevels = self.Rlevels

        #########################################
        counter_Z_saves = 0
        for it in range(NSCAN):
            if(it % np.floor_divide(NSCAN, debug_ratio) == 0):
                logging.debug("Gibbs Iteration # %d" % (it))

            # update Z scores and then S
            if(update_scores):
                Z = self._draw_latent_scores(
                    Z, R, Rlevels, W, n, self.plugin_marginal)
                # Z = np.array(R_scores.draw_latent_scores(Z, R, Rlevels, W, self.plugin_marginal))
                S = Z.T @ Z

            ################## T^-1 ##################
            T_inv = self._draw_T_inv(w12=W[0:p, p:(p+q)],
                                     p=p, q=q,
                                     lambda_=lambda_, GIG=self.GIG)

            ################## W11 ###################
            if(self.use_w):
                self._draw_W11_Wishart(S=S, W=W,
                                       p=p, q=q, n=n,
                                       lambda_=lambda_)
            else:
                W[0:p, 0:p] = self._draw_W11(S=S, W=W,
                                             p=p, q=q, n=n,
                                             lambda_=lambda_)

            ################## W12 ###################
            W[0:p, p:(p+q)] = self._draw_W12(S=S, W=W,
                                             p=p, q=q,
                                             T_inv=T_inv,
                                             lambda_=lambda_)
            ############### W21=W12^T ################
            W[p:(p+q), 0:p] = W[0:p, p:(p+q)].T

            ################## W22 ###################
            if(update_scores):
                W[p:(p+q), p:(p+q)] = self.draw_W22(S=S, W=W,
                                                    p=p, q=q, n=n,
                                                    lambda_=lambda_)
            else:
                W[p:(p+q), p:(p+q)] = np.eye(q)
            # keep track of results
            self.W_list[it, :, :] = W
            self.T_inv_list[it, :, :] = T_inv
            if(it>self.burnin):
                # e.g. http://cas.ee.ic.ac.uk/people/dt10/research/thomas-08-sample-mean-and-variance.pdf
                d = (Z - self.Z_mean) 
                # M_i
                self.Z_mean = self.Z_mean + d/(it+1) 
                # S_i sum of squared differences
                self.Z_ssd = self.Z_ssd + d*(Z - self.Z_mean)
                
                if(self.Z_save_each>0 and (it-self.burnin)% self.Z_save_each == 0):
                    self.Z_saves[counter_Z_saves, :, :] = self.Z_mean
                    counter_Z_saves = counter_Z_saves + 1 
        self.W = W
        self.Z = self.Z_mean
        self.S = S
        self.T_inv = T_inv

        return(self)

    def draw_W22(self, S, W, p, q, n, lambda_):
        # W22 = Wishart(n, (S22+lambda*I)^-1) + W21*(W11)^-1 * W12
        if (n < q):
            a = q - n + 1
        else:
            a = 0
        s22_lambda_inv = util.cholesky_invert(
            S[p:(p+q), p:(p+q)] + np.eye(N=q, M=q)*lambda_)
        tmp = stats.wishart.rvs(
            df=n+a, scale=s22_lambda_inv)
        W22 = tmp + \
            W[0:p, p:(p+q)].T @ np.linalg.solve(W[0:p, 0:p], W[0:p, p:(p+q)])
        return(W22)

    @abstractmethod
    def threshold_matrix(self, thresh_low=0.2):
        """ Returns estimation of the precision matrix at W12.

        Thresholds by looking at the Credible Interval with P(CI)=1-thresh_low,
        i.e. we look at the [thresh_low, 1-thresh_low] quantiles. 

        If the CI for a w12 posterior marginal contains zero, it estimated to be zero. 
        Else, the value is estimated by the 0.5 quantile.

        For the standard Gibbs BMB we look at all accepted samples (everything but the burnin).
        For SA, we look at the samples drawn at the final temperature. 

        Raises:
            NotImplementedError -- Not all subclasses supported yet.

        Returns:
            [ndarray] -- Precision matrix of MB.


        Arguments:
            thresh_low {float} -- Sets the Credible Interval to look at.  (default: {0.2})  
        """

    @staticmethod
    def threshold_matrix_s(W12_hist, thresh_low):
        """Static implementation of :func:`BMB.BMB.threshold_matrix`

        Arguments:
            W12_hist {ndarray} -- drawn samples with shape (NITER, nrows, ncols)
            thresh_low {ndarray} -- [thresh_low, 1-thresh_low] CI

        Returns:
            W12 {ndarray} -- Symmetric matrix, whose indices indicate edges in a dependence graph
        """
        p = W12_hist.shape[1]
        q = W12_hist.shape[2]

        qua = np.array([thresh_low, 0.5, 1-thresh_low])
        W12_hist = W12_hist.reshape(W12_hist.shape[0], p*q)

        CI = mstats.mquantiles(a=W12_hist, prob=qua, axis=0,
                               alphap=1, betap=1)
        CI_sign = (CI[0, :]*CI[2, :]) > 0

        MB_est = np.zeros(p*q)
        w = np.where(CI_sign)
        MB_est[w] = CI[1, w]
        return(MB_est.reshape(p, q))

    def get_adj_matrix(self, thresh_low=0.2):
        """Get Adjacency Matrix corresponding to the dependencies according to the inferred W12 block

        Keyword Arguments:
            thresh_low {float} -- Threshold for determining whether an entry in W12 is zero or not. See @threshold_matrix (default: {0.2})
        """
        p = self.p
        q = self.q

        W12 = self.threshold_matrix(thresh_low=thresh_low)
        adj = np.zeros((p+q, p+q))
        adj[0:p, p:(p+q)] = np.abs(W12) > 0
        return(adj)

    def get_adj_orig(self):
        """Get Adjacency Matrix of the W12 block according to the true inverse covariance matrix.

        Raises:
            RuntimeError -- If no true inverse covariance matrix was passed to the constructor.

        Returns:
            [ndarray] -- [True Adjacency Matrix of W12]
        """
        if(self.TRUE_INVCOV is None):
            raise RuntimeError(
                "No true inverse covariance available.")
        p = self.p
        q = self.q

        adj_orig = np.zeros((p+q, p+q))
        adj_orig[0:p, p:(p+q)] = np.abs(self.TRUE_INVCOV[0:p, p:(p+q)]) > 0
        return adj_orig


class BMB_gibbs(BMB):
    """Bayesian Markov Blanket estimation using a Gibbs sampler.
    """

    def threshold_matrix(self, thresh_low):
        """Returns median of the markov blanket, where 0 is not in the (1-thresh_low) credible interval.
        Excludes values from the burnin.

        Arguments:
            thresh_low {float} -- Look at [thresh_low, 1-thresh_low] quantiles, i.e. the 1-thresh_low credible interval.
        """
        p = self.p
        q = self.q
        W12_hist = self.W_list_acc[:, 0:p, p:(p+q)]
        return(BMB.threshold_matrix_s(W12_hist, thresh_low))


class BMB_SA(BMB):
    """Bayesian Markov Blanket estimation using Simulated Annealing on the of the Gibbs sampler for MAP estimation.
    """

    def __init__(self, X, p, q,
                 lambda_, burnin, NSCAN,
                 NITER_SA, NITER_cd, a=None, T0=1, Tn=0.01,
                 DO_COPULA=True, plugin_threshold=40,
                 TRUE_INVCOV=None, w12_start_from_median=False, gig_seed=4321, use_w=False):
        """[summary]

        Arguments:
            X {ndarray} -- Data Matrix
            p {int} -- number of query variables
            q {int} -- number of non-query variables
            lambda_ {int} -- Sparsity Hyperparameter
            burnin {int} -- Number of Iterations that can be discarded as burnin.
            NSCAN {int} -- Number of Iterations of the Gibbs Sampling before the Annealing
            NITER_SA {int} -- Number of Iterations of the Simulated Annealing
            NITER_cd {int} -- Number of Samples drawn with final temperature

        Keyword Arguments:
            a {float} -- Cooling parameter, depends on cooling method. Is chosen automatically if `None` (default: {None})
            T0 {int} -- Initial Temperature (default: {1})
            Tn {float} -- Final temperature (default: {0.01})
            DO_COPULA {bool} -- Whether to wrap the copula around the BMB for data with non-normal marginals or NAs (default: {True})
            plugin_threshold {int} -- If there are less unique values than defined here in a variable, update the Z scores in the copula  (default: {40})
            TRUE_INVCOV {ndarray} -- True inverse covariance that can be optionally provided for artificial data (default: {None})
            w12_start_from_median {bool} -- Whether to start the SA from the median of the Gibbs samples before (default: {False})
            gig_seed {int} -- Seed for the GIG Random Generator (default: {4321})
        """

        BMB.__init__(self, X=X, p=p, q=q,
                     lambda_=lambda_,
                     burnin=burnin,  NSCAN=NSCAN,
                     DO_COPULA=DO_COPULA, plugin_threshold=plugin_threshold,
                     TRUE_INVCOV=TRUE_INVCOV, gig_seed=4321, use_w=use_w)

        self.NITER_SA = NITER_SA
        self.W_list_SA = np.zeros((NITER_SA, self.W.shape[0], self.W.shape[1]))
        self.T0 = T0
        self.Tn = Tn
        self.NITER_cd = NITER_cd
        self.gig_seed = gig_seed
        self.GIG = gig.GIG(gig_seed)
        self.T_cool_list = np.zeros(NITER_SA)
        self.T_inv_list_SA = np.zeros((NITER_SA, p, q))
        self.use_w = use_w

        if(a is None):
            self.a = np.power(Tn/T0, 1/(NITER_SA-NITER_cd))
        else:
            self.a = a

        self.w12_start_from_median = w12_start_from_median

    def run(self):
        """1. Runs the Gibbs sampler for
            - burnin
            - (optional) estimating the sparsity parameters.

        2. Runs the simulated annealing according to settings specified in the constructor.
        """
        super().run()

        p = self.p
        q = self.q
        n = self.n

        lambda_ = self.lambda_
        NITER_SA = self.NITER_SA
        a = self.a
        if(self.w12_start_from_median):
            W12s = self.W_list_acc[:, 0:p, p:(p+q)]
            thresh_low = .2
            qua = np.array([thresh_low, 0.5, 1-thresh_low])
            W12s = W12s.reshape(W12s.shape[0], p*q)

            CI = mstats.mquantiles(a=W12s, prob=qua, axis=0,
                                   alphap=1, betap=1)
            MB_est = self.W
            MB_est[0:p, p:(p+q)] = CI[1, :].reshape(p, q)
            W = MB_est
        else:
            W = self.W

        S = self.S
        T_inv = self.T_inv

        T0 = self.T0

        for m in range(NITER_SA):
            if(m % np.floor_divide(NITER_SA, 4) == 0):
                logging.debug("SA Iteration # %d" % (m))
            if(m < (NITER_SA - self.NITER_cd)):
                T = self.f_cool(m, T0, a)
            ############# T^-1 #############
            T_inv = self._draw_T_inv(w12=W[0:p, p:(p+q)],
                                     p=p, q=q,
                                     lambda_=lambda_, T=T, GIG=self.GIG)

            ############# W11 #############
            if(self.use_w):
                self._draw_W11_Wishart(S=S, W=W,
                                       p=p, q=q, n=n,
                                       lambda_=lambda_,
                                       T=T)
            else:
                W[0:p, 0:p] = self._draw_W11(S=S, W=W,
                                             p=p, q=q, n=n,
                                             lambda_=lambda_,
                                             T=T)

            ############# W12 #############
            W[0:p, p:(p+q)] = self._draw_W12(S=S, W=W,
                                             p=p, q=q,
                                             T_inv=T_inv,
                                             lambda_=lambda_,
                                             T=T)
            ############# W21=W12 #############
            W[p:(p+q), 0:p] = W[0:p, p:(p+q)].T

            self.W_list_SA[m, :, :] = W
            self.T_cool_list[m] = T
            self.T_inv_list_SA[m, :, :] = T_inv
        self.W = W
        return(self)

    def threshold_matrix(self, thresh_low):
        """Returns median of the markov blanket, where 0 is not in the (1-thresh_low) credible interval.
        Uses only last NITER_cd values, where the temperature is cooled down and fixed.

        Arguments:
            thresh_low {float} -- Look at [thresh_low, 1-thresh_low] quantiles, i.e. the 1-thresh_low credible interval.
        """
        p = self.p
        q = self.q
        W12_hist = self.W_list_SA[-self.NITER_cd:, 0:p, p:(p+q)]
        return(BMB.threshold_matrix_s(W12_hist, thresh_low))

    @staticmethod
    def f_cool(m, T0, a):
        """Exponential cooling schedule, T_m=T_0 * a^m.        
        Arguments:
            m {int} -- Index: T_m
            T0 {float} -- Initial Temperature.
            a {float} -- Factor of the cooling schedule, a<0

        Returns:
            [type] -- [description]
        """
        return T0 * pow(a, m)


class BMB_SA_lincool(BMB_SA):
    @staticmethod
    def f_cool(m, T0, a):
        return T0-m*a


class BMB_SA_logmult(BMB_SA):
    @staticmethod
    def f_cool(m, T0, a):
        return T0/(1+a*np.log(1+m))


class BMB_SA_linmult(BMB_SA):
    @staticmethod
    def f_cool(m, T0, a):
        return T0/(1+a*m)

class BMB_SA_qmult(BMB_SA):
    @staticmethod
    def f_cool(m, T0, a):
        return T0/(1+a*(m**2))


def main():
    raise NotImplementedError(
        "This script is not intended to be run directly.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
