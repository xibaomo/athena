import time

import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
import math
import sys, os
import cupy as cp
import pdb
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gammaln
import warnings

# This turns all warnings into hard errors
# warnings.filterwarnings('error')
np.seterr(over='raise')


def eval_stability(rtns, spacing):
    '''
    estimate statbility of return series by computing averaging difference of CDFs
    '''
    res = len(rtns) % spacing
    rtns = rtns[res:]
    n_intervals = len(rtns) // spacing
    print(f"stability check. spacing: {spacing}")
    ds = []
    for i in range(n_intervals - 1):
        r1 = rtns[i * spacing:(i + 1) * spacing]
        r2 = rtns[(i + 1) * spacing:(i + 2) * spacing]
        d, p = stats.ks_2samp(r1, r2)
        ds.append(d)

    return np.mean(ds)


def get_cdf_value(data, x_target):
    """
    Computes the CDF of 'data' and interpolates to find the 'y' for a given 'x_target'.
    """
    # 1. Prepare the Empirical CDF
    sorted_data = np.sort(data)
    # y-coordinates: cumulative probabilities from 1/n to 1.0
    y_coords = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # 2. Interpolate
    # np.interp(x_to_find, x_data, y_data)
    y_target = np.interp(x_target, sorted_data, y_coords)

    return y_target


class ECDFCal(object):  # empirical cdf calculator
    def __init__(self, rtn0):
        self.ecdf = sm.distributions.ECDF(rtn0)

    def compCDF(self, x):
        return self.ecdf(x)

    def compRangeProb(self, lb, ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly


class WeightedCDFCal(object):  # cdf calculator: weighted sum of recent days
    def __init__(self, rtn0, wts, spacing):
        self.wts = wts
        self.ecdfs = []
        # breakpoint()
        for i in range(len(wts)):
            eid = -spacing * (len(wts) - i - 1)
            if eid == 0:
                eid = -1
            cdf = sm.distributions.ECDF(rtn0[-spacing * (len(wts) - i):eid])
            self.ecdfs.append(cdf)

    def compCDF(self, x):
        weighted_cdf = [c(x) * w for c, w in zip(self.ecdfs, self.wts)]
        return np.sum(weighted_cdf)

    def compRangeProb(self, lb, ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly


class GaussCDFCal(object):
    def __init__(self, rtn0):
        self.mean = np.mean(rtn0)
        self.std = np.std(rtn0)

    def compCDF(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.std)

    def compRangeProb(self, lb, ub):
        return self.compCDF(ub) - self.compCDF(lb)


class LaplaceCDFCal(object):
    def __init__(self, rtn0):
        self.mean = np.mean(rtn0)
        self.scale = np.std(rtn0) / math.sqrt(2.)

    def compCDF(self, x):
        if x <= self.mean:
            return .5 * np.exp((x - self.mean) / self.scale)
        return 1 - .5 * np.exp((self.mean - x) / self.scale)

    def compRangeProb(self, lb, ub):
        return self.compCDF(ub) - self.compCDF(lb)


class MkvRegularCal(object):
    def __init__(self, nstates, cdf_cal):
        self.n_states = nstates
        self.transProbCal = cdf_cal

    def buildTransMat(self, lb_rtn, ub_rtn):
        # rtns = rtns[~np.isnan(rtns)]
        # self.transProbCal = ECDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / (npts - 1)
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p
        P = np.zeros((npts, npts))

        for i in range(npts):
            for j in range(npts):
                P[i, j] = idxDiff2Prob[j - i]

        # normalization
        for i in range(npts):
            s = 0.
            for j in range(npts):
                s += P[i, j]
            if s == 0:
                pdb.set_trace()
            P[i, :] = P[i, :] / s
        self.transMat = P.copy()
        return P

    def compMultiStepProb(self, steps, rtns, lb_rtn, ub_rtn):
        P = self.buildTransMat(rtns, lb_rtn, ub_rtn)
        drtn = (ub_rtn - lb_rtn) / (self.n_states - 1)
        PWP = np.linalg.matrix_power(P, steps)
        idx = int((0 - lb_rtn) / drtn)
        if idx < 0:
            pdb.set_trace()
        return PWP[idx, :]


class MkvAbsorbCal(object):
    def __init__(self, nstates, cdf_cal):
        self.n_states = nstates
        # self.cdfType = cdf
        self.transProbCal = cdf_cal
        # self.cdf_wts=cdf_wts

    # def createCDFCal(self,rtns):
    #     if self.cdfType == 'emp':
    #         self.transProbCal = ECDFCal(rtns)
    #     elif self.cdfType == 'gauss':
    #         self.transProbCal = GaussCDFCal(rtns)
    #     elif self.cdfType == "laplace":
    #         self.transProbCal = LaplaceCDFCal(rtns)
    #     elif self.cdfType == "wts":
    #         self.transProbCal = WeightedCDFCal(rtns,self.cdf_wts)
    #     else:
    #         print("Wrong cdf type: ", self.cdfType)
    #         sys.exit(1)
    def buildTransMat(self, lb_rtn, ub_rtn):
        # rtns = rtns[~np.isnan(rtns)]
        # self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p
        P = np.zeros((npts + 2, npts + 2))

        for i in range(npts):
            for j in range(npts):
                P[i, j] = idxDiff2Prob[j - i]
        for i in range(npts):
            P[i, npts] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, 1.)
            P[i, npts + 1] = self.transProbCal.compRangeProb(-1., (-1 - i) * d + d / 2)
        P[npts, npts] = 1.
        P[npts + 1, npts + 1] = 1.

        # normalization
        for i in range(npts + 2):
            s = 0.
            for j in range(npts + 2):
                s += P[i, j]
            if s == 0:
                pdb.set_trace()
            P[i, :] = P[i, :] / s
        self.transMat = P.copy()
        return P

    def comp1stHitProb(self, steps, start_id, tar_id1, tar_id2):
        P1 = self.transMat.copy()
        P2 = self.transMat.copy()
        f1 = P1[:, tar_id1].copy()
        f2 = P2[:, tar_id2].copy()
        P1[:, tar_id1] = 0
        P2[:, tar_id2] = 0

        mid = start_id
        pb1 = f1[mid]
        pb2 = f2[mid]
        # pdb.set_trace()
        for i in range(steps - 1):
            f1 = P1 @ f1
            f2 = P2 @ f2
            pb1 += f1[mid]
            pb2 += f2[mid]
        return pb1, pb2

    def compWinProb(self, rtns, lb_rtn, ub_rtn):
        rtns = rtns[~np.isnan(rtns)]
        self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p

        C = np.zeros((npts, npts))
        I = np.identity(npts)
        Q = np.zeros((npts, 1))
        one = np.ones((npts, 1))
        # Qsell = np.zeros((npts,1))

        for i in range(npts):
            for j in range(npts):
                C[i, j] = idxDiff2Prob[j - i]
            Q[i] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, .5)

        tmp = I - C
        tic = time.time()
        # pdb.set_trace()
        try:
            tmp = cp.array(tmp)
            tmp = cp.linalg.inv(tmp)
            tmp = cp.asnumpy(tmp)
        except:
            pdb.set_trace()
            print("inversion fails")
            return 0.5, 1e10

        pr = np.matmul(tmp, Q)
        # ps = np.matmul(tmp,Qsell)
        steps = np.matmul(tmp, one)

        # print("matrix ops take: ", time.time()-tic)
        idx = int((0 - lb_rtn) / d)
        p = pr[idx][0]
        sp = steps[idx][0]

        if p < 0:
            p = 0.
        return p, sp

    def compUpAveSteps(self, rtns, lb, ub):
        rtns = rtns[~np.isnan(rtns)]
        self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub - lb) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p

        C = np.zeros((npts, npts))
        I = np.identity(npts)
        Q = np.zeros((npts, 1))
        one = np.ones((npts, 1))
        # Qsell = np.zeros((npts,1))

        for i in range(npts):
            for j in range(npts):
                C[i, j] = idxDiff2Prob[j - i]
            Q[i] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, .5)
        # normalize C and Q
        for i in range(npts):
            s = np.sum(C[i, :]) + Q[i]
            if s == 0:
                pdb.set_trace()
            C[i, :] = C[i, :] / s
            Q[i] = Q[i] / s

        tmp = I - C
        try:
            tmp = np.linalg.inv(tmp)
        except:
            print("inversion fails")
            return 1e10

        steps = np.matmul(tmp, one)

        idx = int((0 - lb) / d)
        sp = steps[idx][0]

        return sp


def dp_minimize(candidates, cal_f, max_n_choose=-1, min_n_choose=10, result_rank=0):
    '''
    candidates: [(sym1,params1),(sym2,params2),...]
    n_choose: number of selections
    cal_f: function to compute cost
    '''

    n = len(candidates)
    k = max_n_choose
    if k < 0:
        k = n
    dp = [[1e20 for _ in range(k + 1)] for _ in range(n + 1)]
    chosen = [[[] for _ in range(k + 1)] for _ in range(n + 1)]

    # build the table bottom-up
    for i in range(1, n + 1):
        # print("checking {}th sym out of {}".format(i, n))
        for j in range(1, k + 1):
            if j <= i:
                args = chosen[i - 1][j - 1] + [candidates[i - 1]]
                cost, _ = cal_f(args)
                if cost < dp[i - 1][j]:
                    dp[i][j] = cost
                    chosen[i][j] = args
                else:
                    dp[i][j] = dp[i - 1][j]
                    chosen[i][j] = chosen[i - 1][j]

    # min_idx = dp[n].index(min(dp[n]))
    # return dp[n][min_idx],chosen[n][min_idx]
    arr = np.array(dp[n][min_n_choose:])
    sorted_id = np.argsort(arr)
    idx = sorted_id[result_rank]
    minval = arr[idx]
    choice = chosen[n][idx + min_n_choose]
    return minval, choice


def __compProb1stHidBounds(rtns, steps, ub_rtn=0.05, lb_rtn=-0.05):
    d = 0.001 / 8
    ns = int((ub_rtn - lb_rtn) / d)
    mkvcal = MkvAbsorbCal(ns)
    # rtns = df['Close'].pct_change().values[-lookback:]
    # pdb.set_trace()
    # print(f"{len(rtns)}-tradehour volatility: {np.std(rtns):.5f}")
    mkvcal.buildTransMat(rtns, lb_rtn, ub_rtn)
    # d = (ub_rtn-lb_rtn)/ns
    mid = int((0 - lb_rtn) / d)
    pbu, pbd = mkvcal.comp1stHitProb(steps, mid, -2, -1)
    return pbu, pbd


def compProb1stHitBounds(steps, cdf_cal, ub_rtn=.5, lb_rtn=-.5):
    d = 0.001 / 4
    ns = int((ub_rtn - lb_rtn) / d)
    # breakpoint()
    mkvcal = MkvAbsorbCal(ns, cdf_cal)
    mkvcal.buildTransMat(lb_rtn, ub_rtn)
    mid = int((0 - lb_rtn) / d)
    pbu, pbd = mkvcal.comp1stHitProb(steps, mid, -2, -1)
    return pbu, pbd


def compMultiStepProb(steps, lb_rtn, ub_rtn, cdf_cal, drtn=0.001 / 4):
    ns = int((ub_rtn - lb_rtn) / drtn)
    mkvcal = MkvRegularCal(ns, cdf_cal)
    P = mkvcal.buildTransMat(lb_rtn, ub_rtn)

    # pdb.set_trace()
    PWP = np.linalg.matrix_power(P, steps)
    idx = int((0 - lb_rtn) / drtn)
    if idx < 0:
        pdb.set_trace()
    return PWP[idx, :]


def compute_steady_dist(rtns, lb_rtn, ub_rtn):
    drtn = 0.001 / 4
    ns = int((ub_rtn - lb_rtn) / drtn)
    mkvcal = MkvRegularCal(ns)
    P = mkvcal.buildTransMat(rtns, lb_rtn, ub_rtn)

    I = np.eye(ns)
    ONE = np.ones((ns, ns))
    pi = np.ones((1, ns))
    tmp = np.eye(ns) - P + ONE
    tmp = np.linalg.inv(tmp)
    pi = pi @ tmp
    return pi.ravel()


def compute_residuals(params, r, x):
    omega, beta, gamma, xi, phi, tau, sigma_u, nu = params
    T = len(r)
    logh = np.zeros(T)
    # logh[0] = np.log(np.var(r))
    logh[0] = np.log(x[0])
    z = np.zeros(T)
    z[0] = r[0] / x[0]

    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(x[t - 1])
        h_t = np.exp(logh[t])

        if h_t < 1e-30:
            h_t = 1e-10
            # breakpoint()
        z[t] = r[t] / np.sqrt(h_t)

    return z


def realized_garch_loglik(params, r, x):
    omega, beta, gamma, xi, phi, tau, sigma_u, nu = params

    T = len(r)
    logh = np.zeros(T)

    if len(x)==0:
        breakpoint()
    logh[0] = np.log(x[0])
    ll = 0.0

    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(x[t - 1])
        logh_clipped = np.clip(logh[t], -20, 20)
        h_t = np.exp(logh_clipped)

        if h_t < 1e-30:
            h_t = 1e-10
            # breakpoint()
        z = r[t] / np.sqrt(h_t)

        # return density
        # breakpoint()
        # normal distribution
        # ll += -0.5 * (np.log(2 * np.pi) + np.log(h_t) + z ** 2)

        # t-distribution
        term1 = gammaln((nu + 1) / 2) - gammaln(nu / 2)
        term2 = -0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h_t)
        term3 = -((nu + 1) / 2) * np.log(1 + (z ** 2) / (nu - 2))

        ll += term1 + term2 + term3
        # measurement density
        # try:
        meas = np.log(x[t]) - (xi + phi * logh[t] + tau * z)
        ll += -0.5 * (np.log(2 * np.pi * sigma_u ** 2) + meas ** 2 / sigma_u ** 2)
        # except:
        #     breakpoint()

    return -ll


def forecast_realized_garch(logh_T, x_T, params, steps):
    omega, beta, gamma, xi, phi, tau, sigma_u, nu = params

    forecasts = []
    logh_next = omega + beta * logh_T + gamma * np.log(x_T)

    for _ in range(steps):
        # expected log realized variance
        logx_next = xi + phi * logh_next

        logh_next = omega + beta * logh_next + gamma * logx_next
        forecasts.append(np.exp(logh_next))

    return np.sqrt(np.array(forecasts))


def compute_total_return_distribution(rtns, bars_per_day, lookback_days, fwd_days, vol_scaler=0.8):
    daily_matrix = rtns[-lookback_days * bars_per_day:].reshape(lookback_days, bars_per_day)
    daily_returns = np.sum(daily_matrix, axis=1)
    daily_rv = np.sum(daily_matrix ** 2, axis=1)
    # breakpoint()
    daily_vol = np.sqrt(daily_rv)
    mu = np.mean(daily_returns)
    residuals = (daily_returns - mu) / daily_vol

    if len(daily_returns) > 2000:
        s1 = eval_stability(daily_returns, 100)
        print(f"ave cdf diff of daily returns: {s1:.4f}")
        s2 = eval_stability(residuals, 100)
        print(f"ave cdf diff of daily residules: {s2:.4f} ")

    init = [-0.1, 0.9, 0.05, 0.1, 0.8, 0.0, 0.2, 5]

    bounds = [
        (-5, 5),  # omega
        (0.1, 0.999),  # beta
        (0.0, 1.0),  # gamma
        (-5, 5),  # xi
        (0.1, 5),  # phi
        (-5, 5),  # tau
        (1e-6, 5),  # sigma_u
        (2.01, 50)  # nu
    ]

    res = minimize(realized_garch_loglik, init,
                   args=(daily_returns, daily_rv),
                   method="L-BFGS-B",
                   bounds=bounds)

    params_hat = res.x
    # print("Estimated parameters:")
    # print(params_hat)
    if len(daily_returns) > 2000:
        z = compute_residuals(params_hat, daily_returns, daily_rv)
        print(f"optimized ave cdf diff of daily residules: {eval_stability(z, 100):.4f}")

    # ======================================================
    # 4️⃣ Filter conditional variance
    # ======================================================

    omega, beta, gamma, xi, phi, tau, sigma_u, nu = params_hat

    T = len(daily_returns)
    logh_filt = np.zeros(T)
    logh_filt[0] = np.log(np.var(daily_returns))

    for t in range(1, T):
        logh_filt[t] = omega + beta * logh_filt[t - 1] + gamma * np.log(daily_rv[t - 1])

    # h_filt = np.exp(logh_filt)

    future_vol = forecast_realized_garch(
        logh_filt[-1],
        daily_rv[-1],
        params_hat,
        steps=fwd_days
    )

    # Monte-Carlo sim
    future_vol = future_vol*vol_scaler
    # print(f"future volatility: ", future_vol)
    n_sim = 100000
    tot_rtn = np.zeros(n_sim)
    for k in range(n_sim):
        random_res = np.random.choice(residuals, size=fwd_days)
        tot_rtn[k] = np.sum(random_res * future_vol) + mu * fwd_days
    # breakpoint()
    return tot_rtn
