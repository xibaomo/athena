import sys,os
import math
import numpy as np
folder = os.environ['ATHENA_HOME'] + "/py_algos/py_basics"
sys.path.append(folder)
from ga_min import *
import functools
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta,date
import yfinance as yf

class KalmanTracker(object):
    def __init__(self,gain=0.4, dt=0.2, max_dim=3):
        self.kalman_gain = gain
        self.dt = dt
        self.max_dim = max_dim

    def estimate_dims(self,z):
        pv_max = 0
        opt_dim = -1
        for i in range(2, self.max_dim + 1):
            arr = np.diff(z, i) / (self.dt ** i)
            s, pv = stats.wilcoxon(arr)

            if pv > pv_max:
                opt_dim = i
                pv_max = pv
        arr = np.diff(z, opt_dim) / (self.dt ** opt_dim)
        print(f"optimal diff order: {opt_dim}, abs(mean): {(np.mean(arr)):.4e}, pv_max of 0-mean: {pv_max:.4e}")
        R = np.var(arr)
        print(f"R: {R:.4e}")
        return opt_dim, R

    def kalmanNdMotion(self,z, R, q, dim):
        N = len(z)
        Rm = np.array([[R]])
        states = np.zeros((N, dim + 1))
        T = self.dt
        X = np.zeros((dim, 1))
        X[0, 0] = z[0]

        P = np.eye(dim) * R
        # pdb.set_trace()
        F = np.zeros((dim, dim))
        for i in range(dim - 1):
            F[i, i + 1] = 1.
        Phi = np.eye(dim)

        # fill matrix Phi
        for i in range(1, dim):
            f = 1. / math.factorial(i)
            Phi = Phi + f * np.linalg.matrix_power(F, i) * (T ** i)

        # fill column vector G
        G = np.zeros((dim, 1))
        for i in range(dim):
            f = 1. / math.factorial(dim - i)
            G[i, 0] = f * (T ** (dim - i))

        # q = R/10
        Q = G @ G.T * q / T
        H = np.zeros((1, dim))
        H[0, 0] = 1.
        # pdb.set_trace()
        innovation = np.zeros(N)
        for i in range(1, N):
            X_ = np.matmul(Phi, X)
            P_ = np.matmul(np.matmul(Phi, P), Phi.transpose()) + Q
            tmp = np.matmul(np.matmul(H, P_), H.transpose()) + Rm

            K = np.matmul(np.matmul(P_, H.transpose()), np.linalg.inv(tmp))
            innovation[i] = z[i] - np.matmul(H, X_).flatten()[0]
            # pdb.set_trace()
            X = X_ + K * innovation[i]
            # Q = np.maximum(0, (innovation**2 - R))
            states[i, :-1] = X.flatten()
            states[i, -1] = K[0][0]

            tmp = np.eye(dim) - np.matmul(K, H)

            P = np.matmul(np.matmul(tmp, P_), tmp.transpose()) + np.matmul(np.matmul(K, Rm), K.transpose())

        # pdb.set_trace()
        return states, innovation

    def calibrateKalmanArgs(self,Z, opt_method, kalman_dim, R):
        def obj_func(x, params):
            Z, N, kalman_dim, R = params
            q, = x
            xs, inno = self.kalmanNdMotion(Z, R, q=q, dim=kalman_dim)
            cost = (xs[-1, -1] - self.kalman_gain)**2

            return cost,

        init_x = np.array([R / 2])
        bounds = [(1e-8, 1e-0)]
        # bounds = None
        result = None
        N = 50
        if opt_method == 0:
            result = minimize(obj_func, init_x, args=((Z, N, kalman_dim, R),), bounds=bounds, method='COBYLA',
                              tol=1e-5)
        elif opt_method == 1:
            tmp_func = functools.partial(obj_func, params=(Z, N, kalman_dim, R))
            result = ga_minimize(tmp_func, len(init_x), bounds, population_size=1000, num_generations=100)
            # print("ga result: ", result.x, result.fun)
            # result = minimize(obj_func,result.x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
        else:
            print("ERROR! opt_method is not supported")

        print("optimal q: ", result.x)
        print("optimal cost: ", result.fun)
        return result.x

    def estimateMotion(self,z,lookback=5):
        kdim,R = self.estimate_dims(z)
        Rz = 1./math.factorial(kdim)*(self.dt**kdim)
        Rz = Rz**2*R

        pm = self.calibrateKalmanArgs(z,opt_method=0,kalman_dim=kdim,R=Rz)

        xs,inno = self.kalmanNdMotion(z,R=Rz,q=pm[0],dim=kdim)

        vs = xs[:,1]
        x = [x for x in range(lookback)]
        p = np.polyfit(x,vs[-lookback:],1)
        return vs,p[0]
def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ticker>")
        sys.exit(1)

    sym = sys.argv[1]
    syms = [sym]
    target_date = datetime.today().strftime('%Y-%m-%d')
    back_days = 180
    start_date = add_days_to_date(target_date, -back_days)
    # data = yf.download(syms, start=start_date, end=target_date)
    data = yf.Ticker(sym).history(start=start_date, end=target_date, interval='1d')

    z = np.log(data['Close'].values)

    kt = KalmanTracker()
    vs,acc = kt.estimateMotion(z)
    print(f"v: {vs[-1]}, acc: {acc}")