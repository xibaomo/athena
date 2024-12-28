import math
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,date
from scipy.stats import spearmanr
from scipy.optimize import minimize
import yfinance as yf
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf
import functools
import pdb
from statsmodels.tsa.stattools import pacf
folder = os.environ['ATHENA_HOME'] + "/py_algos/py_basics"
sys.path.append(folder)
from ga_min import *
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from statsmodels.tsa.arima.model import ARIMA

def compute_smoothness(arr):
    v = arr - np.min(arr)
    v = v/np.max(v)
    return np.std(np.diff(v,2))
def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str

def fft_derivative(y, dx=1):
    """
    Computes the derivative of a 1D array using FFT.

    Parameters:
        y (np.array): The input 1D array (function values).
        dx (float): The spacing between points in the input array (default is 1.0).

    Returns:
        np.array: The derivative of the input array.
    """
    n = len(y)  # Number of points in the input array
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi  # Frequency array (angular frequencies)

    # Perform FFT of the input array
    y_fft = np.fft.fft(y)

    # Multiply by ik to compute the derivative in Fourier space
    y_derivative_fft = 1j * k * y_fft

    # Perform the inverse FFT to get the derivative in the spatial domain
    y_derivative = np.fft.ifft(y_derivative_fft)

    # Return the real part of the derivative (imaginary part should be negligible)
    return np.real(y_derivative)


def estimate_kalman_dim(z,max_dim=3, dt=0.2):
    pv_max = 0
    opt_dim = -1
    for i in range(2,max_dim+1):
        arr = np.diff(z,i)/(dt**i)
        s,pv = stats.wilcoxon(arr)

        if pv > pv_max:
            opt_dim = i
            pv_max = pv
    arr = np.diff(z,opt_dim)/(dt**opt_dim)
    print(f"optimal diff order: {opt_dim}, abs(mean): {(np.mean(arr)):.4e}, pv_max of 0-mean: {pv_max:.4e}")
    R = np.var(arr)
    print(f"R: {R:.4e}")
    return opt_dim, R

def adaptive_kalman_filter(z, F, H, Q, R_init, P0, x0, N):
    """
    Adaptive Kalman Filter with innovation-based estimation of R.

    Args:
        z: Measurement vector (1D array of length T).
        F: State transition matrix.
        H: Measurement matrix.
        Q: Process noise covariance matrix.
        R_init: Initial measurement noise covariance (scalar).
        P0: Initial state covariance matrix.
        x0: Initial state estimate (1D array).
        N: Window size for innovation variance estimation.

    Returns:
        x_est: Estimated states over time (2D array of shape (T, state_dim)).
        P_est: State covariance estimates over time (3D array of shape (T, state_dim, state_dim)).
        R_est: Estimated measurement noise covariance over time (1D array of length T).
    """
    # Number of measurements
    T = len(z)

    # State dimension
    state_dim = F.shape[0]

    # Initialize state and covariance
    x = x0.reshape(-1, 1)  # Convert to column vector (2D array of shape (state_dim, 1))
    P = P0
    R = R_init

    # Store results
    x_est = np.zeros((T, state_dim+1))
    P_est = np.zeros((T, state_dim, state_dim))
    R_est = np.zeros(T)

    # Store recent innovations for estimating R
    innovations = np.zeros(N)

    # Adaptive Kalman filtering loop
    for k in range(T):
        # Prediction step
        x_pred = F @ x  # (state_dim, 1)
        P_pred = F @ P @ F.T + Q

        # Measurement update
        y = z[k] - (H @ x_pred).item()  # Innovation (residual, scalar)
        S = H @ P_pred @ H.T + R  # Innovation covariance (scalar)
        K = P_pred @ H.T / S  # Kalman gain (state_dim, 1)
        x = x_pred + K * y  # Updated state estimate (state_dim, 1)
        P = (np.eye(state_dim) - K @ H) @ P_pred  # Updated state covariance (state_dim, state_dim)

        # Save state estimates
        # pdb.set_trace()
        x_est[k, :state_dim] = x.flatten()  # Store as row in result (convert back to 1D array)
        # pdb.set_trace()
        x_est[k,state_dim] = K[0][0]
        P_est[k, :, :] = P

        # Store the innovation for estimating R
        innovations[k % N] = y ** 2

        # Update measurement noise covariance R based on innovation variance
        if k >= N:
            # pdb.set_trace()
            R = np.mean(innovations)  # Estimate R as the mean innovation variance
            R = R - (H@P_pred@H.T).flatten()[0]
            R = max(R,1.e-6)

        # Store estimated R
        R_est[k] = R

    return x_est, P_est, R_est

def kalman2dmotion_adaptive(Z,q,dt=0.01):
    T = dt
    F = np.array([[1., T], [ 0., 1.]])
    G = np.array([[T ** 2 / 2.], [1.]])
    Q = G @ G.T * q/T
    H = np.array([[1., 0.]])
    R_init = 1.
    P0 = np.eye(2)*R_init
    x0 = np.array([Z[0],0.])

    states,P_est,R_est = adaptive_kalman_filter(Z,F,H,Q,R_init,P0,x0,N=20)

    return states,P_est
def kalman2dmotion(Z,R,q,dt):
    dim = 2
    N = len(Z)
    states = np.zeros((N,dim+1))
    T = dt
    F = np.array([[1., T], [0., 1.]])
    G = np.array([[T ** 2 / 2.], [T]])
    Q = G @ G.T * q / T
    H = np.array([[1., 0.]])
    P = np.eye(2)*10
    X = np.array([Z[0],1.])
    Rm = np.array([[R]])

    for i in range(N):
        if i==0:
            continue
        # pdb.set_trace()
        X_ = F @ X
        P_ = F @ P @ F.T + Q
        tmp = H @ P_ @ H.T + Rm
        # pdb.set_trace()
        # K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(tmp))
        K = P_ @ H.T @ np.linalg.inv(tmp)
        tmp = Z[i] - H @ X_
        X = X_ + K @ tmp
        states[i,:-1] = X

        states[i,-1] = K[0][0]

        tmp = np.eye(dim) - K@H
        # pdb.set_trace()
        # P = np.matmul(np.matmul(tmp,P_),tmp.transpose()) + np.matmul(np.matmul(K,Rm),K.transpose())
        P = tmp@P_@tmp.T + K@Rm@K.T

    return states,P
def adaptive_kalman_motion(Z,q,dt):
    T = dt
    F = np.array([[1., T, T ** 2 / 2], [0., 1., T], [0., 0., 1.]])
    G = np.array([[T ** 3 / 6], [T ** 2 / 2], [1.]])
    Q = G @ G.T * q/dt
    H = np.array([[1., 0., 0.]])
    R_init = 1
    P0 = np.eye(3)*R_init
    x0 = np.array([Z[0],0,0])

    states,P_est,R_est = adaptive_kalman_filter(Z,F,H,Q,R_init,P0,x0,N=10)

    return states,P_est
def kalman3dmotion(Z,R,q,dt):
    dim = 3
    N = len(Z)
    Rm = np.array([[R]])
    states = np.zeros((N,dim+1))
    T=dt
    X = np.zeros(dim)
    X[0] = Z[0]
    P = np.eye(dim)*R
    # pdb.set_trace()
    F = np.array([[1., T, T**2/2],[0., 1., T],[0., 0., 1.]])
    G = np.array([[T**3/6],[T**2/2],[T]])
    Q = q/dt
    H = np.array([[1.,0.,0.]])
    for i in range(N):
        if i==0:
            continue
        # pdb.set_trace()
        X_ = np.matmul(F,X)
        P_ = np.matmul(np.matmul(F,P),F.transpose()) + np.matmul(G*Q,G.transpose())
        tmp = np.matmul(np.matmul(H,P_),H.transpose()) + Rm
        # pdb.set_trace()
        K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(tmp))
        tmp = Z[i] - np.matmul(H,X_)
        X = X_ + np.matmul(K,tmp)
        states[i,:-1] = X
        # states[i,1] = states[i,0]-states[i-1,0]
        # pdb.set_trace()
        states[i,-1] = K[0][0]

        tmp = np.eye(dim) - np.matmul(K,H)
        # pdb.set_trace()
        P = np.matmul(np.matmul(tmp,P_),tmp.transpose()) + np.matmul(np.matmul(K,Rm),K.transpose())

    # print("FInal P: ",P)
    return states,P
def kalman4dmotion(Z,R,q,dt):
    dim = 4
    N = len(Z)
    Rm = np.array([[R]])
    states = np.zeros((N, dim + 1))
    T = dt
    X = np.zeros(dim)
    X[0] = Z[0]
    P = np.eye(dim) * R
    # pdb.set_trace()
    F = np.array([[1., T, T ** 2 / 2, T**3/6], [0., 1., T, T**2/2], [0., 0., 1., T],[0,0,0,1]])
    # G = np.array([[T ** 5 / 20, T**4/8, T**3/6, T**2/6], [T ** 4/8, T**3/3, T**2/2, T/ 2], [T**3/6, T**2/2, T,1.],[T**2/6, T/2, 1,1]])
    G = np.array([[T**4/24,T**3/6,T**2/2,T]]).T
    QQ = G@G.T* q/T
    H = np.array([[1., 0., 0.,0.]])
    for i in range(N):
        if i == 0:
            continue
        # pdb.set_trace()
        X_ = np.matmul(F, X)
        P_ = np.matmul(np.matmul(F, P), F.transpose()) + QQ
        tmp = np.matmul(np.matmul(H, P_), H.transpose()) + Rm
        # pdb.set_trace()
        K = np.matmul(np.matmul(P_, H.transpose()), np.linalg.inv(tmp))
        tmp = Z[i] - np.matmul(H, X_)
        X = X_ + np.matmul(K, tmp)
        states[i, :-1] = X
        # states[i,1] = states[i,0]-states[i-1,0]
        # pdb.set_trace()
        states[i, -1] = K[0][0]

        tmp = np.eye(dim) - np.matmul(K, H)
        # pdb.set_trace()
        P = np.matmul(np.matmul(tmp, P_), tmp.transpose()) + np.matmul(np.matmul(K, Rm), K.transpose())

    # print("FInal P: ",P)
    return states, P
def kalman5dmotion(Z,R,q,dt):
    dim = 5
    N = len(Z)
    Rm = np.array([[R]])
    states = np.zeros((N, dim + 1))
    T = dt
    X = np.zeros(dim)
    X[0] = Z[0]
    P = np.eye(dim) * R
    # pdb.set_trace()
    F = np.array([[1., T, T ** 2 / 2, T ** 3 / 6, T**4/24],
                  [0., 1., T, T ** 2 / 2, T**3/6],
                  [0., 0., 1., T, T**2/2],
                  [0, 0, 0, 1, T],
                  [0, 0, 0, 0, 1]])
    G = np.array([[T**5/120, T ** 4 / 24, T ** 3 / 6, T ** 2 / 2, T]]).T
    QQ = G @ G.T * q / T
    H = np.array([[1., 0., 0., 0.,0]])
    for i in range(N):
        if i == 0:
            continue
        # pdb.set_trace()
        X_ = np.matmul(F, X)
        P_ = np.matmul(np.matmul(F, P), F.transpose()) + QQ
        tmp = np.matmul(np.matmul(H, P_), H.transpose()) + Rm
        # pdb.set_trace()
        K = np.matmul(np.matmul(P_, H.transpose()), np.linalg.inv(tmp))
        tmp = Z[i] - np.matmul(H, X_)
        X = X_ + np.matmul(K, tmp)
        states[i, :-1] = X
        # states[i,1] = states[i,0]-states[i-1,0]
        # pdb.set_trace()
        states[i, -1] = K[0][0]

        tmp = np.eye(dim) - np.matmul(K, H)
        # pdb.set_trace()
        P = np.matmul(np.matmul(tmp, P_), tmp.transpose()) + np.matmul(np.matmul(K, Rm), K.transpose())

    # print("FInal P: ",P)
    return states, P
def kalmanNdmotion_q(z,R,q, dt,dim=-1):
    N = len(z)
    Rm = np.array([[R]])
    states = np.zeros((N, dim + 1))
    T = dt
    X = np.zeros((dim,1))
    X[0,0] = z[0]

    P = np.eye(dim) * R
    # pdb.set_trace()
    F = np.zeros((dim,dim))
    for i in range(dim-1):
        F[i,i+1] = 1.
    Phi = np.eye(dim)

    # fill matrix Phi
    for i in range(1,dim):
        f = 1./math.factorial(i)
        Phi = Phi + f*np.linalg.matrix_power(F,i) * (T**i)

    # fill column vector G
    G = np.zeros((dim,1))
    for i in range(dim):
        f = 1./math.factorial(dim-i)
        G[i,0] = f * (T**(dim-i))

    # q = R/10
    Q = G @ G.T * q / T
    H = np.zeros((1,dim))
    H[0,0]=1.
    # pdb.set_trace()
    innovation = np.zeros(N)
    for i in range(1,N):
        X_ = np.matmul(Phi, X)
        P_ = np.matmul(np.matmul(Phi, P), Phi.transpose()) + Q
        tmp = np.matmul(np.matmul(H, P_), H.transpose()) + Rm

        K = np.matmul(np.matmul(P_, H.transpose()), np.linalg.inv(tmp))
        innovation[i] = z[i] - np.matmul(H, X_).flatten()[0]
        # pdb.set_trace()
        X = X_ + K*innovation[i]
        # Q = np.maximum(0, (innovation**2 - R))
        states[i, :-1] = X.flatten()
        states[i, -1] = K[0][0]

        tmp = np.eye(dim) - np.matmul(K, H)

        P = np.matmul(np.matmul(tmp, P_), tmp.transpose()) + np.matmul(np.matmul(K, Rm), K.transpose())

    # pdb.set_trace()
    return states, innovation

def cal_profit(xs,prices,N,cap = 10000):
    # pdb.set_trace()
    v = xs[-N:,1]
    price = prices[-N:]
    transactions=[]
    is_pos_on = False
    p0=0
    c0 = cap
    # pdb.set_trace()
    for i in range(1,N):
        # if  s[i] > s[i-1] and not is_pos_on and abs(xs[i,-1]-k_eq) < 0.02:
        if v[i] > 0 and v[i] > v[i-1] and not is_pos_on:
            p0 = price[i]
            is_pos_on = True
            trans = [i,-1,-1]
            transactions.append(trans)
        if not is_pos_on:
            continue
        if  v[i] < 0:
            is_pos_on = False
            cap = cap / p0 * price[i]
            transactions[-1][1] = i
            transactions[-1][2] = price[i]/p0 - 1
    if is_pos_on:
        is_pos_on = False
        cap = cap / p0 * price[-1]
        transactions[-1][1] = N-1
        transactions[-1][2] = price[-1]/p0-1

    if len(transactions)> 0:
        print(transactions)
        print(f"Protift -- total: ${cap-c0:.2f}, ave: ${(cap-c0)/len(transactions):.2f}")
    else:
        print(f"Total profit: $0.00")

KALMAN_FUNC = kalmanNdmotion_q
# KALMAN_FUNC = kalmanNdmotion_ar1
# KALMAN_FUNC = kalman2dmotion_arima201
# KALMAN_FUNC = kalman2dmotion_ar2
def obj_func(x,params):
    Z,N,kalman_dim,R,dt = params
    q, = x
    xs, inno = KALMAN_FUNC(Z, R, q=q, dt=dt,dim=kalman_dim)
    # cost = abs(np.var(inno[-N:]))
    cost = abs(xs[-1,-1]-0.4)
    # nu = Z[-N:] - xs[-N:,0]

    # s,pval = stats.normaltest(nu)

    # var = np.var(inno[-N:])
    # cost =  abs(var/R-1)  #- pval + P[1,1]
    return cost,

def calibrate_kalman_args(Z,N=50,opt_method=0, kalman_dim=-1, R=0, dt = 0.1):
    init_x = np.array([R/2])
    bounds = [(1e-8,1e-0)]
    # bounds = None
    result = None
    # pdb.set_trace()
    if opt_method == 0:
        result = minimize(obj_func,init_x,args=((Z,N,kalman_dim,R,dt),),bounds=bounds,method='COBYLA',tol=1e-5)
    elif opt_method == 1:
        tmp_func = functools.partial(obj_func,params=(Z,N,kalman_dim,R,dt))
        result = ga_minimize(tmp_func,len(init_x),bounds,population_size=1000,num_generations=100)
        # print("ga result: ", result.x, result.fun)
        # result = minimize(obj_func,result.x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
    else:
        print("ERROR! opt_method is not supported")

    print("optimal q: ", result.x)
    print("optimal cost: ", result.fun)
    return result.x
def expand_price(data):
    p = np.zeros(len(data)*2)
    hi = data['High'].values
    lw = data['Low'].values
    cls = data['Close'].values
    k=0
    for i in range(len(data)):
        mid = (hi[i]+lw[i])*.5
        p[k] = mid
        k+=1
        p[k] = cls[i]
        k+=1
    return p
def test_stock(sym,target_date=None):
    syms = [sym]
    # target_date = '2024-09-5'
    if target_date is None:
        target_date = datetime.today().strftime('%Y-%m-%d')
    back_days = 180
    start_date = add_days_to_date(target_date,-back_days)
    # data = yf.download(syms, start=start_date, end=target_date)
    data = yf.Ticker(sym).history(start=start_date, end=target_date, interval='1d')
    # pdb.set_trace()
    df = data['Close']
    # prices = expand_price(data)
    prices = data['Close'].values
    print("length of history: ", len(prices))
    z = np.log(prices) #/ df.values[0]
    # pdb.set_trace()
    T = .2
    kdim,R = estimate_kalman_dim(z,dt=T)
    print("Optimal kalman order: ", kdim)
    rtns = np.diff(z)
    result = acorr_ljungbox(rtns, lags=[10], return_df=True)
    print("is rtns white noise: ", result['lb_pvalue'].values[0])
    # Q = np.var(rtns)/10
    print(f"var of rtns : {np.var(rtns):.3e}")
    print(f"var of diff2 rtns : {np.var(np.diff(rtns,2)):.3e}")
    # plot_pacf(rtns,lags=10,method='ywm')

    verify_len = 50
    # Rz = .25 * R * (T ** 4)
    Rz = 1./math.factorial(kdim)*(T**kdim)
    Rz = Rz**2 * R
    print(f"Rz = {Rz:.4e}")
    pm = calibrate_kalman_args(z,N = verify_len, opt_method=0,kalman_dim=kdim, R=Rz,dt=T)


    xs,inno = kalmanNdmotion_q(z,R=Rz,q = pm[0], dt=T,dim=kdim)
    print(f"inno mean: {np.mean(inno[-verify_len:]):.4e}, var: {np.var(inno[-verify_len:]):.4e}")

    cal_profit(xs,prices, N= verify_len)

    plt.figure()
    plt.plot(data['Volume'].values[-60:]*1e-6,'.-')

    n = 5
    x = [x for x in range(n)]
    p1 = np.polyfit(x,xs[-n:,0],1)
    p2 = np.polyfit(x,xs[-n:,1],1)
    print(f"ave speed of last {n} days: {p1[0]:.4e}, slope: {p2[0]:.4e}")

    return xs,z,Rz

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <sym> <target_date (current if empty)>".format(sys.argv[0]))
        sys.exit(1)
    sym = sys.argv[1]
    current_date = date.today()
    target_date = current_date.strftime("%Y-%m-%d")
    if len(sys.argv)>2:
        target_date = sys.argv[2]
    else:
        target_date = None

    xs,zz,rz = test_stock(sym,target_date)

    N=-50
    xs = xs[N:,:]
    z = zz[N:]
    fig,axs = plt.subplots(4,1)
    axs[0].plot(xs[:,0],'.-')
    axs[0].plot(z,'r.-')
    rv = xs[:,1]

    axs[1].plot(rv,'.-')
    axs[1].axhline(y=0, color='red', linestyle='-')
    axs[2].plot(xs[:,2],'.-')
    axs[2].axhline(y=0, color='red', linestyle='-')

    # axs[3].plot(xs[N:,-1],'.-')

    # print("est vs real: ", np.corrcoef(xs[:,0],z)[0,1])
    # print("corr: acc vs v: ", np.corrcoef(xs[:,1],xs[:,2])[0,1])
    res = xs[N:,0]-z[N:]
    axs[3].plot(xs[N:,-1],'.-')
    R_res = np.var(res)
    print(f"res mean: {np.mean(res):.3e},var: {R_res:.3e} ")

    print("\033[31mR_pred/R_res: {}\033[0m".format(rz/R_res))

    s,p_val = stats.normaltest(res)
    print("normal test: ",s,p_val)

    plt.show()