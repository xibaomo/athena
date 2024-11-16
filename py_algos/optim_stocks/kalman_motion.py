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

def __comp_grad(y):
    x = [x for x in range(len(y))]
    x = np.array(x)

    spline = CubicSpline(x, y)

    # Compute the derivative at each point
    y_derivative = spline.derivative()(x)

    return y_derivative

def comp_grad(y):
    return np.diff(y)

def __estimate_poly_order(y,max_order=20,tol=1e-1,rel_tol=1e-1):
    def cost(x,y,order):
        p = np.polyfit(x,y,order)
        pfit = np.poly1d(p)
        err = y - pfit(x)
        mse = np.sqrt(np.mean(err**2))

        return mse
    x = np.linspace(0,1,len(y))
    x = (x-np.mean(x))/np.std(x)
    err = cost(x,y,1)
    n = 2
    while n < max_order:
        new_err = cost(x,y,n)
        if new_err < tol:# and abs(new_err-err)/err < rel_tol:
            break
        # print(abs(err-new_err)/err)
        print(f"{new_err}")
        err = new_err
        n+=1
    return n,cost(x,y,n)

def estimate_kalman_dim(z,max_dim=20):
    # pdb.set_trace()
    #check which order of differencing gives minimum variance
    min_mu = 9999
    opt_dim = -1
    for i in range(1,max_dim+1):
        mu = abs(np.mean(np.diff(z,i)))
        if mu < min_mu:
            opt_dim = i
            min_mu = mu
    print(f"optimal diff order: {opt_dim}, min abs(mean): {(min_mu)}")
    return opt_dim

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
def kalmanNdmotion(Z,R,q,dt,dim=-1):
    N = len(Z)
    Rm = np.array([[R]])
    states = np.zeros((N, dim + 1))
    T = dt
    X = np.zeros(dim)
    X[0] = Z[0]
    P = np.eye(dim) * R
    # pdb.set_trace()
    F = np.zeros((dim,dim))
    for i in range(dim-1):
        F[i,i+1] = 1.
    Phi = np.eye(dim)

    # fill matrix Phi
    for i in range(1,dim):
        f = 1./math.factorial(i)
        Phi = Phi + np.linalg.matrix_power(F,i) * (T**i)

    # fill column vector G
    G = np.zeros((dim,1))
    for i in range(dim):
        f = 1./math.factorial(dim-i)
        G[i,0] = f * (T**(dim-i))

    QQ = G @ G.T * q / T
    # H = np.array([[1., 0., 0., 0., 0]])
    H = np.zeros((1,dim))
    H[0,0]=1.
    for i in range(N):
        if i == 0:
            continue
        # pdb.set_trace()
        X_ = np.matmul(Phi, X)
        P_ = np.matmul(np.matmul(Phi, P), Phi.transpose()) + QQ
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
def __cal_profit(x,log_price,N=100,cap=10000):
    Z = np.exp(log_price)
    q, = x
    if  q <= 0:
        return -1.e20,[],99999.

    # xs, P = kalman_motion(log_price, R, q, dt)
    xs,P = kalman2dmotion_adaptive(log_price,q)
    # return -P[0,0]/R,[]
    # vstd = np.sqrt(P[1,1])
    k_eq = xs[-1,-1]
    price = Z[-N:]
    s = xs[-N:,0]
    v = xs[-N:, 1]
    # a = xs[-N:,2]
    is_pos_on = False
    p0 = -1.
    transactions=[]
    c0=cap
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

    # print("R,q,profit:{:.2f}, {:.2f}, {:.2f} ".format(R,q,cap-c0))
    err = xs[-N:,0] - log_price[-N:]
    sd = np.mean(err**2)
    # ps = P[-N//2:,0,0]
    return (cap-c0),transactions,sd
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

KALMAN_FUNC = kalmanNdmotion
def obj_func(x,params):
    Z,N,kalman_dim = params
    Ri,qi,dt = x
    R = 10**Ri
    q = 10**qi
    if R < q:
        return 9999,

    xs, P = KALMAN_FUNC(Z, R, q=q, dt=dt,dim=kalman_dim)

    nu = Z[-N:] - xs[-N:,0]
    mu = np.mean(nu)
    var = np.var(nu)

    # partial_autocorr = pacf(nu, method='ywm', nlags=5)
    v = xs[-N:,1]

    # smoothness = compute_smoothness(v)
    # cost =  abs(mu) + (abs(var/R-1))  + R*1e3+smoothness*50

    s,pval = stats.normaltest(nu)
    cost =  abs(var/R-1) - pval + P[1,1]
    return cost,
    # ax
def ___obj_func(x,params):
    Z,N = params
    cost,trans,sd = cal_profit(x,Z,N)
    if len(trans)==0:
        return 0,
    return -cost/len(trans),
    # return -cost/sd,
    # return sd,
def __obj_func(x,params):
    Z,N = params
    q,=x
    xs, P = kalman2dmotion_adaptive(Z, q)
    # pdb.set_trace()
    res = Z[-N:] - xs[-N:,0]
    lb_test = acorr_ljungbox(res, lags=[20], return_df=True)

    return lb_test['lb_stat'].values[0],

def calibrate_kalman_args(Z,N=50,opt_method=0,kalman_dim=-1):
    init_x = np.array([1e-4,2e-4,.1])
    bounds = [(-5,-1),(-5,-3),(.2,.2)]
    # bounds = None
    result = None
    # pdb.set_trace()
    if opt_method == 0:
        result = minimize(obj_func,init_x,args=((Z,N,kalman_dim),),bounds=bounds,method='COBYLA',tol=1e-5)
    elif opt_method == 1:
        tmp_func = functools.partial(obj_func,params=(Z,N,kalman_dim))
        result = ga_minimize(tmp_func,len(init_x),bounds,population_size=2000,num_generations=100)
        # print("ga result: ", result.x, result.fun)
        # result = minimize(obj_func,result.x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
    else:
        print("ERROR! opt_method is not supported")

    # print("optimal R,dt: ", result.x)
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
    back_days = 100
    start_date = add_days_to_date(target_date,-back_days)
    # data = yf.download(syms, start=start_date, end=target_date)
    data = yf.Ticker(sym).history(start=start_date, end=target_date, interval='1d')
    # pdb.set_trace()
    df = data['Close']
    prices = expand_price(data)

    print("length of history: ", len(prices))
    z = np.log(prices) #/ df.values[0]
    # pdb.set_trace()

    kalman_dim = estimate_kalman_dim(z)
    print("Optimal kalman order: ", kalman_dim)
    rtns = np.diff(z)
    result = acorr_ljungbox(rtns, lags=[10], return_df=True)
    print("is rtns white noise: ", result['lb_pvalue'].values[0])
    # Q = np.var(rtns)/10
    print(f"var of rtns : {np.var(rtns):.3e}")
    print(f"var of diff2 rtns : {np.var(np.diff(rtns,2)):.3e}")
    # plot_pacf(rtns,lags=10,method='ywm')

    verify_len = 50
    pm = calibrate_kalman_args(z,N = verify_len, opt_method=1,kalman_dim=kalman_dim)

    xs,p = KALMAN_FUNC(z,R=10**pm[0],q=10**pm[1],dt=pm[2],dim=kalman_dim)

    # pf,trans,sd=cal_profit(pm,z)
    print(f"optimal R: {10**pm[0]:.4e},q: {10**pm[1]:.4e}")
    print("est. variance of log-price: ", p[0,0])

    cal_profit(xs,prices, N= verify_len)

    n = 5
    x = [x for x in range(n)]
    p1 = np.polyfit(x,xs[-n:,0],1)
    p2 = np.polyfit(x,xs[-n:,1],1)
    print(f"ave speed of last {n} days: {p1[0]:.4e}, slope: {p2[0]:.4e}")

    return xs,z,pm

def test_stock_iter():
    syms = ['smci']
    target_date = '2024-08-23'
    back_days = 250
    start_date = add_days_to_date(target_date,-back_days)
    data = yf.download(syms, start=start_date, end=target_date)
    df = data['Close']
    # pdb.set_trace()
    print(df.index[-1])

    z = df.values / df.values[0]
    qs = np.linspace(1e-8,1e-2,100)
    ts = np.linspace(1e-3,10,100)
    ps = []
    for t in ts:
        xs,_ = kalman_motion(z,R=2.5e-3,q=1e-6,dt=t)
        res = xs[-100:,0] - z[-100:]
        # result = acorr_ljungbox(res, lags=[10], return_df=True)
        # print("white noise: ", result['lb_pvalue'].values[0])
        # cost = result['lb_pvalue'].values[0]
        cost = abs(np.mean(res))
        ps.append(cost)

    plt.plot(qs,ps,'.')
    plt.show()

    sys.exit()
    return xs,z

def test_uniform_motion():
    N = 100
    v = 1
    R = 1
    z = v * np.array([x for x in range(N)])
    noise = np.random.normal(0, 1, N)
    z = z + noise * np.sqrt(R)
    xs = kalman_motion(z, R)
    # plt.figure()
    # plt.plot(xs[:, 0], '.-')
    # plt.figure()
    # plt.plot(xs[:, 1], '.-')
    # plt.figure()
    # plt.plot(xs[:, 2], '.-')
    # plt.show()
    return xs

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

    xs,zz,pm = test_stock(sym,target_date)

    # xs,z = test_stock_iter()
    N=-50
    xs = xs[N-1:,:]
    z = zz[N:]
    fig,axs = plt.subplots(4,1)
    axs[0].plot(xs[:,0],'.-')
    axs[0].plot(z,'r.-')
    rv = xs[:,1]*10**pm[2]
    # rv = np.diff(xs[:,0])
    # rv = comp_grad(xs[:,0])
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

    print("\033[31mR_pred/R_res: {}\033[0m".format(10**pm[0]/R_res))

    s,p_val = stats.normaltest(res)
    print("normal test: ",s,p_val)
    rs = res[N:] - np.mean(res[N:])
    result = acorr_ljungbox(rs, lags=[10], return_df=True)
    print("white noise: ", result['lb_pvalue'].values[0])

    plot_pacf(rs, lags=10,method='ywm')
    # plot_acf(res[-100:],lags=30)
    # plt.hist(res)
    plt.show()