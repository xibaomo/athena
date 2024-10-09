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
folder = os.environ['ATHENA_HOME'] + "/py_algos/py_basics"
sys.path.append(folder)
from ga_min import *

def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str

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

def kalman2dmotion_adaptive(Z,q,dt=0.05):
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

def kalman_motion(Z,R,q=1e-3,dt=1):
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
    G = np.array([[T**3/6],[T**2/2],[1.]])
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
def cal_profit(x,log_price,N=100,cap=10000):
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

def obj_func(x,params):
    Z,N = params
    cost,trans,sd = cal_profit(x,Z,N)
    if len(trans)==0:
        return 0,
    return -cost/len(trans),
    # return -cost/sd,
    # return sd,
def calibrate_kalman_args(Z,N=100,opt_method=0):
    init_x = np.array([.1])
    bounds = [(1e-5,1.e-3)]
    # bounds = None
    result = None
    # pdb.set_trace()
    if opt_method == 0:
        result = minimize(obj_func,init_x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
    elif opt_method == 1:
        tmp_func = functools.partial(obj_func,params=(Z,N))
        result = ga_minimize(tmp_func,len(init_x),bounds,population_size=1000,num_generations=100)
        # print("ga result: ", result.x, result.fun)
        # result = minimize(obj_func,result.x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
    else:
        print("ERROR! opt_method is not supported")

    # print("optimal dt,R: ", result.x)
    # print("optimal cost: ", result.fun)
    return result.x

def test_stock(sym,target_date=None):
    syms = [sym]
    # target_date = '2024-09-5'
    back_days = 350
    start_date = add_days_to_date(target_date,-back_days)
    data = yf.download(syms, start=start_date, end=target_date)
    df = data['Close']
    # pdb.set_trace()
    print(df.index[-1])

    z = np.log(df.values) #/ df.values[0]
    pm = calibrate_kalman_args(z,opt_method=1)

    # pdb.set_trace()
    # pm = [1e-5]
    xs,p_est = kalman2dmotion_adaptive(z,q=pm[0])

    pf,trans,sd=cal_profit(pm,z)
    print("optimal dt,R,q: ",pm)
    print("Profit: {:.2f}, trans: {}".format(pf,trans))
    print("ave profit: {:.2f}".format(pf/len(trans)))
    print("std of err: ", sd)
    print("estimated var: ",p_est[-5:,:])
    return xs,z

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

    xs,zz = test_stock(sym,target_date)
    # xs,z = test_stock_iter()
    N=-100
    xs = xs[N:,:]
    z = zz[N:]
    fig,axs = plt.subplots(3,1)
    axs[0].plot(xs[:,0],'.-')
    axs[0].plot(z,'r.-')
    axs[1].plot(xs[:,1],'.-')
    axs[1].axhline(y=0, color='red', linestyle='-')
    # axs[2].plot(xs[:,2],'.-')
    # axs[2].axhline(y=0, color='red', linestyle='-')
    axs[2].plot(xs[N:,-1],'.-')

    # print("est vs real: ", np.corrcoef(xs[:,0],z)[0,1])
    # print("corr: acc vs v: ", np.corrcoef(xs[:,1],xs[:,2])[0,1])
    res = xs[-100:,0]-z[-100:]
    print("res mean,var: ", np.mean(res),np.var(res))

    s,p_val = stats.normaltest(res)
    print("normal test: ",s,p_val)
    result = acorr_ljungbox(xs[-100:,1], lags=[10], return_df=True)
    print("white noise: ", result['lb_pvalue'].values[0])

    plot_pacf(xs[-100:,1], lags=30,method='ywm')
    plt.show()