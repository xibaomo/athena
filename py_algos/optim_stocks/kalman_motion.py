import math
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import spearmanr
from scipy.optimize import minimize
import yfinance as yf
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
        X_ = np.matmul(F,X)
        P_ = np.matmul(np.matmul(F,P),F.transpose()) + np.matmul(G*Q,G.transpose())
        tmp = np.matmul(np.matmul(H,P_),H.transpose()) + Rm
        # pdb.set_trace()
        K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(tmp))
        tmp = Z[i] - np.matmul(H,X_)
        X = X_ + np.matmul(K,tmp)
        states[i,:-1] = X
        # pdb.set_trace()
        states[i,-1] = K[0][0]

        tmp = np.eye(dim) - np.matmul(K,H)
        # pdb.set_trace()
        P = np.matmul(np.matmul(tmp,P_),tmp.transpose()) + np.matmul(np.matmul(K,Rm),K.transpose())

    # print("FInal P: ",P)
    return states,P
def cal_profit(x,Z,N=100,cap=10000):
    dt, R, q = x
    if dt <=0 or R <=0 or q <= 0:
        return np.inf,[]
    xs, P = kalman_motion(Z, R, q, dt)
    k_eq = xs[-1,-1]
    price = Z[-N:]
    v = xs[-N:, 1]
    is_pos_on = False
    p0 = -1.
    transactions=[]
    c0=cap
    for i in range(N):
        if v[i] > 0 and not is_pos_on and abs(xs[i,-1]-k_eq) < 0.01:
            p0 = price[i]
            is_pos_on = True
            trans = [i,-1]
            transactions.append(trans)
        if v[i] <= 0 and is_pos_on:
            is_pos_on = False
            cap = cap / p0 * price[i]
            transactions[-1][1] = i
    if is_pos_on:
        is_pos_on = False
        cap = cap / p0 * price[-1]
        transactions[-1][1] = N-1

    # print("R,q,profit:{:.2f}, {:.2f}, {:.2f} ".format(R,q,cap-c0))
    return -(cap-c0),transactions
def calibrate_kalman_args(Z,N=100,opt_method=0):
    def obj_func(x,params):
        if len(x)==0:
            pdb.set_trace()
        Z,N = params
        cost,_ = cal_profit(x,Z,N)
        return cost

    init_x = np.array([.1,100,20])
    bounds = [(1e-2,1e-1),(.1,100),(.1,100)]
    # bounds = None
    result = None
    if opt_method == 0:
        result = minimize(obj_func,init_x,args=((Z,N),),bounds=bounds,method='COBYLA',tol=1e-5)
    elif opt_method == 1:
        result = ga_minimize(obj_func,(Z,N),3,bounds,population_size=200,num_generations=100)
    else:
        print("ERROR! opt_method is not supported")

    # print("optimal dt,R: ", result.x)
    # print("optimal cost: ", result.fun)
    return result.x

def test_stock():
    syms = ['mhk']
    target_date = '2024-08-30'
    back_days = 250
    start_date = add_days_to_date(target_date,-back_days)
    data = yf.download(syms, start=start_date, end=target_date)
    df = data['Close']
    # pdb.set_trace()
    print(df.index[-1])

    z = df.values #/ df.values[0]
    pm = calibrate_kalman_args(z,opt_method=0)
    xs,_ = kalman_motion(z,R=pm[1],q=pm[2],dt=pm[0])
    # xs,_ = kalman_motion(z,R=100,q=1,dt=.01)
    pf,trans=cal_profit(pm,z)
    print("optimal dt,R,q: ",pm)
    print("Profit: {:.2f}, trans: {}".format(pf,trans))
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
    # xs = test_uniform_motion()
    xs,z = test_stock()
    # xs,z = test_stock_iter()
    N=1
    xs = xs[N:,:]
    z = z[N:]
    fig,axs = plt.subplots(3,1)
    axs[0].plot(xs[:,0],'.-')
    axs[0].plot(z,'r.-')
    axs[1].plot(xs[:,1],'.-')
    axs[1].axhline(y=0, color='red', linestyle='-')
    # axs[2].plot(xs[:,2],'.-')
    # axs[2].axhline(y=0, color='red', linestyle='-')
    axs[2].plot(xs[N:,-1],'.-')

    print("est vs real: ", np.corrcoef(xs[:,0],z)[0,1])
    print("corr: acc vs v: ", np.corrcoef(xs[:,1],xs[:,2])[0,1])
    res = xs[-100:,0]-z[-100:]
    print("res mean,var: ", np.mean(res),np.var(res))

    s,p_val = stats.normaltest(res)
    print("normal test: ",s,p_val)
    result = acorr_ljungbox(res, lags=[10], return_df=True)
    print("white noise: ", result['lb_pvalue'].values[0])

    plot_acf(res, lags=20)
    plt.show()