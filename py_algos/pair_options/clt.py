from arch import arch_model
import numpy as np
from scipy.stats import norm


def compute_multistep_distribution_garch(rtns, steps, lb_rtn, ub_rtn, drtn):
    model = arch_model(rtns * 1000, mean="Constant", vol="GARCH", p=1, q=1, dist='t')
    fit = model.fit(disp="off")

    print(fit.summary())

    # Forecast future volatility
    horizon = steps
    forecast = fit.forecast(horizon=horizon)

    # Conditional variance forecasts
    future_var = forecast.variance.values[-1, :]

    cum_mean = np.mean(rtns[-steps*3:]) * steps # zero mean
    cum_var = np.sum(future_var)
    cum_std = np.sqrt(cum_var)/1000

    # breakpoint()
    # Range of cumulative returns
    nr = int((ub_rtn - lb_rtn) // drtn)
    probs = np.zeros(nr)
    for i in range(nr):
        probs[i] = norm.cdf((i + 1) * drtn + lb_rtn, loc=cum_mean, scale=cum_std) - norm.cdf(i * drtn + lb_rtn, loc=cum_mean,
                                                                                    scale=cum_std)

    return probs
