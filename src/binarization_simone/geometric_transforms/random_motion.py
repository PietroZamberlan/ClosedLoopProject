from math import sqrt
from scipy.stats import norm
import scipy
import numpy as np


def brownian(x0, n, dt, delta, out=None, random_state = 2022):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt), random_state = random_state)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out



def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    
    return vola * vola * (1.0 - np.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)


def std(t, mean_rev_speed, vola):
    return np.sqrt(variance(t, mean_rev_speed, vola))\

def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * np.exp(-mean_rev_speed * t) + (1.0 - np.exp(- mean_rev_speed * t)) * mean_rev_level


def ornstein_uhlenbeck(x0, t, mean_rev_speed, mean_rev_level, vola):
    """
    Credits: https://github.com/jwergieluk/ou_noise
    Simulates a sample path
    """
   
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = np.diff(t)
    scale = std(dt, mean_rev_speed, vola)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += mean(x[i - 1], dt[i - 1], mean_rev_speed, mean_rev_level)
    return x

