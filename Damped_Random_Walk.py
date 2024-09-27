import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tau is characteristic timescale
# sigma is magnitude of noise
# mean is the mean that the process eventually returns to
# can use the mean of the real oscillatory signal
def ornstein_uhlenbeck_process(theta, sigma, tau, X0, times, num_runs=1):
    """
    Simulates the Ornstein-Uhlenbeck process using the exact discretization.

    This follows the definition as being a stochastic differential equation:
    dX(t) = -tau * x(t)dt + sigma * dW(t)
    Where dW(t) denotes a Weiner process, or a Gaussian continuous process with independent increments.

    This equation is solved so that the expected value of x(t) given x(s) is:
    E(X(t)|X(s)) = e^(-dt/tau) * X(s) + b * tau * (1 - e^(-dt/tau).
    
    The numerical simulation of this follows the example given at:
    https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/6.1%20Ornstein-Uhlenbeck%20process%20and%20applications.ipynb
    
    Parameters:
    theta (float): Long-term mean (mean reversion level).
    sigma (float): Amplitude of noise.
    tau (float): Mean reversion time constant.
    X0 (float): Initial value of the process.
    T (float): Total time to simulate.
    dt (float): Sampling rate.
    
    Returns:
    List of tuples - for every iteration of this process the times and corresponding noise values are
        added to the list.
    """
    returnvalue = []
    for run in range(num_runs):
        X = np.zeros(len(times))
        X[0] = X0

        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            G = np.random.normal(0, 1)
            X[i] = (theta + 
                    (X[i-1] - theta) * np.exp(-dt / tau) + 
                    sigma * np.sqrt(tau / 2 * (1 - np.exp(-2 * dt / tau))) * G)
        
        returnvalue.append((times, X))
    return returnvalue


# Example

'''
data_path = "/Users/aliceheranval/Desktop/research/caltech reu/data_files/"
data = {}

num=1
data[num] = pd.read_csv(f'{data_path}kb{num}_g.dat', delim_whitespace=True)

sourcenum = 1

current = data[sourcenum]
mjd = current.iloc[:, 0].tolist()
mag = current.iloc[:, 1].tolist()

times, noise = ornstein_uhlenbeck_process(0, 1, 200, 0, mjd, 1)[0]
plt.figure(figsize=(10, 6))
plt.scatter(times, noise)
plt.show()
'''
