import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from scipy.optimize import minimize
import math
from gatspy.periodic import LombScargle
from pathlib import Path
from enum import Enum

# Function to save any image without replacing old ones with the same title
def save(title, path=None):
    if path is None:
        path = f"/default/path/to/file"
    img_num = 1
    fullpath = Path(path + f"{title}{img_num}.png")
    while fullpath.is_file():
        img_num += 1
        fullpath = Path(path + f"{title}{img_num}.png")
    plt.savefig(fullpath, bbox_inches='tight', transparent='true')

# To set different kinds of noise (will update when more kinds are implemented)
class NoiseType(Enum):
    NO_NOISE = 0
    GAUSSIAN = 1
    DAMPED_RANDOM_WALK = 2

# Function to generate sinusoidal signal with noise
def gen_signal(times, sampling_rate=3, amplitude=1, initial_frequency="random", stdev=0, noise: NoiseType=NoiseType.NO_NOISE, y0=0, phase_shift=0):
    if initial_frequency == "random":
        initial_frequency = np.random.uniform(0.001, 0.01)
    if sampling_rate == "random":
        pass # implement this later if we end up trying irregular sampling
    print(initial_frequency)
    signal = [(amplitude * math.sin(x*2*math.pi*initial_frequency - phase_shift)) + y0 for x in times]
    # Gaussian noise
    if noise == NoiseType.NO_NOISE:
        print("Pure signal generated.")
        return signal, signal
    if stdev == 0:
        print("Noise added but standard deviation is zero. Signal will not be noisy.")
    if noise == NoiseType.GAUSSIAN:
        noise = np.random.normal(0, stdev, len(times)) # mean is 0
    elif noise == NoiseType.DAMPED_RANDOM_WALK:
        print("This noise option not implemented yet.")
        # DRW noise creation is in separate file for now
        # TODO combine them
        return signal, signal
    signal_noisy = signal + noise
    return signal, signal_noisy

# Function to apply Lomb-Scargle Periodogram.
# Takes in arraylike objects with times and corresponding signal
# Returns periodicities found in the times with corresponding powers
# Period corresponding with highest power is most likely the period found in the signal
def lombScargle(times, signal, sampling_rate=None):
    lomb_scargle = LombScargle()
    lomb_scargle.fit(times, signal)
    if sampling_rate is not None:
        period, power = lomb_scargle.periodogram_auto(nyquist_factor=(sampling_rate / 2))
    else:
        period, power = lomb_scargle.periodogram_auto(nyquist_factor=(1.5))
    return period, power

# Function that returns the Lorentzian (or Cauchy) distribution
# x0 - location of peak
# gamma - 1/2 of FWHM
# amp - scaling factor (up/down)
# x - arraylike of x values for which Lorentzian is computed
def lorentz_curve(x0, gamma, amp, x):
    return [amp * (gamma ** 2 / (gamma ** 2 + (i - x0) ** 2)) for i in x]

# Function to compute the log-likelihood
# Takes in a list of guesses with x0, gamma, amplitude
# Takes in data we are fitting to, and x values for these data
# Returns log-likelihood for that set of guesses
def log_likelihood(guesses, x0, data, x):
    gamma, amplitude = guesses
    model = lorentz_curve(x0, gamma, amplitude, x)
    return -np.sum(cauchy.logpdf(data, loc=model))

# Function to fit Cauchy (Lorentzian) distribution
# Takes in data and corresponding x values, and location parameter x0 guess
# Returns best fit parameters for x0, gamma, amplitude
def fit_cauchy(x, data, x0):
    data_peak = data[np.argmax(data)]
    gamma_guess = data_peak / 2
    amplitude_guess = max(data)

    bounds = [(None, None), ((max(data) - (max(data) / 4)), (max(data) + (max(data / 4))))]
    result = minimize(log_likelihood, [gamma_guess, amplitude_guess], args=(x0, data, x), bounds=bounds)

    #result = minimize(log_likelihood, [gamma_guess, amplitude_guess], args=(x0, data, x), bounds=[(None, None), (0.4, 0.5)])
    gamma, amplitude = result.x
    gamma = abs(gamma)
    return x0, gamma, amplitude

# Function to find closest value to a target in a list, uses modified binary search algorithm
def match(inlist, target):
    if len(inlist) == 0:
        return -1
    minIndex, maxIndex = 0, len(inlist) - 1
    while minIndex < maxIndex:
        current = (maxIndex - minIndex) // 2 + minIndex
        if inlist[current] == target:
            return current
        elif inlist[current] < target:
            minIndex = current + 1
        else:
            maxIndex = current
    if minIndex == 0:
        return 0
    if abs(inlist[minIndex] - target) < abs(inlist[minIndex - 1] - target):
        return minIndex
    return minIndex - 1
    

# Function to plot FWHM and peak
def plot_fwhm(x0, gamma, x, y):
    fwhm = 2 * gamma
    ind = match(x, x0)
    peak_loc = x[ind]
    peak = y[ind]
    
    plt.axvline(x0 - gamma, color='green', linestyle='--', label=f'FWHM: {fwhm:.7f}')
    plt.axvline(x0 + gamma, color='green', linestyle='--')
    plt.axhline(peak, color='red', linestyle='--', label=f'Peak: {peak_loc:.5f}')
    plt.axvline(peak_loc, color='red', linestyle='--')
    plt.legend()

# Math given Lorentzian shape of power spectrum

# Function to calculate quality factor
# Given HWHM (gamma) and peak frequency
def Q(peak_freq, gamma):
    return peak_freq / (2 * gamma)

# Function to calculate coherence time
# Given HWHM (gamma) and peak frequency
def coherence_time(peak_freq, gamma):
    return Q(peak_freq, gamma) / (math.pi * peak_freq)
