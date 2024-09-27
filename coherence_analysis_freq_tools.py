import LSP
from LSP import save
import matplotlib.pyplot as plt
import numpy as np

#TODO make separate tools for period and frequency

def makelsp(initial_period, baseline, noise):
    initial_frequency = 1/initial_period
    sampling_rate = 3
    times = [i for i in range(0, baseline, sampling_rate)]
    signal, noisy_signal = LSP.gen_signal(times, amplitude=3, stdev=0.21, noise=noise, initial_frequency=initial_frequency)
    periods, power = LSP.lombScargle(times, noisy_signal, sampling_rate)
    freq = [1/x for x in periods]
    x0 = freq[np.argmax(power)]
    return freq, power, x0
    
def fitting(freq, power, x0):
    x0, gamma, amplitude = LSP.fit_cauchy(freq, power, x0)
    print(f"MLE Estimates: x0 = {x0}, gamma = {gamma}, amp = {amplitude}")
    pdf = LSP.lorentz_curve(x0, gamma, amplitude, freq)
    return pdf, x0, gamma, amplitude

# make peak_period true if calculating using periods rather than frequencies
def math(x0, gamma, peak_period=False):
    f0 = x0
    gamma_f = gamma
    if peak_period:
        f0 = 1/x0
        p0 = x0
        gamma_f = convert_gamma_period_to_freq(gamma, p0)
    coherence = LSP.coherence_time(f0, gamma_f)
    print(f"Period: {1/f0}")
    #lifetime = LSP.mode_lifetime(f0, gamma)
    Q = LSP.Q(f0, gamma_f)
    return Q, coherence

# convert gamma (HWHM) value from frequency to period
def convert_gamma_freq_to_period(gamma, peak_freq):
    return (1/(peak_freq - gamma)) - (1/peak_freq)

# convert gamma (HWHM) value from period to frequency
def convert_gamma_period_to_freq(gamma, peak_period):
    return (1/peak_period) - (1/(peak_period + gamma))