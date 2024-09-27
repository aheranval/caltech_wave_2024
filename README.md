# caltech_wave_2024
Code written during the summer of 2024 for a project at Caltech testing the coherence of periodicity in AGN signals from supermassive black hole binaries.

Split into several files:
- LSP.py implements the Lomb-Scargle Periodogram, as well as includes code for generating a signal to analyze, fitting a Lorentzian to a periodogram, and subsequent calculations of Q factor and coherence time.
- Damped_Random_Walk.py has code for simulating damped random walk, as well as sample implementation with reading from a .dat file (not provided) and using its times to generate an Ornstein-Uhlenbeck process.
- Detrending_Code.py allows you to apply a 5th-order polynomial detrending algorithm to data for better periodogram results.
- coherence_analysis_freq_tools.py is a Python file with some example implementation of LSP in the context of analyzing the coherence times of your data or of generated signals.

Suggested order of using this code:
1. Detrend signal
2. Generate LSP for signal
3. Fit Lorentzian to find FWHM
4. Use this to calculate damping factor (Q) and coherence time
5. Generate damped random walk that looks like the signal
6. Detrend damped random walk
7. Fit and calculate coherence time for damped random walk
8. Compare the two
