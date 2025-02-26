import numpy as np
from params import *
import scipy.signal as ss
from main import (load_target_and_interference_signals, create_room_impulse_response, add_noise, compute_rtf_target,
                  run_prop_relax)

test_ind = 0
k_iters = 10**3
cur_win_length = 4096
cur_hop_length = 2048
cur_nfft = 8192

target_interference = load_target_and_interference_signals()

source_signal = target_interference[test_ind][0]
interference_signal = target_interference[test_ind][1]

# Create sampled data
t_60 = np.random.uniform(T_60_RANGE[0], T_60_RANGE[1])
print(f">>> Randomized T_60: {t_60}[s]")
h_source = create_room_impulse_response(
    ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION, plot_rir=False)
x_t_clean = ss.convolve(h_source, source_signal[:, None])

x_t = add_noise(interference_signal, t_60, x_t_clean)

a_f = compute_rtf_target(x_t_clean, FS, win_length=cur_win_length, hop_length=cur_hop_length,
                         n_fft=cur_nfft)

# Through our experiments, we set the initial value of phi_f to zero,
lmbda = 1.8
print(f">>> lmbda: {lmbda}")
rho = 0.005
print(f">>> rho: {rho}")

score_pesq, score_estoi, score_si_sdr, score_dr = run_prop_relax(
    a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters, plot_fig5=True)
