from params import *
import scipy.signal as ss
from main import (load_target_and_interference_signals, create_room_impulse_response, add_noise, compute_rtf_target,
                  run_prop_relax, run_prop_exact, run_prop_relax_with_initialization,
                  display_time_frequency_signal)

test_ind = 0
k_iters = 500
cur_win_length = 2048
cur_hop_length = 1024
cur_nfft = 4096
print(f'{k_iters} *** {cur_win_length}||{cur_nfft}||{cur_hop_length}')

target_interference = load_target_and_interference_signals()
source_signal = target_interference[test_ind][0]
interference_signal = target_interference[test_ind][1]
display_time_frequency_signal(source_signal, cur_hop_length, cur_nfft, cur_win_length, title='Source Signal')

# Create sampled data
t_60 = np.random.uniform(T_60_RANGE[0], T_60_RANGE[1])
print(f">>> Randomized T_60: {t_60}[s]")
h_source = create_room_impulse_response(
    ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION, plot_rir=True)
x_t_clean = ss.convolve(h_source, source_signal[:, None])
x_t = add_noise(interference_signal, t_60, x_t_clean)
display_time_frequency_signal(x_t[:, REF_MIC_INDEX], cur_hop_length, cur_nfft, cur_win_length, title='Noised Signal')

a_f = compute_rtf_target(x_t_clean, FS, win_length=cur_win_length, hop_length=cur_hop_length,
                         n_fft=cur_nfft)
# Through our experiments, we set the initial value of phi_f to zero,
lmbda = 1.8
print(f">>> lmbda: {lmbda}")
rho = 0.005
print(f">>> rho: {rho}")

score_pesq1, score_estoi1, score_si_sdr1, score_dr1 = run_prop_relax(
    a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters, plot_fig5=True)
score_pesq2, score_estoi2, score_si_sdr2, score_dr2 = run_prop_exact(
    a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters, plot_fig5=True)
score_pesq3, score_estoi3, score_si_sdr3, score_dr3 = run_prop_relax_with_initialization(
    a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters, plot_fig5=True)
