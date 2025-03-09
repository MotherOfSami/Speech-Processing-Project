import os.path
import librosa
import numpy as np


import rir_generator as rir
import matplotlib.pyplot as plt
import scipy.signal as ss
import pandas as pd
from params import *
from display_funcs import display_audio_spectrogram, display_audio_time_domain, display_rir_time_domain
from scipy.linalg import eigh
import soundfile as sf
import glob
from scores_utils import evaluate_pesq_score, evaluate_estoi_score, evaluate_si_sdr_score, evaluate_distortion_ratio
from prop_algs import prop_relax, prop_exact, prop_relax_online
from figs_creation import create_fig5, create_fig6
from scipy.signal import convolve
from localization_utils import initialize_with_steering_vec
from tqdm import tqdm

def apply_stft(y, sr, win_length, hop_length, n_fft=None):
    # calculate stft params
    if n_fft is None:
        n_fft = 2 ** (win_length - 1).bit_length()

    # apply stft
    y_stft = librosa.stft(y,
                          win_length=win_length,
                          hop_length=hop_length,
                          n_fft=n_fft
                          )
    return y_stft


def load_input_signal(filename: str, plot_signal: bool = True):
    # Load the data
    signal, fs = librosa.load(filename, sr=FS)
    if plot_signal:
        display_audio_time_domain(signal, fs, title='original signal')
        display_audio_spectrogram(signal, fs, win_length=WIN_LENGTH_LIST[0],
                                  hop_length=HOP_LENGTH_LIST[0],
                                  n_fft=N_FFT_LIST[0])
    return signal, fs


def create_room_impulse_response(
        room_dimensions, mic_positions, fs, reverb_time, source_position,
        plot_mic_index: int = 0, plot_rir: bool = True):
    h_reverberation = generate_rir(room_dimensions, mic_positions, fs, reverb_time, source_position)
    if plot_rir:
        # print(f'h_reverberation shape = {h_reverberation.shape}')
        display_rir_time_domain(fs,
                                f'Room Impulse Response (first mic). Reverberation Time: {reverb_time} sec, source: {source_position}',
                                h_reverberation, plot_mic_index=plot_mic_index)
    return h_reverberation


def generate_rir(room_dimensions, mic_positions, fs, reverb_time, source_position):
    n_sample = np.round(reverb_time * fs).astype(int)
    h = rir.generate(
        c=340,  # Sound velocity (m/s)
        fs=fs,  # Sample frequency (samples/s)
        r=np.array(mic_positions),
        s=source_position,  # Source position [x y z] (m)
        L=room_dimensions,  # Room dimensions [x y z] (m)
        reverberation_time=reverb_time,  # Reverberation time (s)
        nsample=n_sample,  # Number of output samples
    )
    return h


def read_wav_files_from_folder(num_samples):
    dir_path = 'dataset'
    speakers_dirs = glob.glob(dir_path + '/*/*/')
    num_speakers = len(speakers_dirs)
    test_audio_files = np.random.randint(num_speakers, size=(NUM_AUDIO_MIXTURES, 2))
    all_records = []
    for test_ind in range(test_audio_files.shape[0]):
        cur_target_path = speakers_dirs[test_audio_files[test_ind, 0]]
        cur_target_record_start = test_audio_files[test_ind, 1]
        files = glob.glob(cur_target_path + '*.wav')[cur_target_record_start:]
        target_record = []
        for filename in files:
            signal, _ = load_input_signal(filename, plot_signal=False)
            target_record.append(signal)
        target_record = np.concatenate(target_record)
        target_record = target_record[:num_samples]
        all_records.append(target_record)
    return all_records


def compute_rtf_target(s_t, fs, win_length, hop_length, n_fft):
    """The RTF of the target source was computed by applying the eigenvalue decomposition to the SCM
    of another clean source image uttered by the same speaker from the same position

    s_t = clean source image in time domain
    """
    s_n_f = []
    _, n_mics = s_t.shape

    for cur_mic in range(n_mics):
        s_stft_m = apply_stft(s_t[:, cur_mic], fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
        s_n_f += [s_stft_m]
    s_n_f = np.stack(s_n_f)
    n_freq_bins, n_time_frames = s_stft_m.shape
    print(f'n_freq_bins={n_freq_bins}, n_time_frames={n_time_frames}')
    rtf_f = []

    for f in range(n_freq_bins):
        cur_s_n_f = s_n_f[:, f, :]
        cov_x_f = (cur_s_n_f @ cur_s_n_f.conj().T) / n_time_frames
        eigvals, eigvecs = eigh(cov_x_f, eigvals_only=False)
        largest_eigenvec = eigvecs[:, -1]
        cur_rtf_f = largest_eigenvec / largest_eigenvec[0]
        rtf_f += [cur_rtf_f]

    rtf_f = np.stack(rtf_f)
    return rtf_f


def save_audio_signals(fs, signal, filename: str):
    sf.write(filename, signal, fs)


def calc_filtered_signal(w: np.ndarray, sig: np.ndarray) -> np.ndarray:
    num_of_samples, num_of_mics = sig.shape
    res = np.zeros(num_of_samples, dtype=np.complex128)
    sig = sig.astype(np.complex128)
    for mic in range(num_of_mics):
        res += convolve(sig[:, mic], w[mic, :], mode='same')
    return res


def test_prop_relax(target_interference):
    filename_relax = f'performance_prop_relax.csv'
    filename_exact = f'performance_prop_exact.csv'
    filename_relax_init = f'performance_prop_relax_init.csv'

    num_iters = [10, 50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 10 ** 4,
                 2 * (10 ** 4), 3 * (10 ** 4)]
    for param_ind in range(2, len(WIN_LENGTH_LIST) - 1):
        cur_win_length, cur_nfft, cur_hop_length = (WIN_LENGTH_LIST[param_ind], N_FFT_LIST[param_ind],
                                                    HOP_LENGTH_LIST[param_ind])
        print(f'{cur_win_length}||{cur_nfft}||{cur_hop_length}')
        for k_iters in num_iters:
            for test_ind in range(len(target_interference)):
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
                run_save_alg(a_f, cur_hop_length, cur_nfft, cur_win_length, filename_relax, k_iters, lmbda, param_ind,
                             rho, test_ind, x_t, x_t_clean, alg_type='prop-relax')
                run_save_alg(a_f, cur_hop_length, cur_nfft, cur_win_length, filename_exact, k_iters, lmbda, param_ind,
                             rho, test_ind, x_t, x_t_clean, alg_type='prop-exact')
                run_save_alg(a_f, cur_hop_length, cur_nfft, cur_win_length, filename_relax_init, k_iters, lmbda,
                             param_ind, rho, test_ind, x_t, x_t_clean, alg_type='prop-relax-init')


def test_prop_relax_online_stationary(target_interference):
    source_signal = target_interference[0][0][:int(WIN_LENGTH_LIST[2] * 10)]
    interference_signal = target_interference[0][1][:int(WIN_LENGTH_LIST[2] * 10)]

    # Create sampled data
    t_60 = np.random.uniform(T_60_RANGE[0], T_60_RANGE[1])
    print(f">>> Randomized T_60: {t_60}[s]")
    h_source = create_room_impulse_response(
        ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION, plot_rir=False)
    x_t_clean = ss.convolve(h_source, source_signal[:, None])

    save_audio_signals(16000, x_t_clean, "clean_signal.wav")

    x_t = add_noise(interference_signal, t_60, x_t_clean)

    save_audio_signals(16000, x_t, "noisy_signal.wav")

    a_f = compute_rtf_target(x_t_clean, FS, win_length=WIN_LENGTH_LIST[2], hop_length=HOP_LENGTH_LIST[2],
                             n_fft=N_FFT_LIST[2])
    lmbda = 1.8
    rho = 0.005

    y_t = run_prop_online(a_f, HOP_LENGTH_LIST[2], N_FFT_LIST[2], WIN_LENGTH_LIST[2], lmbda, rho, x_t)
    return y_t


def test_prop_relax_online_moving(target_interference):
    source_signal = target_interference[0][0]
    interference_signal = target_interference[0][1]

    t_60 = 0.25
    h_source = create_room_impulse_response(
        ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION, plot_rir=False)
    x_t_clean = ss.convolve(h_source, source_signal[:, None])
    save_audio_signals(FS, x_t_clean[:, 0], 'clean_signal.wav')
    moving_rir = generate_moving_rir(t_60, radius=0.25, theta_start=0*np.pi/180, theta_end=180*np.pi/180,n_positions= NUM_OF_MOVING_NOISE_POSITIONS)
    # moving_rir = np.load('moving_rir.npy')

    moving_noise = generate_moving_noise(interference_signal, moving_rir)
    save_audio_signals(FS, moving_noise[:, [0, 3]], 'moving_noise.wav')


def generate_moving_noise(noise, moving_rir, num_of_noise_positions=NUM_OF_MOVING_NOISE_POSITIONS):
    noise_split = np.array_split(noise, num_of_noise_positions)
    noise_signal_list = []
    for pos_idx in range(NUM_OF_MOVING_NOISE_POSITIONS):
        n_t = ss.convolve(noise_split[pos_idx][:, None], moving_rir[pos_idx, :])
        n_t = n_t[:len(noise_split[pos_idx]) - 1, :]
        noise_signal_list.append(n_t)

    noise_signal = np.concat(noise_signal_list, axis=0)
    return noise_signal


def main():
    # We concatenate the files to 10s audio files and printing the waveform and the STFT

    target_interference = load_target_and_interference_signals()

    test_prop_relax_online_moving(target_interference)


def run_save_alg(a_f, cur_hop_length, cur_nfft, cur_win_length, filename, k_iters, lmbda, param_ind, rho,
                 test_ind, x_t, x_t_clean, alg_type: str = 'prop-relax'):
    if alg_type == 'prop-relax':
        score_pesq, score_estoi, score_si_sdr, score_dr = run_prop_relax(
            a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters)
    elif alg_type == 'prop-exact':
        score_pesq, score_estoi, score_si_sdr, score_dr = run_prop_exact(
            a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters)
    elif alg_type == 'prop-relax-init':
        score_pesq, score_estoi, score_si_sdr, score_dr = run_prop_relax_with_initialization(
            a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean, k_iters)
    else:
        return pd.DataFrame({})
    cur_stat = pd.DataFrame({
        'param_ind': [param_ind],
        'cur_win_length': [cur_win_length],
        'cur_nfft': [cur_nfft],
        'cur_hop_length': [cur_hop_length],
        'k_iters': [k_iters],
        'test_ind': [test_ind],
        'score_pesq': [score_pesq],
        'score_estoi': [score_estoi],
        'score_si_sdr': [score_si_sdr],
        'score_dr': [score_dr]

    })
    file_exists = os.path.exists(filename)
    cur_stat.to_csv(filename, mode="a", header=not file_exists, index=False)
    return cur_stat


def run_prop_online(a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, x_t):
    n_freq_bins, n_mics = a_f.shape
    F = 2 * (n_freq_bins - 1)
    v_init = np.zeros((n_mics + 1, F), dtype=np.complex128)
    y_f_t_prop_relax = prop_relax_online(x_t, a_f, FS,
                                         win_length=cur_win_length,
                                         hop_length=cur_hop_length,
                                         n_fft=cur_nfft,
                                         v=v_init, rho=rho, lmbda=lmbda, beta=0.7)
    y_t = librosa.istft(y_f_t_prop_relax,
                        win_length=cur_win_length,
                        hop_length=cur_hop_length,
                        n_fft=cur_nfft)

    return y_t


def generate_moving_rir(t_60, radius, theta_start, theta_end, n_positions):
    theta_vec = np.linspace(theta_start, theta_end, n_positions)
    noise_x = CENTER[0] + radius * np.cos(theta_vec)
    noise_y = CENTER[1] + radius * np.sin(theta_vec)
    noise_z = CENTER[2]

    rir_list = []
    for pos_idx in tqdm(range(n_positions)):
        rir = generate_rir(ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, [noise_x[pos_idx], noise_y[pos_idx], noise_z])
        rir_list.append(rir)
    stacked_rir = np.stack(rir_list)

    # np.save("moving_rir",stacked_rir)
    return stacked_rir


def add_noise(interference_signal, t_60, x_t_clean):
    h_interference = create_room_impulse_response(
        ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, INTERFERENCE_POSITION, plot_rir=False)
    u_t = ss.convolve(h_interference, interference_signal[:, None])
    x_t = x_t_clean + u_t
    return x_t


def display_time_frequency_signal(source_signal, cur_hop_length, cur_nfft, cur_win_length, title: str):
    display_audio_time_domain(source_signal, FS,
                              title=f'{title}, r={cur_hop_length}, F={cur_nfft}, N={cur_win_length}')
    display_audio_spectrogram(source_signal, FS, win_length=cur_win_length,
                              hop_length=cur_hop_length,
                              title=f'{title}, r={cur_hop_length}, F={cur_nfft}, N={cur_win_length}',
                              n_fft=cur_nfft)
    plt.show()


def run_prop_relax(a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean,
                   k_iters: int = 10 ** 3, plot_fig5: bool = False):
    n_freq_bins, n_mics = a_f.shape
    F = 2 * (n_freq_bins - 1)
    v_init = np.zeros((n_mics + 1, F), dtype=np.complex128)
    w_f_prop_relax, w_f_time_freq, z_f = prop_relax(
        x_t, a_f, FS, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft,
        v=v_init, rho=rho, lmbda=lmbda, K=k_iters)
    y_out = calc_filtered_signal(w_f_prop_relax, x_t)
    y_out = np.concatenate([np.zeros(F // 2 - 1), y_out[:-F // 2 + 1].real])

    score_pesq = evaluate_pesq_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_estoi = evaluate_estoi_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_si_sdr = evaluate_si_sdr_score(x_t_clean[:, REF_MIC_INDEX], y_out)
    score_dr = evaluate_distortion_ratio(F, a_f, w_f_time_freq)

    if plot_fig5:
        fig5 = create_fig5(x_t_clean[:, REF_MIC_INDEX], y_out)
        fig5.show()

        fig6 = create_fig6(z_f)
        fig6.show()

        save_audio_signals(FS, x_t_clean, f'prop_relax_xt_clean_{test_ind}.wav')
        save_audio_signals(FS, x_t, f'prop_relax_xt_{test_ind}.wav')
        save_audio_signals(FS, y_out, f'prop_relax_out_{test_ind}.wav')
        display_time_frequency_signal(y_out, cur_hop_length, cur_nfft, cur_win_length,
                                      title='Estimated Signal, Prop-Relax')

    # save_audio_signals(FS, y_out, f'prop_relax_out_signal_{test_ind}.wav')
    return score_pesq, score_estoi, score_si_sdr, score_dr


def run_prop_relax_with_initialization(a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t,
                                       x_t_clean, k_iters: int = 10 ** 3, plot_fig5: bool = False):
    n_freq_bins, n_mics = a_f.shape
    F = 2 * (n_freq_bins - 1)

    v_init = np.zeros((n_mics + 1, F), dtype=np.complex128)
    v_init[:n_mics, :] = initialize_with_steering_vec(
        FS, x_t_clean,
        win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft)

    w_f_prop_relax, w_f_time_freq, z_f = prop_relax(
        x_t, a_f, FS, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft,
        v=v_init, rho=rho, lmbda=lmbda, K=k_iters)
    y_out = calc_filtered_signal(w_f_prop_relax, x_t)
    y_out = np.concatenate([np.zeros(F // 2 - 1), y_out[:-F // 2 + 1].real])

    score_pesq = evaluate_pesq_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_estoi = evaluate_estoi_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_si_sdr = evaluate_si_sdr_score(x_t_clean[:, REF_MIC_INDEX], y_out)
    score_dr = evaluate_distortion_ratio(F, a_f, w_f_time_freq)

    if plot_fig5:
        fig5 = create_fig5(x_t_clean[:, REF_MIC_INDEX], y_out)
        fig5.show()

        fig6 = create_fig6(z_f)
        fig6.show()

        save_audio_signals(FS, x_t_clean, f'prop_relax_init_xt_clean_{test_ind}.wav')
        save_audio_signals(FS, x_t, f'prop_relax_init_xt_{test_ind}.wav')
        save_audio_signals(FS, y_out, f'prop_relax_init_out_{test_ind}.wav')
        display_time_frequency_signal(y_out, cur_hop_length, cur_nfft, cur_win_length,
                                      title='Estimated Signal, Prop-Relax with Initialization')
    return score_pesq, score_estoi, score_si_sdr, score_dr


def run_prop_exact(a_f, cur_hop_length, cur_nfft, cur_win_length, lmbda, rho, test_ind, x_t, x_t_clean,
                   k_iters: int = 10 ** 3, plot_fig5: bool = False):
    n_freq_bins, n_mics = a_f.shape
    F = 2 * (n_freq_bins - 1)
    v_init = np.zeros((n_mics + 1, F), dtype=np.complex128)
    w_f_prop_exact, w_f_time_freq, z_f = prop_exact(
        x_t, a_f, FS, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft,
        v=v_init, rho=rho, lmbda=lmbda, K=k_iters)
    y_out = calc_filtered_signal(w_f_prop_exact, x_t)
    y_out = np.concatenate([np.zeros(F // 2 - 1), y_out[:-F // 2 + 1].real])

    score_pesq = evaluate_pesq_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_estoi = evaluate_estoi_score(FS, x_t_clean[:, REF_MIC_INDEX], y_out)
    score_si_sdr = evaluate_si_sdr_score(x_t_clean[:, REF_MIC_INDEX], y_out)
    score_dr = evaluate_distortion_ratio(F, a_f, w_f_time_freq)

    if plot_fig5:
        fig5 = create_fig5(x_t_clean[:, REF_MIC_INDEX], y_out)
        fig5.show()

        fig6 = create_fig6(z_f)
        fig6.show()
        display_time_frequency_signal(y_out, cur_hop_length, cur_nfft, cur_win_length,
                                      title='Estimated Signal, Prop-Exact')

        save_audio_signals(FS, x_t_clean, f'prop_exact_xt_clean_{test_ind}.wav')
        save_audio_signals(FS, x_t, f'prop_exact_xt_{test_ind}.wav')
        save_audio_signals(FS, y_out, f'prop_exact_out_{test_ind}.wav')

    # save_audio_signals(FS, y_out, f'prop_exact_out_signal_{test_ind}.wav')
    return score_pesq, score_estoi, score_si_sdr, score_dr


def load_target_and_interference_signals():
    source_signal = read_wav_files_from_folder(RECORDING_TIME * FS)
    interference_signal = read_wav_files_from_folder(RECORDING_TIME * FS)
    combined_speakers = [[source_signal[i], interference_signal[i]] for i in range(len(source_signal))]
    return combined_speakers


if __name__ == '__main__':
    main()
    plt.show()
