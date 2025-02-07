import librosa
import pyroomacoustics as pra
import rir_generator as rir
import matplotlib.pyplot as plt
import os
import scipy.signal as ss
from params import *
from display_funcs import display_audio_spectrogram, display_audio_time_domain, display_rir_time_domain
from scipy.linalg import eigh
import soundfile as sf


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
        display_rir_time_domain(fs, f'Room Impulse Response (first mic). Reverberation Time: {reverb_time} sec',
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


def read_wav_files_from_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            signal, _ = load_input_signal(file_path, plot_signal=False)
            files.append(signal)
    return files


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
    # cov_x_all = []
    for f in range(n_freq_bins):
        cur_s_n_f = s_n_f[:, f, :]
        cov_x_f = (cur_s_n_f @ cur_s_n_f.conj().T) / n_time_frames
        eigvals, eigvecs = eigh(cov_x_f, eigvals_only=False)
        largest_eigenvec = eigvecs[:, -1]
        cur_rtf_f = largest_eigenvec / largest_eigenvec[0]
        rtf_f += [cur_rtf_f]
        # cov_x_all += [cov_x_f]
    rtf_f = np.stack(rtf_f)
    # cov_x_all = np.stack(cov_x_all)
    return rtf_f


def save_audio_signals(fs, signal, filename: str):
    sf.write(filename, signal, fs)


def prox_spatial_filter_complex(v, n_freq_bins, delta):
    F = n_freq_bins + 1
    N_delta = np.arange(-delta, F // 2)
    v_freq = np.fft.ifft(v.conj())
    prox_res = np.zeros_like(v_freq)
    prox_res[N_delta] = v_freq[N_delta]
    res_time_domain = np.fft.fft(prox_res)
    return res_time_domain.conj()


def prox_gain_parameter_real(v):
    return np.maximum(np.real(v), 0)


def prox_quadratic_kkt_inverse(reflect_k_1_f, kkt_inv, rho, d=1):
    n_mics = reflect_k_1_f.shape[0]

    rhs = np.concatenate([rho * reflect_k_1_f, np.array([d])])
    solution = kkt_inv @ rhs
    solution = solution
    x_f = solution[:n_mics]
    return x_f


def precompute_kkt_inverse(extended_scm, alpha, rho):
    n_mics_ext = extended_scm.shape[1]
    n_freq_bin = alpha.shape[1]
    kkt_inv = []
    for f in range(n_freq_bin):
        cur_extended_scm = extended_scm[f, :, :]
        cur_alpha = alpha[:, f].reshape(-1, 1)
        kkt_matrix = np.block([
            [rho * np.eye(n_mics_ext) + cur_extended_scm, cur_alpha],
            [cur_alpha.conj().T, 0]
        ])
        kkt_inv += [np.linalg.inv(kkt_matrix)]
    kkt_inv = np.stack(kkt_inv)
    return kkt_inv


def prop_relax(x_t, a_f,
               fs, win_length, hop_length, n_fft,
               *, v, rho, lmbda, delta=0):
    """
    K = iteration index
    w_f = the spatial filter.
    z_f = frequency-wise gain parameter. non-negative value.
    """
    # convert recorded signal to the time-frequency domain
    n_freq_bins, n_mics, n_time_frames, x_n_f = convert_record_to_time_frequency_domain(
        fs, hop_length, n_fft, win_length, x_t)

    zeta_k_1 = np.zeros_like(v)

    # For Causal MPDR beamformer, ni_f <-- 0 instead of prox(z_f)
    K = 10 ** 3
    print(f">>> K: {K}")
    theta_k_1 = np.zeros_like(v[:n_mics, :])
    ni_k_1 = np.zeros_like(v[n_mics, :]).reshape(1, -1)

    # precompute KKT inverse
    kkt_inv = compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f)
    for k in range(K):
        if k % 1000 == 0:
            print(k)
        # apply the proximity operator for L1 constrain (sparsity) to each frequency bin
        for m in range(n_mics):
            theta_k_1[m, :] = prox_spatial_filter_complex(v[m, :], n_freq_bins, delta)
        for f in range(n_freq_bins):
            ni_k_1[0, f] = prox_gain_parameter_real(v[n_mics, f])
        zeta_k_1[:n_mics, :] = theta_k_1
        zeta_k_1[n_mics, :] = ni_k_1

        for f in range(n_freq_bins):
            # reflection step
            reflect_k_1_f = 2 * zeta_k_1[:, f] - v[:, f]

            # apply the proximity operator for quadratic constraint (power minimization) to each frequency bin
            y_k_1 = prox_quadratic_kkt_inverse(reflect_k_1_f, kkt_inv[f, :, :], rho)

            # update the beam-forming weight matrix
            v[:, f] = v[:, f] + lmbda * (y_k_1 - zeta_k_1[:, f])
    # w_f, z_f = v[:n_mics, :], v[n_mics, :]
    res = []
    for m in range(n_mics):
        cur_theta = prox_spatial_filter_complex(v[m, :], n_freq_bins, delta)
        res += [np.fft.ifft(cur_theta.conj().T)]
    w_f = np.stack(res)
    return w_f, x_n_f


def convert_record_to_time_frequency_domain(fs, hop_length, n_fft, win_length, x_t):
    x_n_f = []
    _, n_mics = x_t.shape
    for cur_mic in range(n_mics):
        x_stft_m = apply_stft(x_t[:, cur_mic], fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
        x_n_f += [x_stft_m]
    x_n_f = np.stack(x_n_f)
    n_freq_bins, n_time_frames = x_stft_m.shape
    return n_freq_bins, n_mics, n_time_frames, x_n_f


def compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f):
    xi_t_f = np.zeros((x_n_f.shape[0] + 1, x_n_f.shape[1], x_n_f.shape[2]), dtype=np.complex128)
    xi_t_f[:n_mics, :, :] = x_n_f  # we extend the STFT of the observed signal
    extended_scm = []
    for f in range(n_freq_bins):
        cur_xi_n_f = xi_t_f[:, f, :]
        cov_x_f = (cur_xi_n_f @ cur_xi_n_f.conj().T) / n_time_frames
        extended_scm += [cov_x_f]
    extended_scm = np.stack(extended_scm)
    alpha_f = np.zeros((a_f.shape[1] + 1, a_f.shape[0]), dtype=np.complex128) - 1
    alpha_f[:n_mics, :] = a_f.T  # we extend the RTF of the target signal
    kkt_inv = precompute_kkt_inverse(extended_scm, alpha_f, rho)
    return kkt_inv


def main():
    # We concatenate the files to 10s audio files and printing the waveform and the STFT
    interference_signal, source_signal = load_target_and_interference_signals()

    cur_win_length, cur_nfft, cur_hop_length = WIN_LENGTH_LIST[2], N_FFT_LIST[2], HOP_LENGTH_LIST[2]
    print(f'>>>>>>>>> cur_win_length={cur_win_length}')
    print(f'>>>>>>>>> cur_nfft={cur_nfft}')
    print(f'>>>>>>>>> cur_hop_length={cur_hop_length}')

    display_audio_time_domain(source_signal, FS, title='original signal')
    display_audio_spectrogram(source_signal, FS, win_length=cur_win_length,
                              hop_length=cur_hop_length,
                              n_fft=cur_nfft)
    plt.show()
    # Create sampled data
    t_60 = np.random.uniform(T_60_RANGE[0], T_60_RANGE[1])
    print(f">>> Randomized T_60: {t_60}[s]")
    h_source = create_room_impulse_response(ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION)
    x_t_clean = ss.convolve(h_source, source_signal[:, None])
    display_audio_time_domain(x_t_clean[:, 0], FS, title='clean target signal - Reference Mic')

    h_interference = create_room_impulse_response(ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, INTERFERENCE_POSITION)
    u_t = ss.convolve(h_interference, interference_signal[:, None])
    x_t = x_t_clean + u_t
    # x_t = x_t_clean
    save_audio_signals(FS, source_signal, 'source_signal.wav')
    save_audio_signals(FS, x_t_clean, 'clean_signal.wav')
    save_audio_signals(FS, x_t, 'noisy_signal.wav')

    a_f = compute_rtf_target(x_t_clean, FS, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft)

    # Through our experiments, we set the initial value of phi_f to zero,
    lmbda = 1.8
    print(f">>> lmbda: {lmbda}")
    rho = 0.005
    print(f">>> rho: {rho}")
    n_freq_bins, n_mics = a_f.shape
    v_init = np.zeros((n_mics + 1, n_freq_bins), dtype=np.complex128)
    w_f_prop_relax, x_n_f = prop_relax(x_t, a_f, FS,
                                       win_length=cur_win_length,
                                       hop_length=cur_hop_length,
                                       n_fft=cur_nfft,
                                       v=v_init, rho=rho, lmbda=lmbda)
    y_stft = np.zeros((n_freq_bins, x_n_f.shape[2]), dtype=np.complex128)
    for m in range(n_mics):
        y_stft += w_f_prop_relax[m, :, None] * x_n_f[m]
    y_out = librosa.istft(y_stft, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft)
    save_audio_signals(FS, y_out, 'out_signal_without_noise.wav')

    print('here')


def load_target_and_interference_signals():
    input_source_signals = read_wav_files_from_folder(TARGET_FOLDER_PATH)
    source_signal = np.concatenate(input_source_signals)
    source_signal = source_signal[:RECORDING_TIME * FS]
    input_interference_signals = read_wav_files_from_folder(INTERFERENCE_FOLDER_PATH)
    interference_signal = np.concatenate(input_interference_signals)
    interference_signal = interference_signal[:RECORDING_TIME * FS]
    return interference_signal, source_signal


if __name__ == '__main__':
    main()
    plt.show()
