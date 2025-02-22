import librosa
import numpy as np
import rir_generator as rir
import matplotlib.pyplot as plt
import scipy.signal as ss
from params import *
from display_funcs import display_audio_spectrogram, display_audio_time_domain, display_rir_time_domain
from scipy.linalg import eigh
import soundfile as sf
import glob


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
    test_audio_files = np.random.randint(num_speakers, size=(10, 2))
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


def prox_spatial_filter_complex(v, delta):
    F = v.shape[0]
    N_delta = np.arange(-delta, F // 2)
    v_freq = np.fft.ifft(v.conj(), axis=0)
    prox_res = np.zeros_like(v_freq)
    prox_res[N_delta] = v_freq[N_delta]
    res_time_domain = np.fft.fft(prox_res, axis=0)
    return res_time_domain.conj()


def prox_gain_parameter_real(v):
    return np.maximum(np.real(v), 0)


def prox_quadratic_kkt_inverse(reflect_k_1_f, kkt_inv, rho, d=1):
    n_mics = reflect_k_1_f.shape[0]

    rhs = np.concatenate([rho * reflect_k_1_f, np.array([d])])
    solution = kkt_inv @ rhs
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
    K = 400
    print(f">>> K: {K}")
    theta_k_1 = np.zeros_like(v[:n_mics, :])
    ni_k_1 = np.zeros_like(v[n_mics, :]).reshape(1, -1)

    # precompute KKT inverse
    kkt_inv, _ = compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f)
    for k in range(K):
        if k % 100 == 0:
            print(k)
        # apply the proximity operator for L1 constrain (sparsity) to each frequency bin
        for m in range(n_mics):
            theta_k_1[m, :] = prox_spatial_filter_complex(v[m, :], delta)
        for f in range(n_fft):
            ni_k_1[0, f] = prox_gain_parameter_real(v[n_mics, f])
        zeta_k_1[:n_mics, :] = theta_k_1
        zeta_k_1[n_mics, :] = ni_k_1

        for f in range(n_fft):
            # reflection step
            reflect_k_1_f = 2 * zeta_k_1[:, f] - v[:, f]

            # apply the proximity operator for quadratic constraint (power minimization) to each frequency bin
            y_k_1 = prox_quadratic_kkt_inverse(reflect_k_1_f, kkt_inv[f, :, :], rho)

            # update the beam-forming weight matrix
            v[:, f] = v[:, f] + lmbda * (y_k_1 - zeta_k_1[:, f])
    # w_f, z_f = v[:n_mics, :], v[n_mics, :]
    res = []
    for m in range(n_mics):
        cur_theta = prox_spatial_filter_complex(v[m, :], delta)
        res += [np.fft.ifft(cur_theta.conj().T)]
    w_f = np.stack(res)
    return w_f


def prop_relax_online(x_t, a_f,
                      fs, win_length, hop_length, n_fft,
                      *, v, rho, lmbda, delta=0, beta=0.0, K=5):
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
    print(f">>> K: {K}")
    theta_k_1 = np.zeros_like(v[:n_mics, :])
    ni_k_1 = np.zeros_like(v[n_mics, :]).reshape(1, -1)

    v_t_list = []
    scm_mat = np.zeros((n_fft, n_mics + 1, n_mics + 1))
    # precompute KKT inverse
    for t in range(3):
        print(t)
        for k in range(K):
            # apply the proximity operator for L1 constrain (sparsity) to each frequency bin
            for m in range(n_mics):
                theta_k_1[m, :] = prox_spatial_filter_complex(v[m, :], delta)
            for f in range(n_fft):
                ni_k_1[0, f] = prox_gain_parameter_real(v[n_mics, f])
            zeta_k_1[:n_mics, :] = theta_k_1
            zeta_k_1[n_mics, :] = ni_k_1

            kkt_inv, scm_mat = compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, 1, rho,
                                                              x_n_f[:, :, t].reshape((n_mics, n_freq_bins, 1)),
                                                              prev_scm=scm_mat, beta=beta)

            for f in range(n_fft):
                # reflection step
                reflect_k_1_f = 2 * zeta_k_1[:, f] - v[:, f]

                # apply the proximity operator for quadratic constraint (power minimization) to each frequency bin

                y_k_1 = prox_quadratic_kkt_inverse(reflect_k_1_f, kkt_inv[f, :, :], rho)

                # update the beam-forming weight matrix
                v[:, f] = v[:, f] + lmbda * (y_k_1 - zeta_k_1[:, f])
        v_t_list += [v]
    # w_f, z_f = v[:n_mics, :], v[n_mics, :]
    v_t = np.stack(v_t_list)
    res = []
    for m in range(n_mics):
        cur_theta = prox_spatial_filter_complex(v_t[:, m, :].T, delta)
        res += [np.fft.ifft(cur_theta.conj().T)]
    w_f = np.stack(res)
    return w_f


def convert_record_to_time_frequency_domain(fs, hop_length, n_fft, win_length, x_t):
    x_n_f = []
    _, n_mics = x_t.shape
    for cur_mic in range(n_mics):
        x_stft_m = apply_stft(x_t[:, cur_mic], fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
        x_n_f += [x_stft_m]
    x_n_f = np.stack(x_n_f)
    n_freq_bins, n_time_frames = x_stft_m.shape
    return n_freq_bins, n_mics, n_time_frames, x_n_f


def reconstruct_full_spectrum(x_half):
    F_half, n_mics = x_half.shape
    F = (F_half - 1) * 2
    x_full = np.zeros((F, n_mics), dtype=np.complex128)

    x_full[:F_half, :] = x_half[:, :]
    x_full[F_half:, :] = np.conj(x_half[1:-1, :][::-1, :])
    return x_full


def reconstruct_full_scm(scm_half):
    F_half, M, _ = scm_half.shape
    F = (F_half - 1) * 2
    scm_full = np.zeros((F, M, M), dtype=np.complex128)
    scm_full[:F_half, :, :] = scm_half
    scm_full[F_half:, :, :] = np.conj(scm_half[1:-1, :, :][::-1, :, :])
    return scm_full


def compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f, prev_scm=None, beta=0):
    xi_t_f = np.zeros((x_n_f.shape[0] + 1, x_n_f.shape[1], x_n_f.shape[2]), dtype=np.complex128)
    xi_t_f[:n_mics, :, :] = x_n_f  # we extend the STFT of the observed signal
    extended_scm = []
    for f in range(n_freq_bins):
        cur_xi_n_f = xi_t_f[:, f, :]
        cov_x_f = (cur_xi_n_f @ cur_xi_n_f.conj().T) / n_time_frames
        extended_scm += [cov_x_f]
    extended_scm = np.stack(extended_scm)

    scm_full = reconstruct_full_scm(extended_scm)
    if prev_scm is not None:
        scm_full = beta * prev_scm + (1 - beta) * scm_full

    a_f_full = reconstruct_full_spectrum(a_f)
    alpha_f = np.zeros((a_f_full.shape[1] + 1, a_f_full.shape[0]), dtype=np.complex128) - 1
    alpha_f[:n_mics, :] = a_f_full.T  # we extend the RTF of the target signal
    kkt_inv = precompute_kkt_inverse(scm_full, alpha_f, rho)
    return kkt_inv, scm_full


def calc_filtered_signal(w: np.ndarray, sig: np.ndarray):
    num_of_samples, num_of_mics = sig.shape
    res = np.zeros(num_of_samples, dtype=np.complex128)
    sig = sig.astype(np.complex128)
    for mic in range(num_of_mics):
        res += np.convolve(sig[:, mic], w[mic, :], mode='same')
    return res


def main():
    # We concatenate the files to 10s audio files and printing the waveform and the STFT
    target_interference = load_target_and_interference_signals()
    for test_ind in range(len(target_interference)):
        cur_win_length, cur_nfft, cur_hop_length = WIN_LENGTH_LIST[2], N_FFT_LIST[2], HOP_LENGTH_LIST[2]
        print(f'>>>>>>>>> cur_win_length={cur_win_length}')
        print(f'>>>>>>>>> cur_nfft={cur_nfft}')
        print(f'>>>>>>>>> cur_hop_length={cur_hop_length}')

        source_signal = target_interference[test_ind][0]
        interference_signal = target_interference[test_ind][1]
        # display_audio_time_domain(source_signal, FS,
        #                           title=f'original signal, r={cur_hop_length}, F={cur_nfft}, N={cur_win_length}')
        # display_audio_spectrogram(source_signal, FS, win_length=cur_win_length,
        #                           hop_length=cur_hop_length,
        #                           title=f'original signal, r={cur_hop_length}, F={cur_nfft}, N={cur_win_length}',
        #                           n_fft=cur_nfft)
        # plt.show()
        # Create sampled data
        t_60 = np.random.uniform(T_60_RANGE[0], T_60_RANGE[1])
        print(f">>> Randomized T_60: {t_60}[s]")
        h_source = create_room_impulse_response(ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, SOURCE_POSITION,
                                                plot_rir=False)
        x_t_clean = ss.convolve(h_source, source_signal[:, None])
        # display_audio_time_domain(x_t_clean[:, REF_MIC_INDEX], FS, title='clean target signal - Reference Mic')
        # plt.show()
        h_interference = create_room_impulse_response(ROOM_DIMENSIONS, MIC_POSITIONS, FS, t_60, INTERFERENCE_POSITION,
                                                      plot_rir=True)
        # plt.show()
        u_t = ss.convolve(h_interference, interference_signal[:, None])
        x_t = x_t_clean + u_t
        # save_audio_signals(FS, source_signal, f'source_signal_{test_ind}.wav')
        save_audio_signals(FS, x_t_clean, f'clean_signal_{test_ind}.wav')
        save_audio_signals(FS, x_t, f'noisy_signal_{test_ind}.wav')

        a_f = compute_rtf_target(x_t_clean, FS, win_length=cur_win_length, hop_length=cur_hop_length, n_fft=cur_nfft)

        # Through our experiments, we set the initial value of phi_f to zero,
        lmbda = 1.8
        print(f">>> lmbda: {lmbda}")
        rho = 0.005
        print(f">>> rho: {rho}")
        n_freq_bins, n_mics = a_f.shape
        F = 2 * (n_freq_bins - 1)
        v_init = np.zeros((n_mics + 1, F), dtype=np.complex128)
        # w_f_prop_relax = prop_relax_online(x_t, a_f, FS,
        #                                    win_length=cur_win_length,
        #                                    hop_length=cur_hop_length,
        #                                    n_fft=cur_nfft,
        #                                    v=v_init, rho=rho, lmbda=lmbda, beta=0.7)

        w_f_prop_relax = prop_relax(x_t, a_f, FS,
                                    win_length=cur_win_length,
                                    hop_length=cur_hop_length,
                                    n_fft=cur_nfft,
                                    v=v_init, rho=rho, lmbda=lmbda)
        y_out = calc_filtered_signal(w_f_prop_relax, x_t)
        save_audio_signals(FS, np.array([y_out.real, y_out.imag]).T, f'out_signal_{test_ind}.wav')


def load_target_and_interference_signals():
    source_signal = read_wav_files_from_folder(RECORDING_TIME * FS)
    interference_signal = read_wav_files_from_folder(RECORDING_TIME * FS)
    combined_speakers = [[source_signal[i], interference_signal[i]] for i in range(len(source_signal))]
    return combined_speakers


if __name__ == '__main__':
    main()
    plt.show()
