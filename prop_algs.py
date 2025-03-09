import librosa
import numpy as np


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


def compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f, prev_scm=None, beta=0.0):
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


def convert_record_to_time_frequency_domain(fs, hop_length, n_fft, win_length, x_t):
    x_n_f = []
    _, n_mics = x_t.shape
    for cur_mic in range(n_mics):
        x_stft_m = apply_stft(x_t[:, cur_mic], fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
        x_n_f += [x_stft_m]
    x_n_f = np.stack(x_n_f)
    n_freq_bins, n_time_frames = x_stft_m.shape
    return n_freq_bins, n_mics, n_time_frames, x_n_f


def prop_exact(x_t, a_f,
               fs, win_length, hop_length, n_fft,
               *, v, rho, lmbda, delta=0, K: int = 10 ** 3):
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
    print(f">>> K: {K}, n_fft: {n_fft}")
    theta_k_1 = np.zeros_like(v[:n_mics, :])
    ni_k_1 = np.zeros_like(v[n_mics, :]).reshape(1, -1)

    # precompute KKT inverse
    kkt_inv, _ = compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f)

    for k in range(K):
        if k % 1000 == 0:
            print(k)
        # apply the proximity operator for L1 constrain (sparsity) to each frequency bin
        for m in range(n_mics):
            theta_k_1[m, :] = prox_spatial_filter_complex(v[m, :], delta)

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

    w_f_time_freq = np.zeros_like(v[:n_mics, :])
    for m in range(n_mics):
        w_f_time_freq[m, :] = prox_spatial_filter_complex(v[m, :], delta)
    z_f = np.zeros_like(v[n_mics, :], dtype='float64')
    for f in range(n_fft):
        z_f[f] = prox_gain_parameter_real(v[n_mics, f])
    return w_f, w_f_time_freq, z_f


def prop_relax(x_t, a_f,
               fs, win_length, hop_length, n_fft,
               *, v, rho, lmbda, delta=0, K: int = 10 ** 3):
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
    print(f">>> K: {K}, n_fft: {n_fft}")
    theta_k_1 = np.zeros_like(v[:n_mics, :])
    ni_k_1 = np.zeros_like(v[n_mics, :]).reshape(1, -1)

    # precompute KKT inverse
    kkt_inv, _ = compute_kkt_inverse_in_advance(a_f, n_freq_bins, n_mics, n_time_frames, rho, x_n_f)

    for k in range(K):
        if k % 1000 == 0:
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

    w_f_time_freq = np.zeros_like(v[:n_mics, :])
    for m in range(n_mics):
        w_f_time_freq[m, :] = prox_spatial_filter_complex(v[m, :], delta)
    z_f = np.zeros_like(v[n_mics, :], dtype='float64')
    for f in range(n_fft):
        z_f[f] = prox_gain_parameter_real(v[n_mics, f])
    return w_f, w_f_time_freq, z_f


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
    for t in range(n_time_frames):
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
    v_t_causal = prox_spatial_filter_complex(v_t.T, delta)

    y_n_f = np.zeros((n_fft, n_time_frames), dtype=np.complex128)
    for m in range(n_mics):
        full_x_n_f = reconstruct_full_spectrum(x_n_f[m, :])
        y_n_f += v_t_causal[:, m, :].conj() * full_x_n_f

    return y_n_f
