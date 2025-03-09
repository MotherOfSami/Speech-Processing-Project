import numpy as np
from params import *
from prop_algs import apply_stft


def apply_srp_phat(fs, x_t, *, win_length, hop_length, n_fft):
    # STEP 1: build spatial grid of possible source positions within the region of interest
    x = np.linspace(SOURCE_MARGIN_X[0], SOURCE_MARGIN_X[1], Nx)
    y = np.linspace(SOURCE_MARGIN_Y[0], SOURCE_MARGIN_Y[1], Ny)
    xv, yv = np.meshgrid(x, y)
    # for each candidate location calculate the set of time delays for signals to travel from that location to each
    # microphone in the array
    n_mics = len(MIC_POSITIONS)
    # STEP 2: Compute GCC-PHAT
    n_pairs = (n_mics * (n_mics - 1)) // 2
    gcc_phat_grid = np.zeros((Ny, Nx, n_pairs))
    pair_ind = 0
    mic_pairs_to_ind = {}
    for m1 in range(1, n_mics + 1):
        for m2 in range(m1 + 1, n_mics + 1):
            # for each candidate location calculate the set of time delays for signals to travel from that location
            # to each microphone in the array
            m1_pos = MIC_POSITIONS[m1 - 1]
            m2_pos = MIC_POSITIONS[m2 - 1]
            d1 = np.sqrt((xv - m1_pos[0]) ** 2 + (yv - m1_pos[1]) ** 2)
            d2 = np.sqrt((xv - m2_pos[0]) ** 2 + (yv - m2_pos[1]) ** 2)
            time_delay = (d1 - d2) / C

            # ======== Compute GCC-PHAT ========
            x_stft_m1 = apply_stft(x_t[:, m1 - 1], FS, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
            x_stft_m2 = apply_stft(x_t[:, m2 - 1], FS, win_length=win_length, hop_length=hop_length, n_fft=n_fft)

            n_freq_bins, n_time_frames = x_stft_m1.shape
            # compute cross correlation for each frame and frequency bin
            cps = x_stft_m1 * x_stft_m2.conj()
            cps_phat = cps / np.abs(cps)  # magnitude normalization to emphasize phase information

            # calculate the time-domain GCC-PHAT for each frame
            n_fft = 2 * n_freq_bins - 2
            freq_grid = np.tile(np.arange(n_freq_bins).reshape((n_freq_bins, 1, 1, 1)),
                                (1, n_time_frames, Ny, Nx))
            time_grid = np.tile(time_delay.reshape((1, 1, Ny, Nx)),
                                (n_freq_bins, n_time_frames, 1, 1))
            exp_part = np.exp(2j * np.pi * (fs / n_fft) * freq_grid * time_grid)
            gcc_phat = exp_part * np.tile(
                cps_phat.reshape(n_freq_bins, n_time_frames, 1, 1),
                (1, 1, Ny, Nx))
            gcc_phat = np.sum(gcc_phat, axis=0)

            # aggregate across frames, assuming static environment
            gcc_phat = np.sum(gcc_phat, axis=0)
            gcc_phat_grid[:, :, pair_ind] = np.abs(gcc_phat)
            mic_pairs_to_ind[(m1, m2)] = pair_ind
            pair_ind = pair_ind + 1
    # for each candidate location p, calculate the SRP-PHAT
    srp = np.sum(gcc_phat_grid, axis=2)
    ind = np.unravel_index(np.argmax(srp, axis=None), srp.shape)
    estimated_source_position = xv[ind], yv[ind]
    return estimated_source_position, srp, xv, yv


def initialize_with_steering_vec(fs, x_t, win_length, hop_length, n_fft):
    estimated_source_position, _, _, _ = apply_srp_phat(
        fs, x_t, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    d = np.sqrt(
        (estimated_source_position[0] - np.array(MIC_POSITIONS)[:, 0]) ** 2 + (
                estimated_source_position[1] - np.array(MIC_POSITIONS)[:, 1]) ** 2)
    time_delay = d / C
    exp_part = np.exp(-2j * np.pi * (fs / n_fft) * (np.arange(n_fft).reshape(-1, 1) @ time_delay.reshape(1, -1)))
    exp_part = exp_part.T

    w_init = exp_part/np.sum(exp_part.conj() * exp_part, axis=0, keepdims=True)
    return w_init
