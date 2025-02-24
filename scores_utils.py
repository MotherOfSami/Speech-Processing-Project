import numpy as np
from pesq import pesq
from pystoi import stoi

def evaluate_pesq_score(fs, ref_mic_signal, deg_signal_complex):
    deg_signal_real = deg_signal_complex.real  # the complex part is an artifact of the processing
    score_pesq = pesq(fs, ref_mic_signal, deg_signal_real, mode='wb')
    return score_pesq


def evaluate_estoi_score(fs, ref_mic_signal, deg_signal_complex):
    deg_signal_real = deg_signal_complex.real  # the complex part is an artifact of the processing
    score_estoi = stoi(ref_mic_signal, deg_signal_real, fs, extended=True)
    return score_estoi


def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s) / np.linalg.norm(s) ** 2
    sdr = 10 * np.log10(np.linalg.norm(alpha * s) ** 2 / np.linalg.norm(
        alpha * s - s_hat) ** 2)
    return sdr


def evaluate_si_sdr_score(ref_mic_signal, deg_signal_complex):
    deg_signal_real = deg_signal_complex.real  # the complex part is an artifact of the processing
    score_si_sdr = si_sdr(ref_mic_signal, deg_signal_real)
    return score_si_sdr


