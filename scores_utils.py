import numpy as np
from pesq import pesq
from pystoi import stoi


def evaluate_pesq_score(fs, ref_mic_signal, deg_signal_complex):
    deg = np.array([deg_signal_complex.real, deg_signal_complex.imag]).T
    score_pesq = [pesq(fs, ref_mic_signal, deg[:, i], mode='wb') for i in range(deg.shape[1]) if deg[:, i].std() > 0]
    score_pesq = np.array(score_pesq).mean()
    return score_pesq


def evaluate_estoi_score(fs, ref_mic_signal, deg_signal_complex):
    deg = np.array([deg_signal_complex.real, deg_signal_complex.imag]).T
    score_estoi = [stoi(ref_mic_signal, deg[:, i], fs, extended=True) for i in range(deg.shape[1]) if deg[:, i].std() > 0]
    score_estoi = np.array(score_estoi).mean()
    return score_estoi
