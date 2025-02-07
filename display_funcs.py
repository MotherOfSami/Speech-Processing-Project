import matplotlib.pyplot as plt
import librosa
import numpy as np


def display_audio_time_domain(y, sr, title: str = ''):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax, marker='.')
    ax.set_title('Audio Waveform - ' + title)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Amplitude')


def display_audio_spectrogram(y, sr, *, win_length, hop_length, title: str = '', n_fft=None):
    # calculate stft params
    if n_fft is None:
        n_fft = 2 ** (win_length - 1).bit_length()

    # apply stft
    y_stft = librosa.stft(y,
                          win_length=win_length,
                          hop_length=hop_length,
                          n_fft=n_fft
                          )

    # display the magnitude spectrum
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(y_stft), ref=np.max),
        sr=sr, hop_length=hop_length, win_length=win_length, n_fft=n_fft,
        x_axis='time', y_axis='linear', ax=ax)
    ax.set_title(f'Linear-Frequency Spectrogram, {title}')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    # plt.show()


def display_rir_time_domain(fs, h1_title, h_reverberation1, plot_mic_index: int):
    time_ax = np.arange(len(h_reverberation1)) / fs
    fig, ax = plt.subplots()
    plt.plot(time_ax, h_reverberation1[:, plot_mic_index])
    ax.set_title(h1_title)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Amplitude')
