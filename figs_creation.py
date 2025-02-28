import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def create_fig4():
    data_prop_relax = pd.read_csv(r"/Users/ronicaduri/PycharmProjects/audioSignals/PROJECT/performance_prop_relax.csv")
    data_prop_exact = pd.read_csv(r"/Users/ronicaduri/PycharmProjects/audioSignals/PROJECT/performance_prop_exact.csv")
    data_prop_relax_init = pd.read_csv(
        r"/Users/ronicaduri/PycharmProjects/audioSignals/PROJECT/performance_prop_relax_init.csv")
    alg_types = {
        'prop-relax': data_prop_relax, 'prop-exact': data_prop_exact, 'prop-relax-init': data_prop_relax_init
    }

    score_types = ['score_si_sdr', 'score_pesq', 'score_estoi', 'score_dr']
    n_fft_opts = [2048, 4096, 8192, 16384]
    fig = make_subplots(rows=4, cols=4, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=("# of DFT points: 2048", "# of DFT points: 4096",
                                        "# of DFT points: 8192", "# of DFT points: 16384"))
    n_colors = 10
    colors = px.colors.sample_colorscale("turbo", [n / (n_colors - 1) if n_colors != 1 else 0 for n in range(n_colors)])

    for cur_fft_ind in range(len(n_fft_opts)):
        cur_fft = n_fft_opts[cur_fft_ind]

        alg_ind = 0
        for alg_name, alg_data in alg_types.items():
            alg_ind = alg_ind + 1
            cur_data = alg_data.loc[alg_data['cur_nfft'] == cur_fft, :].copy()

            for cur_score_ind in range(len(score_types)):
                if cur_fft_ind == 1 and cur_score_ind == 1:
                    show_legend = True
                else:
                    show_legend = False
                cur_score = score_types[cur_score_ind]
                cur_stat = cur_data.groupby('k_iters')[cur_score].mean().reset_index()
                fig.add_trace(go.Scatter(x=cur_stat['k_iters'], y=cur_stat[cur_score],
                                         name=alg_name, marker_color=colors[3*alg_ind],
                                         showlegend=show_legend),
                              row=(cur_score_ind + 1), col=(cur_fft_ind + 1), )

    fig.update_yaxes(title_text="SI-SDR [dB]", row=1, col=1)
    fig.update_yaxes(title_text="PESQ", row=2, col=1)
    fig.update_yaxes(title_text="ESTOI", row=3, col=1)
    fig.update_yaxes(title_text="DR [dB]", row=4, col=1)
    fig.update_xaxes(title_text="# of iterations", type="log")

    return fig


def create_fig5(ref_mic_signal, deg_signal):
    ref_mic_signal_mean = ref_mic_signal - np.mean(ref_mic_signal)
    deg_signal_complex_mean = deg_signal - np.mean(deg_signal)

    cross_corr = np.correlate(ref_mic_signal_mean, deg_signal_complex_mean, mode="full")
    norm_factor = np.sqrt(np.sum(ref_mic_signal_mean ** 2) * np.sum(deg_signal_complex_mean ** 2))
    cross_corr = cross_corr / norm_factor

    lags = np.arange(-(len(deg_signal) - 1), len(ref_mic_signal))
    df = pd.DataFrame({'cross_corr': cross_corr, 'lags': lags})
    fig = px.line(df, x='lags', y='cross_corr',
                  labels={'lags': 'Lag [sample]',
                          'cross_corr': 'Cross-correlation'})
    fig.add_vline(x=0, line_dash='dash', line_color='black')
    return fig


def create_fig6(z_f):
    prob = 20 * np.log10(1 + z_f)
    fig = go.Figure(data=[go.Histogram(x=prob, histnorm='probability', nbinsx=500)])
    fig.update_layout(yaxis_title_text='Probability',
                      xaxis_title_text='Gain [dB]')
    return fig


if __name__ == '__main__':
    fig4_ = create_fig4()
    fig4_.show()
