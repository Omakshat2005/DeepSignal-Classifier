"""
RF Signal Intelligence Dashboard
Minimal, warm-toned Streamlit UI for LSTM modulation classification.
"""

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import gc

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF Signal Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Source+Code+Pro:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:          #f5f0e8;
    --surface:     #faf7f2;
    --border:      #e0d8cc;
    --text-dark:   #2c2416;
    --text-mid:    #5c4f3a;
    --text-light:  #9c8c78;
    --accent:      #b85c38;
    --teal:        #3a7c6e;
    --sand:        #c8a96e;
    --sand-soft:   #e8d4a8;
    --success:     #4a7c59;
    --error:       #9c3a28;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text-dark) !important;
    font-family: 'Outfit', sans-serif !important;
}

.page-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-dark);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
}
.page-tagline {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.72rem;
    color: var(--text-light);
    margin-top: 0.4rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.section-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem 0.9rem;
}
.metric-card-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 0.35rem;
}
.metric-card-value {
    font-family: 'Lora', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.1;
}
.metric-card-value.accent { color: var(--accent); }
.metric-card-value.teal   { color: var(--teal); }
.metric-card-value.sand   { color: var(--sand); }

.badge-correct {
    display: inline-block;
    background: #e8f3ec;
    color: var(--success);
    border: 1px solid #b8d8c4;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    letter-spacing: 0.06em;
    margin-top: 0.35rem;
}
.badge-wrong {
    display: inline-block;
    background: #f5e8e6;
    color: var(--error);
    border: 1px solid #d8b8b4;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    letter-spacing: 0.06em;
    margin-top: 0.35rem;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-mid) !important;
    font-size: 0.88rem !important;
}

.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text-dark) !important;
    border-radius: 6px !important;
}
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--sand);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    font-size: 0.8rem;
    color: var(--text-mid);
    font-family: 'Source Code Pro', monospace;
    line-height: 2;
}
.info-box b { color: var(--text-dark); font-weight: 600; }

.noise-warning {
    background: #fdf3ee;
    border: 1px solid #e8c4a8;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: var(--accent);
    margin-top: 0.5rem;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-family: 'Source Code Pro', monospace;
    font-size: 0.62rem;
    color: var(--text-light);
    letter-spacing: 0.15em;
    text-align: center;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
BG       = "#f5f0e8"
SURFACE  = "#faf7f2"
BORDER   = "#ddd4c4"
TEXT_D   = "#2c2416"
TEXT_M   = "#5c4f3a"
TEXT_L   = "#9c8c78"
ACCENT   = "#b85c38"
TEAL     = "#3a7c6e"
TEAL_S   = "#7abcae"
SAND     = "#c8a96e"
SAND_S   = "#e8d4a8"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_M,
    "xtick.color":       TEXT_L,
    "ytick.color":       TEXT_L,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "text.color":        TEXT_D,
    "grid.color":        BORDER,
    "grid.linewidth":    0.55,
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
# Use this EXACT list in both files
classes = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'WBFM', 'AM-SSB', 'AM-DSB']
# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_data():
    with open('Rml.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    mod_to_id = {name: i for i, name in enumerate(classes)}
    id_to_mod = {i: name for name, i in mod_to_id.items()}
    X, lbl = [], []
    for k, v in data.items():
        X.append(v)
        for _ in range(v.shape[0]):
            lbl.append(k)

    X = np.vstack(X)
    X = np.transpose(X, (0, 2, 1)).astype('float32')
    mods_labels = np.array([mod_to_id[l[0]] for l in lbl])
    snr_labels  = np.array([l[1] for l in lbl])

    Y = to_categorical(mods_labels)
    _, X_test, _, Y_test, _, snr_test = train_test_split(
        X, Y, snr_labels, test_size=0.2, random_state=42
    )
    del X, Y, mods_labels
    gc.collect()
    return X_test, Y_test, snr_test, id_to_mod, mod_to_id

@st.cache_resource(show_spinner=False)
def load_classifier():
    return load_model('mod_classifier_lstm.h5')

# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.tick_params(colors=TEXT_L, length=3)
    ax.grid(True, alpha=0.45, linestyle='--', linewidth=0.5)

def add_awgn(signal, snr_db):
    if snr_db >= 40:
        return signal
    power = np.mean(signal ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), signal.shape)

def plot_time_domain(ax, signal):
    t = np.arange(signal.shape[0])
    ax.plot(t, signal[:, 0], color=ACCENT, lw=1.4, alpha=0.9,  label='I — In-Phase')
    ax.plot(t, signal[:, 1], color=TEAL,   lw=1.4, alpha=0.85, label='Q — Quadrature')
    ax.fill_between(t, signal[:, 0], alpha=0.07, color=ACCENT)
    ax.fill_between(t, signal[:, 1], alpha=0.07, color=TEAL)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.set_xlabel("Sample Index", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_ylabel("Amplitude", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_title("Time Domain  ·  I & Q Waveforms",
                 fontsize=9.5, fontweight='bold', color=TEXT_D, pad=10)
    ax.legend(fontsize=7.5, loc='upper right',
              facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_M, framealpha=0.9)
    ax.set_xlim(0, len(t) - 1)
    style_ax(ax)

def plot_constellation(ax, signal):
    I, Q = signal[:, 0], signal[:, 1]
    h, xe, ye = np.histogram2d(I, Q, bins=55)
    ax.imshow(h.T, extent=[xe[0], xe[-1], ye[0], ye[-1]],
              origin='lower', cmap='YlOrBr', alpha=0.3,
              aspect='auto', interpolation='bilinear')
    ax.scatter(I, Q, c=TEAL, s=4, alpha=0.35, linewidths=0)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.axvline(0, color=BORDER, lw=0.8)
    ax.set_xlabel("In-Phase (I)", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_ylabel("Quadrature (Q)", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_title("Constellation Diagram",
                 fontsize=9.5, fontweight='bold', color=TEXT_D, pad=10)
    ax.set_aspect('equal', adjustable='box')
    style_ax(ax)

def plot_psd(ax, signal):
    cs = signal[:, 0] + 1j * signal[:, 1]
    psd = np.abs(np.fft.fftshift(np.fft.fft(cs, n=512))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(512))
    psd_db = 10 * np.log10(psd + 1e-12)
    ax.plot(freq, psd_db, color=SAND, lw=1.4, alpha=0.95)
    ax.fill_between(freq, psd_db, psd_db.min() - 1, alpha=0.15, color=SAND)
    ax.set_xlabel("Normalised Frequency", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_ylabel("Power (dB)", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_title("Power Spectral Density",
                 fontsize=9.5, fontweight='bold', color=TEXT_D, pad=10)
    ax.set_xlim(-0.5, 0.5)
    style_ax(ax)

def plot_confidence(ax, probs, id_to_mod):
    idx    = np.argsort(probs)[::-1]
    labels = [id_to_mod[i] for i in idx]
    values = probs[idx]

    bar_colors = [ACCENT if i == 0 else (SAND_S if v > 0.05 else BORDER)
                  for i, v in enumerate(values)]

    bars = ax.barh(labels[::-1], values[::-1],
                   color=bar_colors[::-1], height=0.55, edgecolor='none')

    for bar, val in zip(bars, values[::-1]):
        if val > 0.01:
            ax.text(val + 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val * 100:.1f}%',
                    va='center', ha='left',
                    fontsize=7.5, color=TEXT_M, fontfamily='monospace')

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Confidence", fontsize=8, color=TEXT_M, labelpad=5)
    ax.set_title("Classification Confidence",
                 fontsize=9.5, fontweight='bold', color=TEXT_D, pad=10)
    ax.tick_params(axis='y', labelsize=8, colors=TEXT_M)
    style_ax(ax)

# ─── APP ──────────────────────────────────────────────────────────────────────
def main():

    st.markdown("""
    <div class="page-header">
        <div class="page-title">RF Signal Intelligence</div>
        <div class="page-tagline">LSTM Modulation Classifier &nbsp;·&nbsp; RadioML 2016.10A</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data and model…"):
        try:
            X_test, Y_test, snr_test, id_to_mod, mod_to_id = load_data()
            model = load_classifier()
        except FileNotFoundError as e:
            st.error(f"File not found: {e}\n\nPlace `Rml.pkl` and `mod_classifier_lstm.h5` in the same directory.")
            return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Signal Parameters")

        snr_options  = sorted(np.unique(snr_test).tolist())
        selected_snr = st.select_slider(
            "SNR Level (dB)", options=snr_options,
            value=snr_options[len(snr_options) // 2]
        )

        snr_mask  = snr_test == selected_snr
        filtered  = np.where(snr_mask)[0]
        true_ids  = np.argmax(Y_test[filtered], axis=1)
        avail     = ["All"] + sorted(set(id_to_mod[i] for i in true_ids))
        mod_filter = st.selectbox("Filter by True Modulation", options=avail)

        if mod_filter != "All":
            filtered = filtered[true_ids == mod_to_id[mod_filter]]

        n = len(filtered)
        st.caption(f"{n} sample{'s' if n != 1 else ''} match this filter")

        if n == 0:
            st.warning("No samples match. Adjust the filters.")
            return
        if n > 1:
            sample_idx = st.slider("Sample Index", 0, n - 1, 0)
        else:
            # If only 1 sample exists, set index to 0 automatically
            sample_idx = 0
            st.info("Showing the only available sample for this filter.")
        sample_idx = st.slider("Sample Index", 0, n - 1, 0)

        st.markdown("---")
        st.markdown("### Noise Injection")
        extra_noise = st.slider(
            "Extra AWGN (dB)", min_value=-20, max_value=40, value=40, step=2,
            help="40 dB = no extra noise. Lower values degrade the signal."
        )
        if extra_noise < 40:
            st.markdown(
                f'<div class="noise-warning">⚠ Injecting noise at {extra_noise} dB</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            f"""<div class="info-box">
            <b>Model</b> &nbsp;&nbsp;&nbsp;&nbsp;Stacked LSTM × 2<br>
            <b>Classes</b> &nbsp;&nbsp;{len(id_to_mod)}<br>
            <b>Test set</b> &nbsp;{len(X_test):,} samples<br>
            <b>SNR range</b> −18 → 20 dB
            </div>""",
            unsafe_allow_html=True
        )

    # ── Sample ────────────────────────────────────────────────────────────────
    global_idx  = filtered[sample_idx]
    raw_signal  = X_test[global_idx]
    true_label  = id_to_mod[np.argmax(Y_test[global_idx])]
    sample_snr  = snr_test[global_idx]
    signal      = add_awgn(raw_signal.copy(), extra_noise)

    probs      = model.predict(signal[np.newaxis, ...], verbose=0)[0]
    pred_id    = int(np.argmax(probs))
    pred_label = id_to_mod[pred_id]
    confidence = float(probs[pred_id])
    correct    = pred_label == true_label

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-label">Prediction</div>
            <div class="metric-card-value accent">{pred_label}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-label">Confidence</div>
            <div class="metric-card-value">{confidence * 100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-label">True Label</div>
            <div class="metric-card-value teal">{true_label}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        badge = '<span class="badge-correct">✓ Correct</span>' if correct else '<span class="badge-wrong">✗ Incorrect</span>'
        st.markdown(f"""<div class="metric-card">
            <div class="metric-card-label">SNR Level</div>
            <div class="metric-card-value sand">{sample_snr} dB</div>
            {badge}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Signal Analysis</div>', unsafe_allow_html=True)
        fig = plt.figure(figsize=(9, 6.8), facecolor=BG)
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.4)
        plot_time_domain(fig.add_subplot(gs[0, :]), signal)
        plot_constellation(fig.add_subplot(gs[1, 0]), signal)
        plot_psd(fig.add_subplot(gs[1, 1]), signal)
        fig.tight_layout(pad=1.8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown('<div class="section-label">Classifier Output</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4.8, 6.8), facecolor=BG)
        plot_confidence(ax2, probs, id_to_mod)
        fig2.tight_layout(pad=1.8)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown(
        '<div class="footer">RF Signal Intelligence &nbsp;·&nbsp; RadioML 2016.10A &nbsp;·&nbsp; LSTM Classifier</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
