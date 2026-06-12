"""
Penicillin Fermentation Optimizer - Streamlit Web Application
Final Year Project - Muhammad Luqman Bin Abd Latif
Universiti Kuala Lumpur (MICET), 2026

A machine learning decision-support tool for penicillin fed-batch fermentation,
trained on the real IndPenSim industrial benchmark dataset (Goldrick et al., 2019).
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Penicillin Fermentation Optimizer",
    page_icon="🧫",
    layout="wide",
)

FEATURES = ['temp_C', 'pH', 'DO_mgL', 'substrate_gL',
            'feed_rate_Lh', 'aeration_Lh', 'biomass_gL', 'duration_h']
LABELS = {
    'temp_C': 'Temperature (°C)', 'pH': 'pH', 'DO_mgL': 'Dissolved O₂ (mg/L)',
    'substrate_gL': 'Substrate (g/L)', 'feed_rate_Lh': 'Feed Rate (L/h)',
    'aeration_Lh': 'Aeration (L/h)', 'biomass_gL': 'Biomass (g/L)',
    'duration_h': 'Duration (h)',
}
TARGET = 'penicillin_gL'

# Column aliases for CSV auto-cleaning
ALIASES = {
    'temp_C': ['temp', 'temperature', 'temp_c', 't', 'temp_k', 'temperature_k'],
    'pH': ['ph', 'ph_value', 'acidity'],
    'DO_mgL': ['do', 'do_mgl', 'dissolved_oxygen', 'do2', 'o2', 'oxygen', 'dissolved_o2'],
    'substrate_gL': ['substrate', 'glucose', 'sugar', 's', 'carbon', 'substrate_concentration'],
    'feed_rate_Lh': ['feed', 'feed_rate', 'feedrate', 'fs', 'sugar_feed'],
    'aeration_Lh': ['aeration', 'air', 'airflow', 'fg', 'aeration_rate'],
    'biomass_gL': ['biomass', 'cells', 'cell', 'x', 'cell_concentration'],
    'duration_h': ['duration', 'time', 'time_h', 'batch_time', 'hours', 'total_time'],
}
TARGET_ALIASES = ['penicillin', 'penicillin_gl', 'yield', 'p', 'product',
                  'penicillin_concentration', 'titre', 'titer']


# ----------------------------------------------------------------------------
# Data loading and model training (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_base_data():
    """Load the bundled IndPenSim dataset."""
    df = pd.read_csv("IndPenSim_100_batches.csv")
    return df[FEATURES + [TARGET]].copy()


@st.cache_resource
def train_model(data_hash, X_values, y_values):
    """Train Random Forest. Cached on a hash of the data so re-training only
    happens when the dataset actually changes."""
    X = pd.DataFrame(X_values, columns=FEATURES)
    y = pd.Series(y_values)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        'r2': r2_score(y_te, pred),
        'rmse': np.sqrt(mean_squared_error(y_te, pred)),
        'mae': mean_absolute_error(y_te, pred),
        'y_test': y_te.values, 'y_pred': pred,
    }
    perm = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=42)
    metrics['perm_importance'] = perm.importances_mean
    return model, metrics


def normalize_key(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())


def clean_uploaded_csv(df_raw):
    """Auto-clean an uploaded CSV: match columns, convert Kelvin, fill gaps."""
    report = []
    headers = list(df_raw.columns)
    norm_headers = {normalize_key(h): h for h in headers}

    col_map = {}
    for canon in FEATURES:
        found = None
        if normalize_key(canon) in norm_headers:
            found = norm_headers[normalize_key(canon)]
        else:
            for alias in ALIASES[canon]:
                if normalize_key(alias) in norm_headers:
                    found = norm_headers[normalize_key(alias)]
                    break
        if found:
            col_map[canon] = found
            report.append(("ok", f"✓ {LABELS[canon]} ← \"{found}\""))
        else:
            report.append(("warn", f"⚠ {LABELS[canon]}: not found — will use median"))

    target_col = None
    for alias in TARGET_ALIASES:
        if normalize_key(alias) in norm_headers:
            target_col = norm_headers[normalize_key(alias)]
            break
    if target_col is None:
        return None, [("err", "✗ No penicillin/yield column found. File must include measured yield.")]
    report.append(("ok", f"✓ Penicillin yield ← \"{target_col}\""))

    base = load_base_data()
    medians = base[FEATURES].median()

    # Kelvin detection
    kelvin = False
    if 'temp_C' in col_map:
        sample = pd.to_numeric(df_raw[col_map['temp_C']], errors='coerce').dropna()
        if len(sample) and sample.mean() > 200:
            kelvin = True
            report.append(("warn", "⚠ Temperature looks like Kelvin (avg > 200) — auto-converting to °C"))

    rows, skipped, filled = [], 0, 0
    targets = []
    for _, r in df_raw.iterrows():
        ty = pd.to_numeric(r.get(target_col), errors='coerce')
        if pd.isna(ty) or ty < 0:
            skipped += 1
            continue
        row = {}
        for canon in FEATURES:
            if canon in col_map:
                v = pd.to_numeric(r.get(col_map[canon]), errors='coerce')
                if pd.isna(v):
                    v = medians[canon]
                    filled += 1
            else:
                v = medians[canon]
            if canon == 'temp_C' and kelvin:
                v = v - 273.15
            row[canon] = v
        rows.append(row)
        targets.append(ty)

    if not rows:
        return None, report + [("err", "✗ No valid rows after cleaning.")]

    clean_df = pd.DataFrame(rows)
    clean_df[TARGET] = targets
    msg = f"Cleaned: {len(rows)} valid rows kept"
    if skipped:
        msg += f", {skipped} skipped (bad/blank yield)"
    if filled:
        msg += f", {filled} missing cells filled with medians"
    report.append(("ok", msg))
    return clean_df, report


def optimize(model, ranges, locked_values):
    """Two-stage search for maximum predicted yield, respecting locked vars."""
    rng = np.random.RandomState(0)
    free = [f for f in FEATURES if f not in locked_values]
    # Stage 1: random sweep
    N = 40000
    samp = {}
    for f in FEATURES:
        if f in locked_values:
            samp[f] = np.full(N, locked_values[f])
        else:
            samp[f] = rng.uniform(ranges[f][0], ranges[f][1], N)
    S = pd.DataFrame(samp)[FEATURES]
    preds = model.predict(S)
    best_idx = preds.argmax()
    best_x = S.iloc[best_idx].to_dict()
    best_y = preds[best_idx]
    # Stage 2: coordinate refinement
    for _ in range(3):
        for f in free:
            grid = np.linspace(ranges[f][0], ranges[f][1], 60)
            trial = pd.DataFrame([best_x] * 60)[FEATURES]
            trial[f] = grid
            tp = model.predict(trial)
            if tp.max() > best_y:
                best_y = tp.max()
                best_x[f] = grid[tp.argmax()]
    return best_x, best_y


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if 'data' not in st.session_state:
    st.session_state.data = load_base_data()
    st.session_state.added = 0

data = st.session_state.data
ranges = {f: (float(data[f].min()), float(data[f].max())) for f in FEATURES}
medians = {f: float(data[f].median()) for f in FEATURES}

data_hash = hash(pd.util.hash_pandas_object(data).sum())
model, metrics = train_model(data_hash,
                             data[FEATURES].values.tolist(),
                             data[TARGET].values.tolist())

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧫 Penicillin Fermentation Optimizer")
st.caption("Machine Learning Decision-Support Tool · Real IndPenSim Benchmark "
           "(Goldrick et al., 2019) · Random Forest Regression")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test R²", f"{metrics['r2']:.3f}")
m2.metric("RMSE (g/L)", f"{metrics['rmse']:.2f}")
m3.metric("Batches", f"{len(data)}" + (f" (+{st.session_state.added})" if st.session_state.added else ""))
m4.metric("Variables", "8")
m5.metric("Mean Yield", f"{data[TARGET].mean():.1f} g/L")

st.divider()

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🎛️ Process Conditions")
    st.caption("Set each variable. Tick **Lock** to fix it during optimisation.")
    inputs, locks = {}, {}
    for f in FEATURES:
        c1, c2 = st.columns([4, 1])
        with c1:
            inputs[f] = st.slider(LABELS[f], ranges[f][0], ranges[f][1],
                                  medians[f], key=f"sl_{f}")
        with c2:
            st.write("")
            st.write("")
            locks[f] = st.checkbox("🔒", key=f"lock_{f}", help=f"Lock {LABELS[f]}")

with right:
    st.subheader("📊 Predicted Yield")
    x = pd.DataFrame([inputs])[FEATURES]
    pred = model.predict(x)[0]
    dmean = data[TARGET].mean()
    dmax = data[TARGET].max()

    st.markdown(f"<h1 style='color:#0D9488;margin:0'>{pred:.2f} "
                f"<span style='font-size:24px;color:#888'>g/L</span></h1>",
                unsafe_allow_html=True)
    st.progress(min(1.0, pred / dmax))
    vs = (pred / dmean - 1) * 100
    st.caption(f"{'+' if vs >= 0 else ''}{vs:.0f}% vs dataset mean ({dmean:.1f} g/L) · "
               f"{pred/dmax*100:.0f}% of max observed")

    st.write("")
    if st.button("⚡ Find Maximum Yield Conditions", type="primary", use_container_width=True):
        locked_vals = {f: inputs[f] for f in FEATURES if locks[f]}
        with st.spinner("Searching 40,000+ combinations..."):
            best_x, best_y = optimize(model, ranges, locked_vals)
        st.session_state.opt = (best_x, best_y, locked_vals)

    if 'opt' in st.session_state:
        best_x, best_y, locked_vals = st.session_state.opt
        n_lock = len(locked_vals)
        note = (f"Searched with {n_lock} parameter(s) held fixed."
                if n_lock else "Searched the full feasible operating space.")
        st.caption(note)
        opt_df = pd.DataFrame({
            "Variable": [LABELS[f] + (" 🔒" if f in locked_vals else "") for f in FEATURES],
            "Optimal value": [f"{best_x[f]:.2f}" for f in FEATURES],
        })
        st.dataframe(opt_df, hide_index=True, use_container_width=True)
        st.success(f"**Maximum predicted yield: {best_y:.2f} g/L**")

st.divider()

# ----------------------------------------------------------------------------
# Add data + importance/accuracy
# ----------------------------------------------------------------------------
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("➕ Add Data & Re-train")
    st.caption("Add batches to update the model live. One batch barely shifts a "
               "100-batch model — that stability is expected. Add several to see change.")

    tab1, tab2 = st.tabs(["Type one batch", "Upload CSV"])

    with tab1:
        with st.form("add_batch", clear_on_submit=False):
            cols = st.columns(2)
            new_vals = {}
            for i, f in enumerate(FEATURES):
                with cols[i % 2]:
                    new_vals[f] = st.number_input(LABELS[f], value=medians[f],
                                                  key=f"nb_{f}", format="%.2f")
            new_y = st.number_input("Measured Penicillin Yield (g/L)",
                                    value=30.0, format="%.2f")
            submitted = st.form_submit_button("🔄 Add Batch & Re-train",
                                              type="primary", use_container_width=True)
            if submitted:
                new_row = {**new_vals, TARGET: new_y}
                st.session_state.data = pd.concat(
                    [st.session_state.data, pd.DataFrame([new_row])],
                    ignore_index=True)
                st.session_state.added += 1
                st.rerun()

    with tab2:
        st.download_button(
            "⬇ Download blank CSV template",
            data=",".join(FEATURES + [TARGET]) + "\n" +
                 ",".join([f"{medians[f]:.2f}" for f in FEATURES] + ["30.0"]) + "\n",
            file_name="batch_template.csv", mime="text/csv")
        up = st.file_uploader("Upload a CSV of batches", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                clean_df, report = clean_uploaded_csv(raw)
                for kind, msg in report:
                    if kind == "ok":
                        st.success(msg, icon="✅")
                    elif kind == "warn":
                        st.warning(msg, icon="⚠️")
                    else:
                        st.error(msg, icon="🚫")
                if clean_df is not None:
                    if st.button(f"🔄 Import {len(clean_df)} batches & Re-train",
                                 type="primary", use_container_width=True):
                        st.session_state.data = pd.concat(
                            [st.session_state.data, clean_df], ignore_index=True)
                        st.session_state.added += len(clean_df)
                        st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")

    if st.session_state.added > 0:
        if st.button("↺ Reset to original 100 batches", use_container_width=True):
            st.session_state.data = load_base_data()
            st.session_state.added = 0
            if 'opt' in st.session_state:
                del st.session_state.opt
            st.rerun()

with c_right:
    st.subheader("🎯 Feature Importance")
    imp = pd.DataFrame({
        "Feature": [LABELS[f] for f in FEATURES],
        "Importance": metrics['perm_importance'],
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Teal")
    fig_imp.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("📈 Model Accuracy")
    acc = pd.DataFrame({"Actual": metrics['y_test'], "Predicted": metrics['y_pred']})
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=acc["Actual"], y=acc["Predicted"], mode="markers",
                                 marker=dict(color="#0D9488", size=9), name="Test batches"))
    lim = [0, max(acc.max()) * 1.05]
    fig_acc.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(dash="dash", color="gray"), name="Perfect fit"))
    fig_acc.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Actual Yield (g/L)", yaxis_title="Predicted Yield (g/L)")
    st.plotly_chart(fig_acc, use_container_width=True)

st.divider()
st.caption("Penicillin Fermentation Optimizer · Muhammad Luqman Bin Abd Latif · "
           "Universiti Kuala Lumpur (MICET) · 2026 · "
           "Built with the real IndPenSim benchmark dataset (Goldrick et al., 2019)")
