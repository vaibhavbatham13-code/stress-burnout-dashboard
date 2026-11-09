"""
Real-time Mental Health & Stress Pattern Analytics Dashboard
------------------------------------------------------------
Run with:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run app.py

Disclaimer: this is an educational demo trained on SYNTHETIC data. It is not a
medical device and does not provide a diagnosis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FEATURES = ["sleep", "screen", "activity", "stress", "caffeine"]
PRETTY = {
    "sleep": "Sleep (hrs)",
    "screen": "Screen time (hrs)",
    "activity": "Physical activity (min)",
    "stress": "Workload / stress (1-10)",
    "caffeine": "Caffeine (cups)",
}
LABELS = ["Low", "Moderate", "High"]
COLORS = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}
DATA_FILE = Path(__file__).resolve().parent / "entries.csv"

st.set_page_config(
    page_title="Stress & Burnout Analytics",
    page_icon="🧠",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show_plot(fig, **kwargs):
    """Render a Plotly figure, compatible with old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------
# Synthetic data + model
# ----------------------------------------------------------------------------
def _risk_score(sleep, screen, activity, stress, caffeine, rng=None):
    """Hidden 'ground truth' relationship used to generate synthetic labels."""
    score = (
        0.55 * (stress - 5)
        + 0.60 * np.maximum(0, 7.5 - sleep)
        + 0.15 * np.maximum(0, screen - 6)
        - 0.012 * (np.clip(activity, 0, 120) - 30)
        + 0.18 * np.maximum(0, caffeine - 3)
    )
    if rng is not None:
        score = score + rng.normal(0, 0.5, size=np.shape(score))
    return score


def _score_to_label(score):
    return np.where(score < 0.8, 0, np.where(score < 2.6, 1, 2))


@st.cache_resource(show_spinner="Training the burnout risk model on synthetic data...")
def train_model(n_samples: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "sleep": rng.uniform(3, 10, n_samples).round(1),
            "screen": rng.uniform(1, 14, n_samples).round(1),
            "activity": rng.integers(0, 200, n_samples),
            "stress": rng.integers(1, 11, n_samples),
            "caffeine": rng.integers(0, 9, n_samples),
        }
    )
    score = _risk_score(
        df["sleep"].values,
        df["screen"].values,
        df["activity"].values,
        df["stress"].values,
        df["caffeine"].values,
        rng,
    )
    df["label"] = _score_to_label(score)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["label"], test_size=0.2, random_state=seed, stratify=df["label"]
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    background = df.sample(350, random_state=seed).copy()
    background["risk"] = [LABELS[i] for i in background["label"]]
    return model, background, acc


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Append risk label, confidence and burnout index columns."""
    df = df.copy()
    proba = model.predict_proba(df[FEATURES])
    idx = proba.argmax(axis=1)
    df["risk"] = [LABELS[i] for i in idx]
    df["confidence"] = (proba.max(axis=1) * 100).round(1)
    df["burnout_index"] = ((proba[:, 1] * 0.5 + proba[:, 2]) * 100).round(1)
    return df


def predict_one(model, inputs: dict):
    row = pd.DataFrame([inputs])[FEATURES]
    proba = model.predict_proba(row)[0]
    idx = int(proba.argmax())
    index = float((proba[1] * 0.5 + proba[2]) * 100)
    return LABELS[idx], float(proba[idx] * 100), proba, index


def seed_history(model, days: int = 14, end=None) -> pd.DataFrame:
    """Create a plausible two-week history ending on `end` (default: yesterday)."""
    rng = np.random.default_rng(7)
    if end is None:
        end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days)
    sleep = np.clip(rng.normal(6.4, 1.1, days), 3.5, 9.5).round(1)
    stress = np.clip(np.round(10.5 - 0.7 * sleep + rng.normal(0, 1.0, days)), 1, 10)
    df = pd.DataFrame(
        {
            "date": dates,
            "sleep": sleep,
            "screen": np.clip(rng.normal(8, 2.2, days), 2, 14).round(1),
            "activity": np.clip(rng.normal(35, 25, days), 0, 150).round(0),
            "stress": stress,
            "caffeine": np.clip(rng.normal(3, 1.3, days), 0, 8).round(0),
        }
    )
    return add_predictions(df, model)


def sample_history(model) -> pd.DataFrame:
    """Sample history whose end date stays fixed for the whole browser session."""
    if "seed_end" not in st.session_state:
        st.session_state.seed_end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    return seed_history(model, end=st.session_state.seed_end)


def load_history(model):
    """Load saved entries from DATA_FILE, or return None if there are none."""
    if not DATA_FILE.exists():
        return None
    try:
        raw = pd.read_csv(DATA_FILE, parse_dates=["date"])
        raw = raw[["date"] + FEATURES].dropna()
    except (ValueError, KeyError, pd.errors.ParserError):
        return None
    if raw.empty:
        return None
    raw = raw.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    # Predictions are recomputed from the current model; only raw inputs are stored.
    return add_predictions(raw, model)


def save_history(hist: pd.DataFrame) -> None:
    """Persist raw inputs (not predictions) so they stay valid if the model changes."""
    hist[["date"] + FEATURES].to_csv(DATA_FILE, index=False)


# ----------------------------------------------------------------------------
# Recommendation engine
# ----------------------------------------------------------------------------
def build_recommendations(inp: dict, risk: str):
    tips = []

    def add(priority, area, title, detail):
        tips.append(
            {"priority": priority, "area": area, "title": title, "detail": detail}
        )

    sleep, screen = inp["sleep"], inp["screen"]
    activity, stress, caffeine = inp["activity"], inp["stress"], inp["caffeine"]

    # Sleep
    if sleep < 6:
        add(3, "Sleep", "Start a 45-minute wind-down routine",
            "Dim lights, put your phone away, and do something calming (stretching, "
            "reading, journaling). Aim for a fixed lights-out time tonight to "
            "recover toward 7-8 hours.")
    elif sleep < 7:
        add(2, "Sleep", "Add 30-45 minutes of sleep",
            "Move your bedtime earlier in 15-minute steps and keep your wake-up "
            "time consistent, even on weekends.")
    elif sleep > 9.5:
        add(1, "Sleep", "Check your sleep quality",
            "Very long sleep can signal poor sleep quality or low mood. Keep a "
            "regular schedule and get morning daylight.")
    else:
        add(1, "Sleep", "Keep your sleep rhythm",
            "Your sleep is in a healthy range. Protect it by keeping consistent "
            "bed and wake times.")

    # Screen time
    if screen > 12:
        add(3, "Screen time", "Schedule hard screen breaks",
            "Block two 30-minute screen-free windows today (e.g. lunch and "
            "evening) and set a device curfew 1 hour before bed.")
    elif screen > 8:
        add(2, "Screen time", "Use the 20-20-20 rule",
            "Every 20 minutes, look at something 20 feet away for 20 seconds. "
            "Turn on grayscale or night mode after sunset.")
    else:
        add(1, "Screen time", "Maintain healthy screen habits",
            "Your screen time is manageable. Keep meals and bedtime screen-free.")

    # Physical activity
    if activity < 20:
        add(3 if stress >= 7 else 2, "Activity", "Add three 10-minute walks",
            "Short walks after meals lower stress hormones and improve sleep. "
            "Build gradually toward 30 minutes a day.")
    elif activity < 30:
        add(2, "Activity", "Top up to 30 minutes",
            "You are close. One extra brisk walk or a short mobility session "
            "gets you to the recommended daily minimum.")
    else:
        add(1, "Activity", "Great job staying active",
            "Keep your movement routine, and include one lighter recovery day "
            "each week.")

    # Workload / stress
    if stress >= 8:
        add(3, "Workload", "Triage your workload today",
            "List your tasks, pick the top 3, and defer or delegate the rest. "
            "Take a 5-minute breathing break (4-4-6 breathing) every 90 minutes "
            "and talk to a manager or colleague about capacity.")
    elif stress >= 6:
        add(2, "Workload", "Schedule micro-breaks",
            "Use 5-minute breaks between tasks and end the workday with a short "
            "shutdown ritual (review tomorrow's plan, close laptop).")
    else:
        add(1, "Workload", "Keep boundaries healthy",
            "Your load feels manageable. Keep clear start and end times for work.")

    # Caffeine
    if caffeine >= 6:
        add(3, "Caffeine", "Taper caffeine gradually",
            "Cut by one cup every 2-3 days to avoid withdrawal headaches. "
            "Replace later cups with water, herbal tea or decaf.")
    elif caffeine >= 4:
        add(2, "Caffeine", "Set a 2 PM caffeine cutoff",
            "Caffeine has a half-life of about 5-6 hours, so afternoon cups can "
            "reduce deep sleep even if you fall asleep fine.")
    else:
        add(1, "Caffeine", "Caffeine intake is moderate",
            "Keep it up, and avoid caffeine within 8 hours of bedtime.")

    # Overall risk-level additions
    if risk == "High":
        add(3, "Support", "Reach out to someone you trust",
            "Tell a friend, family member or colleague how you're doing. If "
            "feeling overwhelmed persists for more than two weeks, consider "
            "speaking with a qualified mental health professional.")
    elif risk == "Moderate":
        add(2, "Recovery", "Plan one recovery block this week",
            "Reserve 2-3 hours for something restorative: nature, a hobby, or "
            "time with friends, with no work notifications.")

    tips.sort(key=lambda t: -t["priority"])
    return tips


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def gauge_chart(index: float, risk: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=index,
            number={"suffix": "/100"},
            title={"text": "Burnout Risk Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS[risk]},
                "steps": [
                    {"range": [0, 33], "color": "rgba(46,204,113,0.25)"},
                    {"range": [33, 66], "color": "rgba(243,156,18,0.25)"},
                    {"range": [66, 100], "color": "rgba(231,76,60,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def probability_chart(proba):
    fig = go.Figure(
        go.Bar(
            x=LABELS,
            y=proba * 100,
            marker_color=[COLORS[l] for l in LABELS],
            text=[f"{p * 100:.1f}%" for p in proba],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model class probabilities",
        yaxis=dict(title="Probability (%)", range=[0, 110]),
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def factor_scores(inp: dict) -> dict:
    """Convert raw inputs to 0-100 'healthiness' scores (100 = ideal)."""
    return {
        "Sleep": min(inp["sleep"] / 8, 1) * 100,
        "Activity": min(inp["activity"] / 60, 1) * 100,
        "Screen balance": float(np.clip(1 - (inp["screen"] - 4) / 8, 0, 1) * 100),
        "Caffeine balance": float(np.clip(1 - (inp["caffeine"] - 2) / 6, 0, 1) * 100),
        "Workload balance": float(np.clip(1 - (inp["stress"] - 3) / 7, 0, 1) * 100),
    }


def radar_chart(inp: dict):
    scores = factor_scores(inp)
    cats = list(scores.keys())
    vals = list(scores.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[100] * (len(cats) + 1),
            theta=cats + [cats[0]],
            name="Ideal",
            line=dict(color="#95a5a6", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="Today",
            line=dict(color="#3498db"),
        )
    )
    fig.update_layout(
        title="Lifestyle balance (100 = ideal)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=380,
        margin=dict(l=40, r=40, t=60, b=30),
    )
    return fig


def importance_chart(model):
    imp = model.named_steps["clf"].feature_importances_
    df = pd.DataFrame(
        {"Factor": [PRETTY[f] for f in FEATURES], "Importance": imp}
    ).sort_values("Importance")
    fig = px.bar(df, x="Importance", y="Factor", orientation="h",
                 title="What drives the model's prediction")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=30))
    return fig


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
model, background, model_acc = train_model()

if "history" not in st.session_state:
    saved = load_history(model)
    st.session_state.history = saved if saved is not None else sample_history(model)
if "checklist" not in st.session_state:
    st.session_state.checklist = {}

# ----------------------------------------------------------------------------
# Sidebar: daily input form
# ----------------------------------------------------------------------------
st.sidebar.title("Daily Check-in")
st.sidebar.caption("Adjust the values; the prediction updates instantly.")

entry_date = st.sidebar.date_input(
    "Entry date", value=pd.Timestamp.today().date(), max_value=pd.Timestamp.today().date()
)
sleep_in = st.sidebar.slider("Sleep hours", 0.0, 12.0, 7.0, 0.5)
screen_in = st.sidebar.slider("Screen time (hours)", 0.0, 16.0, 7.0, 0.5)
activity_in = st.sidebar.slider("Physical activity (minutes)", 0, 300, 30, 5)
stress_in = st.sidebar.slider("Workload / stress rating (1-10)", 1, 10, 5)
caffeine_in = st.sidebar.number_input(
    "Caffeine (cups of coffee or equivalent)", min_value=0.0, max_value=15.0,
    value=2.0, step=0.5,
)
st.sidebar.caption(f"About {caffeine_in * 95:.0f} mg of caffeine (approx. 95 mg per cup).")

inputs = {
    "sleep": float(sleep_in),
    "screen": float(screen_in),
    "activity": float(activity_in),
    "stress": float(stress_in),
    "caffeine": float(caffeine_in),
}
risk, confidence, proba, burnout_index = predict_one(model, inputs)

if st.sidebar.button("Log today's entry", type="primary"):
    ts = pd.Timestamp(entry_date)
    hist = st.session_state.history
    existed = (hist["date"] == ts).any()
    new_row = add_predictions(pd.DataFrame([{"date": ts, **inputs}]), model)
    hist = hist[hist["date"] != ts]
    hist = pd.concat([hist, new_row], ignore_index=True).sort_values("date")
    st.session_state.history = hist.reset_index(drop=True)
    save_history(st.session_state.history)
    st.session_state.checklist = {}
    st.sidebar.success(
        f"{'Updated' if existed else 'Logged'} entry for {ts:%d %b %Y}."
    )

if st.sidebar.button("Reset to sample history"):
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    st.session_state.history = sample_history(model)
    st.session_state.checklist = {}
    st.sidebar.info("History reset to sample data. Saved entries were deleted.")

st.sidebar.divider()
st.sidebar.caption(
    f"Model: Random Forest on synthetic data (hold-out accuracy {model_acc * 100:.1f}%). "
    "Educational demo only; not medical advice."
)

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
st.title("🧠 Mental Health & Stress Pattern Analytics")
st.caption(f"Today: {pd.Timestamp.today():%A, %d %B %Y}")

tab1, tab2, tab3 = st.tabs(
    [
        "Daily Assessment & Prediction",
        "Historical Analytics & Trends",
        "Personalized Wellness Action Plan",
    ]
)

hist = st.session_state.history.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------- Tab 1 ----
with tab1:
    last7 = hist.tail(7)
    stress_delta = inputs["stress"] - last7["stress"].mean() if len(last7) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div style='padding:14px;border-radius:12px;background:{COLORS[risk]}22;"
        f"border:1px solid {COLORS[risk]}'>"
        f"<div style='font-size:0.85rem;opacity:0.8'>Predicted burnout risk</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{COLORS[risk]}'>{risk}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c2.metric("Model confidence", f"{confidence:.1f}%")
    c3.metric("Burnout index", f"{burnout_index:.0f} / 100")
    c4.metric(
        "Stress vs. last 7-day avg",
        f"{inputs['stress']:.0f} / 10",
        delta=f"{stress_delta:+.1f}",
        delta_color="inverse",
    )

    if risk == "High":
        st.error("High risk detected. Check the Action Plan tab for priority steps.")
    elif risk == "Moderate":
        st.warning("Moderate risk. A few small changes could bring you down a level.")
    else:
        st.success("Low risk. Your current routine looks sustainable.")

    g1, g2 = st.columns(2)
    with g1:
        show_plot(gauge_chart(burnout_index, risk))
    with g2:
        show_plot(probability_chart(proba))

    r1, r2 = st.columns(2)
    with r1:
        show_plot(radar_chart(inputs))
    with r2:
        show_plot(importance_chart(model))

# ---------------------------------------------------------------- Tab 2 ----
with tab2:
    window = st.radio(
        "Time window", ["Last 7 days", "Last 14 days", "Last 30 days", "All"],
        horizontal=True,
    )
    n_days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}.get(window)
    if n_days:
        cutoff = hist["date"].max() - pd.Timedelta(days=n_days - 1)
        view = hist[hist["date"] >= cutoff].copy()
    else:
        view = hist.copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Entries in window", len(view))
    k2.metric("Avg stress", f"{view['stress'].mean():.1f} / 10" if len(view) else "n/a")
    k3.metric("Avg sleep", f"{view['sleep'].mean():.1f} h" if len(view) else "n/a")
    k4.metric(
        "High-risk days",
        f"{(view['risk'] == 'High').mean() * 100:.0f}%" if len(view) else "n/a",
    )

    # Line chart: stress trend
    view["stress_ma"] = view["stress"].rolling(3, min_periods=1).mean()
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(x=view["date"], y=view["stress"], mode="lines+markers",
                   name="Daily stress", line=dict(color="#e74c3c"))
    )
    fig_line.add_trace(
        go.Scatter(x=view["date"], y=view["stress_ma"], mode="lines",
                   name="3-day average", line=dict(color="#3498db", dash="dash"))
    )
    fig_line.add_hline(y=7, line_dash="dot", line_color="gray",
                       annotation_text="High-stress threshold")
    fig_line.update_layout(
        title="Stress level trend",
        yaxis=dict(title="Stress (1-10)", range=[0, 10.5]),
        xaxis_title=None,
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    show_plot(fig_line)

    a, b = st.columns(2)

    # Scatter: sleep vs stress
    with a:
        fig_sc = go.Figure()
        fig_sc.add_trace(
            go.Scatter(
                x=background["sleep"], y=background["stress"], mode="markers",
                name="Reference population (synthetic)",
                marker=dict(color="lightgray", size=5, opacity=0.35),
            )
        )
        for lvl in LABELS:
            sub = view[view["risk"] == lvl]
            if len(sub):
                fig_sc.add_trace(
                    go.Scatter(
                        x=sub["sleep"], y=sub["stress"], mode="markers",
                        name=f"You: {lvl}",
                        marker=dict(color=COLORS[lvl], size=11,
                                    line=dict(width=1, color="white")),
                    )
                )
        if len(view) >= 3 and view["sleep"].std() > 0:
            slope, intercept = np.polyfit(view["sleep"], view["stress"], 1)
            xs = np.linspace(view["sleep"].min(), view["sleep"].max(), 20)
            fig_sc.add_trace(
                go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                           name="Your trend", line=dict(color="#34495e", dash="dash"))
            )
        fig_sc.update_layout(
            title="Sleep vs. stress",
            xaxis_title="Sleep (hours)", yaxis_title="Stress (1-10)",
            height=400, margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", y=-0.25),
        )
        show_plot(fig_sc)

    # Bar: factor breakdown averages
    with b:
        avg_inp = {f: float(view[f].mean()) for f in FEATURES} if len(view) else inputs
        scores = factor_scores(avg_inp)
        today_scores = factor_scores(inputs)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Window average", x=list(scores.keys()),
                                 y=list(scores.values()), marker_color="#3498db"))
        fig_bar.add_trace(go.Bar(name="Today (sidebar)", x=list(today_scores.keys()),
                                 y=list(today_scores.values()), marker_color="#9b59b6"))
        fig_bar.update_layout(
            title="Lifestyle factor breakdown (100 = ideal)",
            barmode="group", yaxis=dict(range=[0, 105]),
            height=400, margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", y=-0.25),
        )
        show_plot(fig_bar)

    d, e = st.columns(2)
    with d:
        fig_idx = px.bar(
            view, x="date", y="burnout_index", color="risk",
            color_discrete_map=COLORS, category_orders={"risk": LABELS},
            title="Daily burnout index",
        )
        fig_idx.update_layout(height=360, xaxis_title=None, yaxis_title="Index (0-100)",
                              margin=dict(l=20, r=20, t=60, b=20))
        show_plot(fig_idx)
    with e:
        if len(view) >= 3:
            corr = view[FEATURES].rename(columns=PRETTY).corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Correlation between factors")
            fig_corr.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
            show_plot(fig_corr)
        else:
            st.info("Log at least 3 entries to see factor correlations.")

    with st.expander("View and export logged entries"):
        table = view.copy()
        table["date"] = table["date"].dt.strftime("%Y-%m-%d")
        show_df(
            table.drop(columns=["stress_ma"], errors="ignore")
            .rename(columns={**PRETTY, "risk": "Risk", "confidence": "Confidence (%)",
                             "burnout_index": "Burnout index", "date": "Date"})
        )
        st.download_button(
            "Download history as CSV",
            data=hist.to_csv(index=False).encode("utf-8"),
            file_name="stress_history.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- Tab 3 ----
with tab3:
    st.subheader(f"Your plan for today: {risk} risk")
    tips = build_recommendations(inputs, risk)
    prio_label = {3: "High priority", 2: "Medium priority", 1: "Maintain"}
    prio_icon = {3: "🔴", 2: "🟠", 1: "🟢"}

    left, right = st.columns([3, 2])

    with left:
        for t in tips:
            with st.expander(
                f"{prio_icon[t['priority']]} {t['area']}: {t['title']}  "
                f"({prio_label[t['priority']]})",
                expanded=t["priority"] == 3,
            ):
                st.write(t["detail"])

    with right:
        st.markdown("#### This week's action checklist")
        actionable = [t for t in tips if t["priority"] >= 2][:5] or tips[:3]
        done = 0
        for t in actionable:
            key = f"chk_{t['area']}_{t['title']}"
            val = st.checkbox(t["title"], key=key,
                              value=st.session_state.checklist.get(key, False))
            st.session_state.checklist[key] = val
            done += int(val)
        st.progress(done / len(actionable) if actionable else 0.0,
                    text=f"{done} of {len(actionable)} completed")

    st.divider()
    st.markdown("#### Pattern insights from your history")
    i1, i2, i3 = st.columns(3)

    recent = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=7)]
    prior = hist[
        (hist["date"] <= hist["date"].max() - pd.Timedelta(days=7))
        & (hist["date"] > hist["date"].max() - pd.Timedelta(days=14))
    ]
    if len(recent) and len(prior):
        delta = recent["stress"].mean() - prior["stress"].mean()
        i1.metric("Stress: this week vs. previous", f"{recent['stress'].mean():.1f}",
                  delta=f"{delta:+.1f}", delta_color="inverse")
    else:
        i1.metric("Stress: this week vs. previous", "n/a")

    short = hist[hist["sleep"] < 6.5]
    enough = hist[hist["sleep"] >= 6.5]
    if len(short) and len(enough):
        i2.metric("Stress after short sleep (<6.5h)", f"{short['stress'].mean():.1f}",
                  delta=f"{short['stress'].mean() - enough['stress'].mean():+.1f} vs. longer sleep",
                  delta_color="inverse")
    else:
        i2.metric("Stress after short sleep (<6.5h)", "n/a")

    if len(hist) >= 3 and hist["activity"].std() > 0 and hist["stress"].std() > 0:
        r = hist["activity"].corr(hist["stress"])
        i3.metric("Activity vs. stress correlation", f"{r:+.2f}")
    else:
        i3.metric("Activity vs. stress correlation", "n/a")

    st.caption(
        "This dashboard uses a model trained on synthetic data for demonstration. "
        "It does not diagnose any condition. If you feel persistently overwhelmed, "
        "please talk to a qualified mental health professional or a local helpline."
    )