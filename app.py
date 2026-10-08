import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Air Quality Analysis", layout="wide")

POLL = ["PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3"]
BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_CUTS = [50, 100, 200, 300, 400]  # upper edges of the first five buckets
DEFAULT_FILE = "city_day.csv"
RANDOM_STATE = 42


# ---------------- Helpers ----------------

def aqi_to_bucket(value: float) -> str:
    """Map a numeric AQI to the official Indian CPCB category."""
    return BUCKETS[int(np.digitize(value, BUCKET_CUTS, right=True))]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature at a quantile learned from the TRAINING data only
    (avoids leaking test-set statistics, unlike clipping the whole dataset)."""

    def __init__(self, q=0.99):
        self.q = q

    def fit(self, X, y=None):
        self.upper_ = np.quantile(np.asarray(X), self.q, axis=0)
        return self

    def transform(self, X):
        out = np.minimum(np.asarray(X), self.upper_)
        return pd.DataFrame(out, columns=getattr(X, "columns", None), index=getattr(X, "index", None))


def heatmap(mat: pd.DataFrame, x_title: str, y_title: str, fmt: str = "d"):
    long = mat.stack().reset_index()
    long.columns = ["y", "x", "value"]
    base = alt.Chart(long).encode(
        x=alt.X("x:N", sort=list(mat.columns), title=x_title),
        y=alt.Y("y:N", sort=list(mat.index), title=y_title),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"), legend=None)
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("value:Q", format=fmt))
    return (rect + text).properties(height=max(250, 40 * len(mat)))


def clf_metrics(y_true, y_pred) -> dict:
    kw = dict(average="weighted", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision_score(y_true, y_pred, **kw),
        "Recall (weighted)": recall_score(y_true, y_pred, **kw),
        "F1 (weighted)": f1_score(y_true, y_pred, **kw),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------- Data ----------------

@st.cache_data
def load(src):
    return pd.read_csv(src)


@st.cache_data
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["AQI", "AQI_Bucket"])
    # Fill gaps with the city's own median first, then the global median.
    df[POLL] = df.groupby("City")[POLL].transform(lambda s: s.fillna(s.median()))
    df[POLL] = df[POLL].fillna(df[POLL].median())
    return df.reset_index(drop=True)


# ---------------- Models ----------------

@st.cache_resource(show_spinner="Training models (first run only)...")
def train(df: pd.DataFrame, split_mode: str) -> dict:
    chrono = split_mode.startswith("Chrono")
    data = df.sort_values("Date").reset_index(drop=True) if chrono else df
    X, y_aqi, y_cls = data[POLL], data["AQI"], data["AQI_Bucket"]

    if chrono:
        cut = int(len(data) * 0.8)
        Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
        ytr_aqi, yte_aqi = y_aqi.iloc[:cut], y_aqi.iloc[cut:]
        ytr_cls, yte_cls = y_cls.iloc[:cut], y_cls.iloc[cut:]
    else:
        Xtr, Xte, ytr_aqi, yte_aqi, ytr_cls, yte_cls = train_test_split(
            X, y_aqi, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
        )

    # Regression: linear baseline vs. non-linear model
    lr = make_pipeline(Winsorizer(), StandardScaler(), LinearRegression()).fit(Xtr, ytr_aqi)
    rf_reg = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(Xtr, ytr_aqi)

    # Classification
    grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [6, 10, 15, None],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="f1_macro",  # rare classes (Severe, Good) matter too
        n_jobs=-1,
    ).fit(Xtr, ytr_cls)
    nb = make_pipeline(Winsorizer(), GaussianNB()).fit(Xtr, ytr_cls)
    rf_clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ).fit(Xtr, ytr_cls)

    return {
        "lr": lr,
        "rf_reg": rf_reg,
        "dt": grid.best_estimator_,
        "dt_params": grid.best_params_,
        "nb": nb,
        "rf_clf": rf_clf,
        "Xte": Xte.reset_index(drop=True),
        "yte": pd.DataFrame(
            {"AQI": yte_aqi.values, "AQI_Bucket": yte_cls.values}
        ),
    }


@st.cache_data(show_spinner="Computing permutation importance...")
def perm_importance(_model, name, split_mode, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    res = permutation_importance(
        _model, X, y, scoring="accuracy", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.Series(res.importances_mean, index=X.columns)


@st.cache_data(show_spinner="Mining association rules...")
def mine_rules(df: pd.DataFrame, sup: float, conf: float) -> pd.DataFrame:
    samp = df.sample(min(8000, len(df)), random_state=1)
    cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    # rank(method="first") avoids qcut failing on heavily tied values
    cat = pd.DataFrame({
        p: pd.qcut(samp[p].rank(method="first"), 3, labels=["Low", "Medium", "High"])
        for p in cols
    })
    cat["AQI"] = samp["AQI_Bucket"].values
    basket = pd.get_dummies(cat, prefix_sep="=", dtype=bool)

    freq = apriori(basket, min_support=sup, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    if rules.empty:
        return pd.DataFrame()
    rules["IF"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["THEN"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules[["IF", "THEN", "support", "confidence", "lift"]]


# ---------------- Sidebar & loading ----------------

st.title("Air Quality Analysis - India (Data Mining Mini Project)")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload city_day.csv", type="csv")
    split_mode = st.radio(
        "Train / test split",
        ["Random (stratified)", "Chronological (last 20% = test)"],
        help="Chronological is a stricter, more realistic test: the model "
        "predicts days it has never seen, later in time.",
    )

source = uploaded if uploaded is not None else (DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None)
if source is None:
    st.info("Upload `city_day.csv` in the sidebar (or place it next to this script).")
    st.stop()

raw = load(source)
missing_cols = {"City", "Date", "AQI", "AQI_Bucket", *POLL} - set(raw.columns)
if missing_cols:
    st.error(f"Missing required columns: {sorted(missing_cols)}")
    st.stop()

df = preprocess(raw)
labels = [b for b in BUCKETS if b in set(df["AQI_Bucket"])]

bundle = train(df, split_mode)
Xte, yte = bundle["Xte"], bundle["yte"]

# Predictions are computed once and shared by every tab
reg_models = {"Linear Regression": bundle["lr"], "Random Forest": bundle["rf_reg"]}
reg_preds = {n: m.predict(Xte) for n, m in reg_models.items()}
clf_models = {
    "Decision Tree": bundle["dt"],
    "Naive Bayes": bundle["nb"],
    "Random Forest": bundle["rf_clf"],
}
clf_preds = {n: m.predict(Xte) for n, m in clf_models.items()}

t1, t2, t3, t4, t5, t6 = st.tabs([
    "1. Data & Preprocessing",
    "2. Regression",
    "3. Classification",
    "4. Clustering",
    "5. Association Rules",
    "6. Predict AQI",
])


# ---------------- 1. Preprocessing ----------------

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before", f"{len(raw):,}")
    c2.metric("Rows after", f"{len(df):,}")
    c3.metric("Cities", df["City"].nunique())
    c4.metric("Date range", f"{df['Date'].min():%Y-%m} to {df['Date'].max():%Y-%m}")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values before cleaning (%)")
        miss = (raw.isna().mean() * 100).round(1)
        st.bar_chart(miss[miss > 0].sort_values(ascending=False))
    with right:
        st.subheader("AQI category distribution")
        st.bar_chart(df["AQI_Bucket"].value_counts().reindex(labels))

    st.caption(
        "Missing pollutant values are filled with the city's median (then the global "
        "median). Rows without an AQI value are dropped. Outlier clipping happens "
        "inside the model pipelines, using training data only."
    )

    st.subheader("Average AQI by city")
    st.bar_chart(df.groupby("City")["AQI"].mean().sort_values(ascending=False))

    city = st.selectbox("Monthly AQI trend for city", sorted(df["City"].unique()))
    st.line_chart(df[df["City"] == city].set_index("Date")["AQI"].resample("MS").mean())

    st.subheader("Correlation between pollutants and AQI")
    st.altair_chart(
        heatmap(df[POLL + ["AQI"]].corr().round(2), "", "", fmt=".2f"),
    )

    st.subheader("Cleaned data sample")
    st.dataframe(df.head(20))


# ---------------- 2. Regression ----------------

with t2:
    st.write(
        "Target: **AQI** (numeric). Linear Regression is the baseline; Random Forest "
        "can capture the non-linear way AQI is computed (the worst pollutant dominates)."
    )

    rows = {
        n: {
            "R2": r2_score(yte["AQI"], p),
            "MAE": mean_absolute_error(yte["AQI"], p),
            "RMSE": float(np.sqrt(mean_squared_error(yte["AQI"], p))),
        }
        for n, p in reg_preds.items()
    }
    st.dataframe(pd.DataFrame(rows).T.round(3))

    reg_choice = st.radio("Show details for", list(reg_models), horizontal=True, key="reg_choice")
    pred = reg_preds[reg_choice]

    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs predicted AQI")
        scatter = pd.DataFrame({"Actual": yte["AQI"].values, "Predicted": pred})
        st.scatter_chart(scatter.sample(min(3000, len(scatter)), random_state=0), x="Actual", y="Predicted")
    with right:
        st.subheader("Mean absolute error by true AQI category")
        err = (
            pd.DataFrame({"cat": yte["AQI_Bucket"], "err": np.abs(yte["AQI"].values - pred)})
            .groupby("cat")["err"].mean()
            .reindex(labels)
        )
        st.bar_chart(err)

    st.subheader("Effect of each pollutant on AQI")
    st.caption("Linear Regression coefficients on standardised inputs (AQI change per 1 std-dev increase), so they are comparable.")
    st.bar_chart(pd.Series(bundle["lr"][-1].coef_, index=POLL))


# ---------------- 3. Classification ----------------

with t3:
    st.write(f"Target: **AQI_Bucket** ({', '.join(labels)})")
    st.caption(f"Split: {split_mode}. Decision Tree tuned by 5-fold grid search on macro-F1 "
               f"(best: {bundle['dt_params']}). Tree models use balanced class weights.")

    perf = pd.DataFrame({n: clf_metrics(yte["AQI_Bucket"], p) for n, p in clf_preds.items()}).T
    st.subheader("Model comparison")
    st.dataframe(perf.style.format("{:.2%}"))
    st.bar_chart(perf[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (macro)"]])

    choice = st.radio("Details for", list(clf_models), horizontal=True, key="clf_choice")
    preds = clf_preds[choice]

    left, right = st.columns(2)
    with left:
        st.subheader(f"{choice} - Confusion matrix")
        cm = pd.crosstab(yte["AQI_Bucket"], preds, rownames=["Actual"], colnames=["Predicted"])
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)
        st.altair_chart(heatmap(cm, "Predicted", "Actual"))
    with right:
        st.subheader(f"{choice} - Per-class report")
        rep = pd.DataFrame(
            classification_report(yte["AQI_Bucket"], preds, labels=labels, output_dict=True, zero_division=0)
        ).T
        st.dataframe(rep.round(3))

    st.subheader("Feature importance (permutation, test set)")
    st.caption("How much accuracy drops when one pollutant is shuffled - a fair, model-agnostic comparison.")
    n_eval = min(2000, len(Xte))
    idx = np.random.RandomState(0).choice(len(Xte), n_eval, replace=False)
    imp = pd.DataFrame({
        n: perm_importance(m, n, split_mode, Xte.iloc[idx], yte["AQI_Bucket"].iloc[idx])
        for n, m in clf_models.items()
    })
    st.bar_chart(imp)


# ---------------- 4. Clustering ----------------

with t4:
    st.write("Cities are grouped by their average pollutant levels (K-Means).")

    cd = df.groupby("City")[POLL].mean()
    Z = StandardScaler().fit_transform(cd)
    ks = list(range(2, min(8, len(cd) - 1) + 1))

    inertia, sil = [], []
    for k_ in ks:
        km = KMeans(k_, n_init=10, random_state=RANDOM_STATE).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_))

    left, right = st.columns(2)
    left.subheader("Elbow curve (lower is tighter)")
    left.line_chart(pd.Series(inertia, index=ks))
    right.subheader("Silhouette score (higher is better)")
    right.line_chart(pd.Series(sil, index=ks))

    best_k = ks[int(np.argmax(sil))]
    k = st.slider("Number of clusters (k)", ks[0], ks[-1], best_k,
                  help=f"Defaulted to the k with the best silhouette score ({best_k}).")

    cd["Cluster"] = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
    cd["Avg AQI"] = df.groupby("City")["AQI"].mean()

    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    coords = pcs.transform(Z)
    plot_df = cd.reset_index()[["City", "Cluster", "Avg AQI"]].assign(PC1=coords[:, 0], PC2=coords[:, 1])
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    st.subheader("Cities projected onto 2 principal components")
    st.caption(f"The two components keep {pcs.explained_variance_ratio_.sum():.0%} of the variation across all {len(POLL)} pollutants.")
    st.altair_chart(
        alt.Chart(plot_df).mark_circle(size=140).encode(
            x="PC1:Q", y="PC2:Q", color="Cluster:N", tooltip=["City", "Cluster", "Avg AQI"]
        ).interactive(),
    )

    st.dataframe(cd.sort_values("Cluster").round(1))
    st.subheader("Cluster profile")
    st.dataframe(cd.groupby("Cluster").mean().round(1))


# ---------------- 5. Association Rules ----------------

with t5:
    st.write("Pollutant levels are split into Low / Medium / High terciles, then Apriori finds co-occurring patterns.")

    s1, s2, s3 = st.columns(3)
    sup = s1.slider("Minimum support", 0.05, 0.5, 0.15, 0.05)
    conf = s2.slider("Minimum confidence", 0.5, 0.95, 0.7, 0.05)
    min_lift = s3.slider("Minimum lift", 1.0, 3.0, 1.0, 0.1,
                         help="Lift > 1 means the pattern is more common than chance.")

    rules = mine_rules(df, sup, conf)
    if rules.empty:
        st.warning("No rules found. Lower the minimum support or confidence.")
    else:
        if st.checkbox("Show only rules that predict an AQI level", True):
            rules = rules[rules["THEN"].str.startswith("AQI=")]
        rules = rules[rules["lift"] >= min_lift]
        if rules.empty:
            st.warning("No rules match these filters.")
        else:
            st.caption(f"{len(rules)} rules match (top 30 by lift shown).")
            st.dataframe(rules.sort_values("lift", ascending=False).round(3).head(30))


# ---------------- 6. Predict ----------------

with t6:
    st.write("Enter pollutant levels (µg/m³, CO in mg/m³) to get predictions from every model.")

    cols = st.columns(4)
    vals = {
        p: cols[i % 4].number_input(
            p, 0.0, float(df[p].max() * 2), float(df[p].median()), key=f"in_{p}"
        )
        for i, p in enumerate(POLL)
    }

    if st.button("Predict"):
        x = pd.DataFrame([vals])

        c1, c2 = st.columns(2)
        for col, (name, model) in zip((c1, c2), reg_models.items()):
            aqi = max(0.0, float(model.predict(x)[0]))
            col.metric(f"{name} - predicted AQI", f"{aqi:.0f}")
            col.caption(f"Category: {aqi_to_bucket(aqi)}")

        rows, proba_rf = [], None
        for name, model in clf_models.items():
            proba = model.predict_proba(x)[0]
            rows.append({
                "Model": name,
                "Predicted category": model.classes_[int(np.argmax(proba))],
                "Confidence": f"{proba.max():.0%}",
            })
            if name == "Random Forest":
                proba_rf = pd.Series(proba, index=model.classes_).reindex(labels).fillna(0)
        st.subheader("Category predictions")
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

        st.subheader("Random Forest class probabilities")
        st.bar_chart(proba_rf)

    st.caption(
        "Scale: Good 0-50, Satisfactory 51-100, Moderate 101-200, "
        "Poor 201-300, Very Poor 301-400, Severe 401+"
    )
