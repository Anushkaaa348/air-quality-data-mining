import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Air Quality Analysis", layout="wide")
st.title("Air Quality Analysis - India (Data Mining Mini Project)")

POLL = ["PM2.5", "PM10", "NO", "NO2", "NH3", "CO", "SO2", "O3"]

# ---------------- Load data ----------------
import os

if not os.path.exists("city_day.csv"):
    st.error("city_day.csv not found in the project folder.")
    st.stop()

up = "city_day.csv"


@st.cache_data
def load(f):
    return pd.read_csv(f)

@st.cache_data
def preprocess(raw):
    df = raw.drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["AQI", "AQI_Bucket"])

    df[POLL] = df.groupby("City")[POLL].transform(
        lambda s: s.fillna(s.median())
    )

    df[POLL] = df[POLL].fillna(df[POLL].median())

    for c in POLL:
        df[c] = df[c].clip(upper=df[c].quantile(0.99))

    return df
@st.cache_resource
def train(df):
    X = df[POLL]
    y_aqi = df["AQI"]
    y_class = df["AQI_Bucket"]

    Xtr, Xte, ytr_aqi, yte_aqi, ytr_class, yte_class = train_test_split(
        X,
        y_aqi,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    reg = LinearRegression().fit(Xtr, ytr_aqi)

    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [6, 8, 10, 12, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(Xtr, ytr_class)
    dt = grid.best_estimator_

    nb = GaussianNB().fit(Xtr, ytr_class)

    yte = pd.DataFrame({
        "AQI": yte_aqi.values,
        "AQI_Bucket": yte_class.values
    })

    return reg, dt, nb, Xte, yte


@st.cache_resource
def train(df):
    X = df[POLL]
    y_aqi = df["AQI"]
    y_class = df["AQI_Bucket"]

    Xtr, Xte, ytr_aqi, yte_aqi, ytr_class, yte_class = train_test_split(
        X,
        y_aqi,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    reg = LinearRegression().fit(Xtr, ytr_aqi)

    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [6, 8, 10, 12, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(Xtr, ytr_class)
    dt = grid.best_estimator_

    nb = GaussianNB().fit(Xtr, ytr_class)

    yte = pd.DataFrame({
        "AQI": yte_aqi.values,
        "AQI_Bucket": yte_class.values
    })

    return reg, dt, nb, Xte, yte

raw = load(up)
df = preprocess(raw)
reg, dt, nb, Xte, yte = train(df)

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["1. Data & Preprocessing", "2. Regression", "3. Classification",
     "4. Clustering", "5. Association Rules", "6. Predict AQI"])

# ---------------- 1. Preprocessing ----------------
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", len(raw))
    c2.metric("Rows after", len(df))
    c3.metric("Cities", df["City"].nunique())
    st.subheader("Missing values before cleaning")
    st.dataframe(raw.isna().sum().rename("missing").to_frame().T)
    st.subheader("Average AQI by city")
    st.bar_chart(df.groupby("City")["AQI"].mean().sort_values(ascending=False))
    city = st.selectbox("Monthly AQI trend for city", sorted(df["City"].unique()))
    st.line_chart(df[df["City"] == city].set_index("Date")["AQI"].resample("MS").mean())
    st.subheader("Cleaned data sample")
    st.dataframe(df.head(20))

# ---------------- 2. Regression ----------------
with t2:
    pred = reg.predict(Xte)
    c1, c2 = st.columns(2)
    c1.metric("R2 score", round(r2_score(yte["AQI"], pred), 3))
    c2.metric("Mean absolute error", round(mean_absolute_error(yte["AQI"], pred), 2))
    st.subheader("Effect of each pollutant on AQI (coefficients)")
    st.bar_chart(pd.Series(reg.coef_, index=POLL))
    st.subheader("Actual vs predicted AQI")
    st.scatter_chart(pd.DataFrame({"Actual": yte["AQI"].values, "Predicted": pred}),
                     x="Actual", y="Predicted")

# ---------------- 3. Classification ----------------
with t3:
    st.write("Target: **AQI_Bucket** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)")
    res = {}
    for name, m in [("Decision Tree ", dt), ("Naive Bayes", nb)]:
        res[name] = accuracy_score(yte["AQI_Bucket"], m.predict(Xte))
    c1, c2 = st.columns(2)
    for col, (n, a) in zip([c1, c2], res.items()):
        col.metric(n + " accuracy", f"{a:.1%}")
    choice = st.radio("Confusion matrix for", list(res), horizontal=True)
    model = dt if choice.startswith("Decision") else nb
    st.dataframe(pd.crosstab(yte["AQI_Bucket"], model.predict(Xte),
                             rownames=["Actual"], colnames=["Predicted"]))
    st.subheader("Decision tree - feature importance")
    st.bar_chart(pd.Series(dt.feature_importances_, index=POLL))

# ---------------- 4. Clustering ----------------
with t4:
    st.write("Cities are grouped by their average pollutant levels (K-Means).")
    cd = df.groupby("City")[POLL].mean()
    Z = StandardScaler().fit_transform(cd)
    inertia = [KMeans(k, n_init=10, random_state=42).fit(Z).inertia_ for k in range(2, 9)]
    st.subheader("Elbow curve")
    st.line_chart(pd.Series(inertia, index=range(2, 9)))
    k = st.slider("Number of clusters (k)", 2, 8, 3)
    cd["Cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    cd["Avg AQI"] = df.groupby("City")["AQI"].mean()
    st.scatter_chart(cd.reset_index().astype({"Cluster": str}), x="PM2.5", y="PM10", color="Cluster")
    st.dataframe(cd.sort_values("Cluster").round(1))
    st.subheader("Cluster profile")
    st.dataframe(cd.groupby("Cluster").mean().round(1))

# ---------------- 5. Association rules ----------------
with t5:
    st.write("Pollutant levels are discretised into Low / Medium / High, then Apriori finds patterns.")
    s1, s2 = st.columns(2)
    sup = s1.slider("Minimum support", 0.05, 0.5, 0.15, 0.05)
    conf = s2.slider("Minimum confidence", 0.5, 0.95, 0.7, 0.05)
    samp = df.sample(min(8000, len(df)), random_state=1)
    cat = pd.DataFrame({p: pd.qcut(samp[p], 3, labels=["Low", "Medium", "High"], duplicates="drop")
                        for p in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]})
    cat["AQI"] = samp["AQI_Bucket"].values
    basket = pd.get_dummies(cat, prefix_sep="=", dtype=bool)
    freq = apriori(basket, min_support=sup, use_colnames=True)
    if freq.empty:
        st.warning("No frequent itemsets. Lower the minimum support.")
    else:
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        if rules.empty:
            st.warning("No rules found. Lower the confidence.")
        else:
            rules["IF"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            rules["THEN"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
            only_aqi = st.checkbox("Show only rules that predict an AQI level", True)
            if only_aqi:
                rules = rules[rules["THEN"].str.startswith("AQI=")]
            st.dataframe(rules[["IF", "THEN", "support", "confidence", "lift"]]
                         .sort_values("lift", ascending=False).round(3).head(30))

# ---------------- 6. Predict ----------------
with t6:
    st.write("Enter pollutant levels to get a prediction.")
    cols = st.columns(4)
    vals = {}
    for i, p in enumerate(POLL):
        vals[p] = cols[i % 4].number_input(p, 0.0, float(df[p].max() * 2), float(df[p].median()))
    if st.button("Predict"):
        x = pd.DataFrame([vals])
        st.success(f"Predicted AQI (Regression): {reg.predict(x)[0]:.0f}")
        st.info(f"Category (Decision Tree): {dt.predict(x)[0]}")
        st.info(f"Category (Naive Bayes): {nb.predict(x)[0]}")
        st.caption("Scale: Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, "
                   "Very Poor 301-400, Severe 401+")
