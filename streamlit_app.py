"""
Porter Delivery Time Predictor — Streamlit App
================================================
Loads the trained Keras neural network + preprocessing artifacts
and lets the user get live delivery-time predictions, plus a
dashboard of the EDA / model performance from training.

Run:
    streamlit run streamlit_app.py

All required files are expected in the same directory as this file:

    model.pkl
    scaler.pkl
    feature_columns.pkl
    category_reference.pkl
    metrics.json
    sample_data.csv

    predicted_vs_actual.png
    training_history.png
    target_distribution.png
    correlation_heatmap.png
    delivery_time_by_hour.png
    delivery_time_by_category.png
    outstanding_vs_delivery.png
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from tensorflow import keras


# ------------------------------------------------------------------
# Paths & page config
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Porter Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
)


# ------------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------------

NUMERIC_FEATURES = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_partners",
    "total_busy_partners",
    "total_outstanding_orders",
    "order_hour",
    "order_day_of_week",
    "order_is_weekend",
    "hour_sin",
    "hour_cos",
    "price_range",
    "avg_item_price",
    "items_per_distinct",
    "busy_partner_ratio",
    "available_partners",
    "outstanding_per_partner",
]

CATEGORICAL_FEATURES = [
    "market_id",
    "order_protocol",
    "store_primary_category",
]


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    with open(
        os.path.join(BASE_DIR, "model.pkl"),
        "rb"
    ) as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(
        os.path.join(BASE_DIR, "scaler.pkl"),
        "rb"
    ) as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_columns():
    with open(
        os.path.join(BASE_DIR, "feature_columns.pkl"),
        "rb"
    ) as f:
        return pickle.load(f)


@st.cache_resource
def load_category_reference():
    with open(
        os.path.join(BASE_DIR, "category_reference.pkl"),
        "rb"
    ) as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    with open(
        os.path.join(BASE_DIR, "metrics.json"),
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@st.cache_data
def load_sample_data():
    return pd.read_csv(
        os.path.join(BASE_DIR, "sample_data.csv")
    )


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def engineer_single_input(
    raw: dict,
    category_reference: dict,
    feature_columns: list
) -> pd.DataFrame:

    df = pd.DataFrame([raw])

    # Time features
    df["order_hour"] = df["order_hour"].astype(int)

    df["order_day_of_week"] = (
        df["order_day_of_week"].astype(int)
    )

    df["order_is_weekend"] = (
        df["order_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["order_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["order_hour"] / 24
    )

    # Price features
    df["price_range"] = (
        df["max_item_price"]
        - df["min_item_price"]
    )

    df["avg_item_price"] = (
        df["subtotal"]
        / df["total_items"].replace(0, 1)
    )

    df["items_per_distinct"] = (
        df["total_items"]
        / df["num_distinct_items"].replace(0, 1)
    )

    # Marketplace load features
    df["busy_partner_ratio"] = (
        df["total_busy_partners"]
        / (df["total_onshift_partners"] + 1)
    )

    df["available_partners"] = (
        df["total_onshift_partners"]
        - df["total_busy_partners"]
    ).clip(lower=0)

    df["outstanding_per_partner"] = (
        df["total_outstanding_orders"]
        / (df["total_onshift_partners"] + 1)
    )

    # --------------------------------------------------------------
    # Handle unseen store category
    # --------------------------------------------------------------

    if (
        df.loc[0, "store_primary_category"]
        not in category_reference["store_primary_category"]
    ):

        if "other" in category_reference[
            "store_primary_category"
        ]:
            df["store_primary_category"] = "other"

        else:
            df["store_primary_category"] = (
                category_reference[
                    "store_primary_category"
                ][0]
            )

    # --------------------------------------------------------------
    # Select features
    # --------------------------------------------------------------

    X = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES
    )

    # Align with exact training-time feature columns
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_delivery_time(raw: dict) -> float:

    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()
    category_reference = load_category_reference()

    X = engineer_single_input(
        raw,
        category_reference,
        feature_columns
    )

    # Scale features exactly as during training
    X_scaled = scaler.transform(X)

    # Model prediction
    pred = model.predict(
        X_scaled,
        verbose=0
    ).flatten()[0]

    # Prevent negative prediction
    return max(float(pred), 1.0)


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------

st.sidebar.title("🛵 Porter Delivery Time")

page = st.sidebar.radio(
    "Navigate",
    [
        "Predict",
        "Model Performance",
        "EDA Dashboard",
        "About",
    ],
)


# ------------------------------------------------------------------
# Check required files
# ------------------------------------------------------------------

REQUIRED_FILES = [
    "model.pkl",
    "scaler.pkl",
    "feature_columns.pkl",
    "category_reference.pkl",
    "metrics.json",
    "sample_data.csv",
]

missing_files = [
    file
    for file in REQUIRED_FILES
    if not os.path.exists(
        os.path.join(BASE_DIR, file)
    )
]


if missing_files:

    st.error(
        "Required files are missing from the project root."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


category_reference = load_category_reference()


# ==================================================================
# PAGE: PREDICT
# ==================================================================

if page == "Predict":

    st.title(
        "🛵 Porter Delivery Time Predictor"
    )

    st.caption(
        "Neural-network regression model estimating "
        "delivery duration (minutes) from order & "
        "marketplace conditions."
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------------
    # Order Details
    # --------------------------------------------------------------

    with col1:

        st.subheader("Order Details")

        total_items = st.number_input(
            "Total items",
            min_value=1,
            max_value=50,
            value=3,
        )

        num_distinct_items = st.number_input(
            "Distinct items",
            min_value=1,
            max_value=50,
            value=2,
        )

        subtotal = st.number_input(
            "Subtotal (cents)",
            min_value=0,
            max_value=20000,
            value=2500,
            step=100,
        )

        min_item_price = st.number_input(
            "Min item price (cents)",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
        )

        max_item_price = st.number_input(
            "Max item price (cents)",
            min_value=0,
            max_value=10000,
            value=1500,
            step=50,
        )

    # --------------------------------------------------------------
    # Store & Order Context
    # --------------------------------------------------------------

    with col2:

        st.subheader(
            "Store & Order Context"
        )

        store_primary_category = st.selectbox(
            "Store category",
            options=category_reference[
                "store_primary_category"
            ],
        )

        market_id = st.selectbox(
            "Market ID",
            options=category_reference[
                "market_id"
            ],
        )

        order_protocol = st.selectbox(
            "Order protocol",
            options=category_reference[
                "order_protocol"
            ],
        )

        order_hour = st.slider(
            "Order hour (24h)",
            0,
            23,
            19,
        )

        order_day_of_week = st.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda x: [
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun",
            ][x],
            index=4,
        )

    # --------------------------------------------------------------
    # Marketplace Conditions
    # --------------------------------------------------------------

    with col3:

        st.subheader(
            "Marketplace Conditions"
        )

        total_onshift_partners = st.number_input(
            "Partners on shift",
            min_value=0,
            max_value=200,
            value=20,
        )

        total_busy_partners = st.number_input(
            "Busy partners",
            min_value=0,
            max_value=200,
            value=15,
        )

        total_outstanding_orders = st.number_input(
            "Outstanding orders",
            min_value=0,
            max_value=300,
            value=25,
        )

    st.markdown("---")

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    if st.button(
        "🔮 Predict Delivery Time",
        type="primary",
        use_container_width=True,
    ):

        raw_input = {
            "total_items": total_items,
            "subtotal": subtotal,
            "num_distinct_items": num_distinct_items,
            "min_item_price": min_item_price,
            "max_item_price": max_item_price,
            "total_onshift_partners":
                total_onshift_partners,
            "total_busy_partners":
                total_busy_partners,
            "total_outstanding_orders":
                total_outstanding_orders,
            "order_hour": order_hour,
            "order_day_of_week":
                order_day_of_week,
            "market_id": market_id,
            "order_protocol":
                order_protocol,
            "store_primary_category":
                store_primary_category,
        }

        with st.spinner(
            "Running neural network..."
        ):

            prediction = predict_delivery_time(
                raw_input
            )

        c1, c2 = st.columns([1, 2])

        with c1:

            st.metric(
                "Estimated Delivery Time",
                f"{prediction:.1f} min",
            )

            eta_low = max(
                prediction - 8,
                1,
            )

            eta_high = prediction + 8

            st.caption(
                f"Typical range: "
                f"{eta_low:.0f}–"
                f"{eta_high:.0f} min"
            )

        with c2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={
                        "text":
                        "Delivery ETA (minutes)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 120]
                        },
                        "bar": {
                            "color": "#FF6B35"
                        },
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#d4f7d4",
                            },
                            {
                                "range": [30, 60],
                                "color": "#fff2cc",
                            },
                            {
                                "range": [60, 120],
                                "color": "#f8cccc",
                            },
                        ],
                    },
                )
            )

            fig.update_layout(
                height=280,
                margin=dict(
                    t=40,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ==================================================================
# PAGE: MODEL PERFORMANCE
# ==================================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        f"{metrics['mae']:.2f} min",
    )

    c2.metric(
        "RMSE",
        f"{metrics['rmse']:.2f} min",
    )

    c3.metric(
        "R² Score",
        f"{metrics['r2']:.3f}",
    )

    c4.metric(
        "Test Set Size",
        f"{metrics['n_test']:,}",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------------
    # Predicted vs Actual
    # --------------------------------------------------------------

    with col1:

        st.subheader(
            "Predicted vs Actual"
        )

        img_path = os.path.join(
            BASE_DIR,
            "predicted_vs_actual.png",
        )

        if os.path.exists(img_path):

            st.image(
                img_path,
                use_container_width=True,
            )

        else:

            st.warning(
                "predicted_vs_actual.png not found."
            )

    # --------------------------------------------------------------
    # Training History
    # --------------------------------------------------------------

    with col2:

        st.subheader(
            "Training History"
        )

        img_path = os.path.join(
            BASE_DIR,
            "training_history.png",
        )

        if os.path.exists(img_path):

            st.image(
                img_path,
                use_container_width=True,
            )

        else:

            st.warning(
                "training_history.png not found."
            )

    st.info(
        "The model is trained with Huber loss "
        "(robust to outliers) using a feed-forward "
        "neural network (256→128→64→32→1) with "
        "batch normalization, dropout regularization, "
        "early stopping, and learning-rate scheduling."
    )


# ==================================================================
# PAGE: EDA DASHBOARD
# ==================================================================

elif page == "EDA Dashboard":

    st.title(
        "🔎 Exploratory Data Analysis"
    )

    sample_df = load_sample_data()

    st.subheader(
        "Live filters (sample of cleaned training data)"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------------
    # Hour filter
    # --------------------------------------------------------------

    with col1:

        hour_range = st.slider(
            "Order hour range",
            0,
            23,
            (0, 23),
        )

    # --------------------------------------------------------------
    # Category filter
    # --------------------------------------------------------------

    with col2:

        cats = sorted(
            sample_df[
                "store_primary_category"
            ]
            .unique()
            .tolist()
        )

        selected_cats = st.multiselect(
            "Store categories",
            cats,
            default=cats[:8],
        )

    # --------------------------------------------------------------
    # Filter data
    # --------------------------------------------------------------

    filtered = sample_df[
        (
            sample_df["order_hour"].between(
                hour_range[0],
                hour_range[1],
            )
        )
        &
        (
            sample_df[
                "store_primary_category"
            ].isin(selected_cats)
            if selected_cats
            else True
        )
    ]

    # --------------------------------------------------------------
    # Delivery time distribution
    # --------------------------------------------------------------

    fig = px.histogram(
        filtered,
        x="delivery_time_minutes",
        nbins=50,
        title=(
            "Delivery Time Distribution "
            "(filtered)"
        ),
        color_discrete_sequence=[
            "#FF6B35"
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------------
    # Delivery time by category
    # --------------------------------------------------------------

    fig2 = px.box(
        filtered,
        x="store_primary_category",
        y="delivery_time_minutes",
        title=(
            "Delivery Time by Category "
            "(filtered)"
        ),
    )

    fig2.update_xaxes(
        tickangle=45
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader(
        "Pre-generated EDA plots "
        "(full training set)"
    )

    # All PNG files are directly in the
    # project root — no eda_plots folder.

    plot_files = [
        (
            "target_distribution.png",
            "Target Distribution",
        ),
        (
            "correlation_heatmap.png",
            "Correlation Heatmap",
        ),
        (
            "delivery_time_by_hour.png",
            "Delivery Time by Hour",
        ),
        (
            "delivery_time_by_category.png",
            "Delivery Time by Category",
        ),
        (
            "outstanding_vs_delivery.png",
            "Outstanding Orders vs Delivery Time",
        ),
    ]

    cols = st.columns(2)

    for i, (fname, title) in enumerate(
        plot_files
    ):

        path = os.path.join(
            BASE_DIR,
            fname,
        )

        if os.path.exists(path):

            with cols[i % 2]:

                st.markdown(
                    f"**{title}**"
                )

                st.image(
                    path,
                    use_container_width=True,
                )

        else:

            with cols[i % 2]:

                st.warning(
                    f"{fname} not found."
                )


# ==================================================================
# PAGE: ABOUT
# ==================================================================

else:

    st.title(
        "ℹ️ About This Project"
    )

    st.markdown(
        """
This app predicts **Porter delivery time (in minutes)**
using a neural-network regression model trained on
historical delivery data.

### Pipeline

1. **Data cleaning** — timestamp parsing, target derivation,
   missing-value imputation, outlier capping, deduplication.

2. **Feature engineering** — cyclic time encodings,
   price ratios, marketplace load ratios
   (busy-partner ratio, outstanding orders per partner),
   and category cardinality reduction.

3. **EDA** — distribution, correlation, and
   category/time-based analyses.

4. **Modeling** — a Keras feed-forward neural network
   (Dense + BatchNorm + Dropout) trained with Huber loss,
   early stopping, and LR scheduling.

5. **Evaluation** — MAE, RMSE, R² on a held-out test set.

6. **Deployment** — model & preprocessing objects
   saved and served through this Streamlit app.

### Tech Stack

Python, Pandas, Scikit-learn, TensorFlow/Keras,
Streamlit, Plotly.
"""
    )
