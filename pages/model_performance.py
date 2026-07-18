"""
pages/model_performance.py
-----------------------------
Model Performance page: displays evaluation metrics for both the
classification model comparison and the ATS regression model
comparison, plus a confusion matrix for the deployed classifier.
"""

import streamlit as st

from utils.charts import confusion_matrix_heatmap, model_accuracy_comparison


def render(models):
    st.markdown("## 📈 Model Performance")
    st.caption("Evaluation metrics comparing all candidate ML models trained in train.py.")

    if not models.metrics:
        st.warning("No metrics found. Run `python train.py` to generate models/metrics.json.")
        return

    tab1, tab2 = st.tabs(["Classification Models", "ATS Regression Models"])

    with tab1:
        clf = models.metrics["classification"]
        st.markdown(f"**Best model deployed:** `{clf['best_model']}`")

        metric_choice = st.selectbox(
            "Metric to compare", ["accuracy", "precision", "recall", "f1_score"], key="clf_metric"
        )
        st.plotly_chart(model_accuracy_comparison(clf["results"], metric=metric_choice),
                         width='stretch')

        st.markdown("#### Full Metrics Table")
        import pandas as pd
        df = pd.DataFrame(clf["results"]).T
        st.dataframe(df, width='stretch')

        if models.confusion_matrix is not None and models.classes:
            st.markdown("#### Confusion Matrix (Best Model)")
            st.plotly_chart(
                confusion_matrix_heatmap(models.confusion_matrix, models.classes),
                width='stretch',
            )

    with tab2:
        reg = models.metrics["regression"]
        st.markdown(f"**Best model deployed:** `{reg['best_model']}`")

        metric_choice = st.selectbox(
            "Metric to compare", ["mae", "rmse", "r2_score"], key="reg_metric"
        )
        st.plotly_chart(model_accuracy_comparison(reg["results"], metric=metric_choice),
                         width='stretch')

        st.markdown("#### Full Metrics Table")
        import pandas as pd
        df = pd.DataFrame(reg["results"]).T
        st.dataframe(df, width='stretch')

        st.caption(
            "MAE / RMSE: lower is better (average prediction error in ATS score points). "
            "R²: higher is better (proportion of variance explained, max = 1.0)."
        )
