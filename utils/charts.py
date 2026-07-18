"""
utils/charts.py
-----------------
Reusable Plotly chart builders for the Streamlit app: ATS gauge,
probability bar chart, model comparison charts, confusion matrix
heatmap, skill distribution, and job match gauge.
"""

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go


def ats_gauge(score: float, title: str = "ATS Score") -> go.Figure:
    if score >= 75:
        bar_color = "#22C55E"
    elif score >= 50:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 50], "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100], "color": "#DCFCE7"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def match_gauge(pct: float, title: str = "Job Match") -> go.Figure:
    return ats_gauge(pct, title=title)


def category_probability_bar(proba_map: Dict[str, float]) -> go.Figure:
    categories = list(proba_map.keys())
    probs = [round(p * 100, 2) for p in proba_map.values()]
    sorted_pairs = sorted(zip(categories, probs), key=lambda x: x[1], reverse=True)
    categories, probs = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(probs), y=list(categories), orientation="h",
        marker_color="#2563EB", text=[f"{p}%" for p in probs], textposition="outside",
    ))
    fig.update_layout(
        title="Category Prediction Probabilities",
        xaxis_title="Probability (%)", yaxis_title="",
        height=350, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(range=[0, 105]),
    )
    return fig


def model_accuracy_comparison(results: Dict[str, Dict], metric: str = "accuracy") -> go.Figure:
    names = list(results.keys())
    values = [results[n].get(metric, 0) * (100 if metric != "r2_score" else 1) for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color="#7C3AED",
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=f"Model Comparison — {metric.replace('_', ' ').title()}",
        yaxis_title=metric.replace("_", " ").title(),
        height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale="Blues", text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=450, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def skill_distribution_bar(skills: List[str]) -> go.Figure:
    unique_skills = list(dict.fromkeys(skills))
    fig = go.Figure(go.Bar(
        x=unique_skills, y=[1] * len(unique_skills),
        marker_color="#0EA5E9",
    ))
    fig.update_layout(
        title="Detected Skills",
        showlegend=False, height=320,
        yaxis=dict(showticklabels=False, title=""),
        margin=dict(l=10, r=10, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def resume_score_radar(scores: Dict[str, float]) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself", line_color="#2563EB",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=400, margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def ranking_bar_chart(names: List[str], scores: List[float], title: str = "Resume Ranking") -> go.Figure:
    sorted_pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    names_sorted, scores_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(go.Bar(
        x=list(scores_sorted), y=list(names_sorted), orientation="h",
        marker_color="#16A34A", text=[f"{s:.1f}" for s in scores_sorted], textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Score", height=max(300, 50 * len(names_sorted)),
        margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(range=[0, 105]),
    )
    return fig
