"""
Model evaluation, comparison, and metric visualization module for NewsInsight AI.
Provides Plotly figures for confusion matrices, metric comparisons, and classification reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config.settings import EVAL_RESULTS_PATH, COLOR_PALETTE


def save_evaluation_results(eval_dict: Dict[str, Any], filepath: Path = EVAL_RESULTS_PATH) -> None:
    """Saves combined evaluation results for topic and sentiment models to JSON."""
    with open(filepath, "w") as f:
        json.dump(eval_dict, f, indent=2)


def load_evaluation_results(filepath: Path = EVAL_RESULTS_PATH) -> Optional[Dict[str, Any]]:
    """Loads saved evaluation results from JSON if available."""
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def create_model_comparison_bar_chart(models_dict: Dict[str, Any], title: str = "Model Performance Comparison") -> go.Figure:
    """
    Creates a grouped bar chart comparing Accuracy, Macro-F1, Precision, and Recall.
    """
    model_names = list(models_dict.keys())
    accuracies = [models_dict[m]["accuracy"] for m in model_names]
    macro_f1s = [models_dict[m]["macro_f1"] for m in model_names]
    precisions = [models_dict[m]["precision"] for m in model_names]
    recalls = [models_dict[m]["recall"] for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy",
        x=model_names,
        y=accuracies,
        marker_color="#3B82F6",
        text=[f"{v:.3f}" for v in accuracies],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Macro-F1",
        x=model_names,
        y=macro_f1s,
        marker_color="#8B5CF6",
        text=[f"{v:.3f}" for v in macro_f1s],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Precision",
        x=model_names,
        y=precisions,
        marker_color="#10B981",
        text=[f"{v:.3f}" for v in precisions],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Recall",
        x=model_names,
        y=recalls,
        marker_color="#F59E0B",
        text=[f"{v:.3f}" for v in recalls],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F8FAFC", size=16)),
        barmode="group",
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        font=dict(color="#94A3B8"),
        yaxis=dict(range=[0, 1.05], gridcolor="#334155", title="Score"),
        xaxis=dict(gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_confusion_matrix_heatmap(
    confusion_matrix_data: List[List[int]],
    labels: List[str],
    model_name: str
) -> go.Figure:
    """
    Renders a styled confusion matrix heatmap with counts and text annotations.
    """
    cm = np.array(confusion_matrix_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        colorscale=[[0, "#1E293B"], [0.5, "#3B82F6"], [1.0, "#8B5CF6"]],
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#F8FAFC"}
    ))

    fig.update_layout(
        title=dict(text=f"Confusion Matrix: {model_name}", font=dict(color="#F8FAFC", size=15)),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        font=dict(color="#94A3B8"),
        xaxis=dict(tickangle=-25),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=40, t=50, b=50)
    )
    return fig


def create_classification_report_table(report_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Formats scikit-learn's classification_report dictionary into a clean pandas DataFrame.
    """
    rows = []
    for key, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append({
                "Class / Metric": key,
                "Precision": round(metrics.get("precision", 0.0), 3),
                "Recall": round(metrics.get("recall", 0.0), 3),
                "F1-Score": round(metrics.get("f1-score", 0.0), 3),
                "Support": int(metrics.get("support", 0))
            })
        elif key == "accuracy":
            rows.append({
                "Class / Metric": "Overall Accuracy",
                "Precision": "-",
                "Recall": "-",
                "F1-Score": round(float(metrics), 3),
                "Support": sum([v.get("support", 0) for k, v in report_dict.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")])
            })
    return pd.DataFrame(rows)
