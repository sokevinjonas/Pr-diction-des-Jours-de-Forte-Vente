"""
Dashboard de monitoring — Visualiser la santé du modèle en production.
Affiche les métriques, les dérives et les alertes.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.observability import (
    get_metrics_history, get_alerts, get_predictions
)
from src.monitoring import ModelMonitor

st.set_page_config(page_title="Monitoring ML", layout="wide")

st.title("🔍 Monitoring — Santé du modèle")

# === Sidebar ===
st.sidebar.header("Configuration")
use_case = st.sidebar.selectbox(
    "Sélectionner un commerce",
    ["supermarche", "restaurant", "mobile_money", "grossiste"],
)

baseline_mape = st.sidebar.slider("Seuil MAPE d'alerte", 10, 50, 20)
days_check = st.sidebar.slider("Période de vérification (jours)", 7, 90, 7)

# === Initialiser le monitor ===
monitor = ModelMonitor(use_case, baseline_mape=baseline_mape)

# === Health Check ===
st.header("⚕️ Diagnostic général")

recent_preds = get_predictions(use_case, days=days_check, include_nulls=False)
health_report = monitor.run_full_check(recent_preds)

# Afficher le statut global
col1, col2, col3 = st.columns(3)

with col1:
    health = health_report["health"]
    color = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}[health]
    st.metric("Santé globale", f"{color} {health.upper()}")

with col2:
    perf = health_report["checks"]["performance_drift"]
    st.metric("Dérive performance", f"{perf.get('drift_pct', 0):.0f}%", perf.get("status"))

with col3:
    coverage = health_report["checks"]["prediction_coverage"]
    st.metric("Couverture prédictions", f"{coverage.get('coverage_pct', 0):.0f}%")

# === Détails des checks ===
st.subheader("Détails des vérifications")

for check_name, check_result in health_report["checks"].items():
    status_emoji = {"ok": "✅", "warning": "⚠️", "critical": "❌"}[check_result.get("status", "ok")]
    message = check_result.get("message", "")

    with st.expander(f"{status_emoji} {check_name.replace('_', ' ').title()}"):
        st.write(message)
        for key, value in check_result.items():
            if key not in ["status", "message"]:
                st.write(f"  **{key}:** {value}")

# === Metrics historiques ===
st.header("📊 Historique des métriques")

metrics_df = get_metrics_history(use_case, days=days_check)

if not metrics_df.empty:
    # Graphique MAPE dans le temps
    fig_mape = go.Figure()
    fig_mape.add_trace(go.Scatter(
        x=metrics_df["date_calcul"],
        y=metrics_df["mape"],
        mode="lines+markers",
        name="MAPE",
        line=dict(color="blue", width=2),
        marker=dict(size=6),
    ))
    fig_mape.add_hline(y=baseline_mape, line_dash="dash", line_color="red",
                       annotation_text="Seuil d'alerte")
    fig_mape.update_layout(
        title="MAPE dans le temps",
        xaxis_title="Date",
        yaxis_title="MAPE (%)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig_mape, use_container_width=True)

    # Graphique MAE vs RMSE
    col1, col2 = st.columns(2)

    with col1:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=metrics_df["date_calcul"],
            y=metrics_df["mae"],
            mode="lines+markers",
            name="MAE",
            fill="tozeroy",
            line=dict(color="green"),
        ))
        fig_mae.update_layout(title="MAE (Erreur absolue moyenne)", height=350)
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Scatter(
            x=metrics_df["date_calcul"],
            y=metrics_df["rmse"],
            mode="lines+markers",
            name="RMSE",
            fill="tozeroy",
            line=dict(color="orange"),
        ))
        fig_rmse.update_layout(title="RMSE (Racine de l'erreur quadratique)", height=350)
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Tableau détaillé
    st.subheader("Données brutes")
    st.dataframe(
        metrics_df.sort_values("date_calcul", ascending=False),
        use_container_width=True,
    )
else:
    st.info("Pas de métriques enregistrées pour cette période.")

# === Prédictions récentes ===
st.header("🎯 Prédictions récentes")

preds = get_predictions(use_case, days=days_check, include_nulls=False)

if not preds.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total prédictions", len(preds))
        st.metric("Erreur moyenne", f"{preds['erreur_pct'].mean():.1f}%")

    with col2:
        st.metric("Erreur max", f"{preds['erreur_pct'].max():.1f}%")
        st.metric("Pics détectés", preds["alerte_pic"].sum())

    # Graphique erreurs
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=preds["date_prediction"],
        y=preds["erreur_pct"],
        mode="markers",
        name="Erreur %",
        marker=dict(
            size=8,
            color=preds["erreur_pct"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Erreur %"),
        ),
    ))
    fig_err.update_layout(
        title="Erreur de prédiction par jour",
        xaxis_title="Date",
        yaxis_title="Erreur (%)",
        height=400,
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # Tableau
    display_cols = ["date_prediction", "montant_predit", "montant_reel", "erreur_pct", "alerte_pic"]
    st.dataframe(
        preds[display_cols].head(20).sort_values("date_prediction", ascending=False),
        use_container_width=True,
    )
else:
    st.info("Pas de prédictions enregistrées.")

# === Alertes ===
st.header("🚨 Alertes récentes")

alerts = get_alerts(use_case, days=days_check)

if not alerts.empty:
    for _, alert in alerts.sort_values("timestamp", ascending=False).iterrows():
        emoji = "❌" if alert["type_alerte"] == "model_drift" else "⚠️"
        st.warning(f"{emoji} [{alert['timestamp'][:10]}] {alert['message']}")
else:
    st.success("✅ Aucune alerte récente")

# === Export ===
st.header("📥 Export")

if st.button("Exporter métriques en CSV"):
    from src.observability import export_metrics
    path = export_metrics(use_case)
    st.success(f"Exporté dans {path}")
