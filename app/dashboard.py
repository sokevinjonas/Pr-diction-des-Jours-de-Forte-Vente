"""
Dashboard Streamlit — Prévision des Jours de Forte Vente.
Fonctionne sans connexion internet (données locales).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

from src.utils import load_config, PROJECT_ROOT
from src.data_loader import load_ventes
from src.predict import predict_next_days
from src.model import train_pipeline, MODELS_DIR


st.set_page_config(
    page_title="Prevision des Ventes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS custom avec loading spinner stylé ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 4px solid #e8eaf6;
    border-top: 4px solid #1976d2;
    animation: spin 0.8s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-pulse {
    display: flex;
    gap: 8px;
    margin-top: 24px;
}

.loading-pulse span {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #1976d2;
    animation: pulse 1.4s ease-in-out infinite;
}

.loading-pulse span:nth-child(2) { animation-delay: 0.2s; }
.loading-pulse span:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

.loading-text {
    margin-top: 20px;
    font-size: 1.1rem;
    font-weight: 500;
    color: #546e7a;
    letter-spacing: 0.3px;
}

.loading-subtext {
    margin-top: 8px;
    font-size: 0.85rem;
    color: #90a4ae;
}

/* Progress bar custom */
.progress-bar-container {
    width: 280px;
    height: 4px;
    background: #e8eaf6;
    border-radius: 4px;
    margin-top: 24px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #1976d2, #42a5f5, #1976d2);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 4px;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Skeleton loading pour les cartes */
.skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: skeleton-loading 1.5s infinite;
    border-radius: 8px;
}

@keyframes skeleton-loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.alerte-rouge {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 4px;
}

.alerte-orange {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def show_loading(message="Calcul des previsions en cours", detail="Analyse des tendances et evenements..."):
    """Affiche un indicateur de chargement stylé."""
    return st.markdown(f"""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-pulse">
                <span></span><span></span><span></span>
            </div>
            <div class="loading-text">{message}</div>
            <div class="loading-subtext">{detail}</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_skeleton_cards():
    """Affiche des cartes skeleton pendant le chargement."""
    cols = st.columns(4)
    for col in cols:
        col.markdown("""
            <div style="padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0;">
                <div class="skeleton" style="height: 14px; width: 60%; margin-bottom: 12px;"></div>
                <div class="skeleton" style="height: 28px; width: 80%;"></div>
            </div>
        """, unsafe_allow_html=True)


def main():
    config = load_config()

    # --- Sidebar ---
    st.sidebar.title("Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Importer vos donnees",
        type=["csv", "xlsx", "xls", "json"],
        help="Formats : CSV, Excel, JSON. Le systeme detecte automatiquement les colonnes."
    )

    if uploaded_file is not None:
        st.sidebar.success(f"Fichier charge : {uploaded_file.name}")

    st.sidebar.divider()

    use_case = st.sidebar.selectbox(
        "Type de commerce (pour les recommandations)",
        ["supermarche", "restaurant", "mobile_money", "grossiste"],
        format_func=lambda x: {
            "supermarche": "Supermarche / Grande Surface",
            "restaurant": "Restaurant / Fast Food",
            "mobile_money": "Telephonie / Mobile Money",
            "grossiste": "Grossiste / Distributeur",
        }[x],
        help="Utilise uniquement pour adapter les recommandations (personnel, stock, etc.)"
    )

    horizon_jours = st.sidebar.slider("Horizon de prevision (jours)", 7, 30, 14)

    st.sidebar.divider()

    if st.sidebar.button("Reentrainer le modele"):
        with st.spinner("Entrainement en cours..."):
            _, metrics, _ = train_pipeline(use_case=use_case)
            st.sidebar.success(
                f"Modele entraine ! MAPE = {metrics['mape']:.1f}%"
            )

    # --- Vérifier que le modèle existe (sauf si fichier uploadé → auto-train) ---
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists() and uploaded_file is None:
        st.title("Prevision des Jours de Forte Vente")
        st.warning(
            "Aucun modele entraine. Importez un fichier de donnees ou cliquez sur "
            "'Reentrainer le modele' dans la barre laterale."
        )
        st.info(
            "Le systeme s'adapte automatiquement : importez n'importe quel CSV "
            "avec des dates et des montants, il se reentraine tout seul."
        )
        return

    # --- Page principale ---
    st.title("Prevision des Jours de Forte Vente")

    # --- Pré-charger les données uploadées (évite les problèmes de curseur file-like) ---
    df_uploaded = None
    if uploaded_file is not None:
        uploaded_file.seek(0)
        from src.data_loader import load_ventes as _load
        try:
            df_uploaded = _load(filepath=uploaded_file)
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            return

    loading_placeholder = st.empty()

    with loading_placeholder.container():
        if df_uploaded is not None:
            show_loading(
                message="Adaptation du modele a vos donnees",
                detail=f"Detection OK — {len(df_uploaded)} jours trouves. Entrainement en cours..."
            )
        else:
            show_loading(
                message="Calcul des previsions en cours",
                detail=f"Analyse de {horizon_jours} jours pour {use_case.replace('_', ' ')}..."
            )

    try:
        if df_uploaded is not None:
            df_forecast = predict_next_days(
                horizon=horizon_jours,
                use_case=use_case,
                df_input=df_uploaded,
            )
        else:
            df_forecast = predict_next_days(
                horizon=horizon_jours,
                use_case=use_case,
            )
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Erreur de prediction : {e}")
        return

    # Supprimer le loading une fois terminé
    loading_placeholder.empty()

    # --- KPIs en haut ---
    col1, col2, col3, col4 = st.columns(4)

    nb_alertes_rouge = (df_forecast["niveau_alerte"] == "ROUGE").sum()
    nb_alertes_orange = (df_forecast["niveau_alerte"] == "ORANGE").sum()
    ca_total_prevu = df_forecast["prediction"].sum()
    variation_moy = df_forecast["variation_pct"].mean()

    col1.metric("CA Total Prevu", f"{ca_total_prevu:,.0f} FCFA")
    col2.metric("Variation Moyenne", f"{variation_moy:+.1f}%")
    col3.metric("Alertes Rouges", nb_alertes_rouge)
    col4.metric("Alertes Oranges", nb_alertes_orange)

    st.divider()

    # --- Graphique principal ---
    col_graph, col_alertes = st.columns([3, 1])

    with col_graph:
        st.subheader("Previsions sur les prochains jours")

        color_map = {"NORMAL": "#4caf50", "ORANGE": "#ff9800", "ROUGE": "#f44336"}

        fig = px.bar(
            df_forecast,
            x="date",
            y="prediction",
            color="niveau_alerte",
            color_discrete_map=color_map,
            hover_data=["jour", "variation", "message"],
            labels={
                "prediction": "CA Prevu (FCFA)",
                "date": "Date",
                "niveau_alerte": "Alerte",
            },
        )
        fig.update_layout(
            xaxis_tickformat="%a %d/%m",
            yaxis_tickformat=",",
            showlegend=True,
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    with col_alertes:
        st.subheader("Alertes")
        alertes = df_forecast[df_forecast["niveau_alerte"] != "NORMAL"]
        if alertes.empty:
            st.success("Aucune alerte — semaine normale prevue.")
        else:
            for _, row in alertes.iterrows():
                if row["niveau_alerte"] == "ROUGE":
                    st.error(f"**{row['jour']} {row['date']}**\n\n{row['message']}")
                else:
                    st.warning(f"**{row['jour']} {row['date']}**\n\n{row['message']}")

    # --- Tableau détaillé ---
    st.subheader("Detail des previsions")

    df_display = df_forecast[["date", "jour", "prediction", "variation", "niveau_alerte", "message"]].copy()
    df_display.columns = ["Date", "Jour", "CA Prevu (FCFA)", "Variation", "Alerte", "Recommandation"]

    st.dataframe(
        df_display.style.apply(
            lambda row: [
                "background-color: #ffebee" if row["Alerte"] == "ROUGE"
                else "background-color: #fff3e0" if row["Alerte"] == "ORANGE"
                else "" for _ in row
            ],
            axis=1,
        ),
        width="stretch",
        hide_index=True,
    )

    # --- Historique ---
    st.divider()
    st.subheader("Historique des ventes")

    df_hist = df_uploaded if df_uploaded is not None else load_ventes(use_case=use_case)
    jours_hist = st.slider("Nombre de jours d'historique", 30, min(len(df_hist), 365), min(90, len(df_hist)))
    df_hist_recent = df_hist.tail(jours_hist)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_hist_recent["date"],
        y=df_hist_recent["montant_total"],
        mode="lines",
        name="Ventes reelles",
        line=dict(color="#1976d2"),
    ))

    df_hist_recent = df_hist_recent.copy()
    df_hist_recent["moy_7j"] = df_hist_recent["montant_total"].rolling(7).mean()
    fig_hist.add_trace(go.Scatter(
        x=df_hist_recent["date"],
        y=df_hist_recent["moy_7j"],
        mode="lines",
        name="Moyenne 7 jours",
        line=dict(color="#ff9800", dash="dash"),
    ))

    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="Montant (FCFA)",
        yaxis_tickformat=",",
        height=350,
    )
    st.plotly_chart(fig_hist, width="stretch")

    # --- Export ---
    st.divider()
    st.subheader("Exporter les previsions")

    col_excel, col_csv = st.columns(2)

    with col_excel:
        buffer = BytesIO()
        df_display.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Telecharger en Excel",
            data=buffer.getvalue(),
            file_name=f"previsions_{use_case}_{horizon_jours}j.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col_csv:
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Telecharger en CSV",
            data=csv_data,
            file_name=f"previsions_{use_case}_{horizon_jours}j.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
