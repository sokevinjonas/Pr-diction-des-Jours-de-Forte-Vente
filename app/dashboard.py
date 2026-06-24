"""
Dashboard Streamlit — Prévision des Jours de Forte Vente.
Interface moderne avec mapping interactif des colonnes.
"""

import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from src.utils import load_config, PROJECT_ROOT
from src.data_loader import load_ventes, _try_read_file, _detect_date_column, _detect_montant_column
from src.predict import predict_next_days
from src.model import train_pipeline, MODELS_DIR

# --- Logger ---
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "errors.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard")


def log_error(context, exception):
    tb = traceback.format_exc()
    logger.error(f"[{context}] {type(exception).__name__}: {exception}\n{tb}")


st.set_page_config(
    page_title="Prevision des Ventes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #1565c0;
    --primary-light: #42a5f5;
    --success: #2e7d32;
    --warning: #f57c00;
    --danger: #c62828;
    --bg-card: #ffffff;
    --bg-subtle: #f8fafc;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --radius: 12px;
}

.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f1f5f9;
}

/* Header custom */
.dashboard-header {
    background: linear-gradient(135deg, var(--primary) 0%, #0d47a1 100%);
    padding: 28px 32px;
    border-radius: var(--radius);
    margin-bottom: 24px;
    color: white;
}

.dashboard-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}

.dashboard-header p {
    margin: 6px 0 0;
    font-size: 0.9rem;
    opacity: 0.85;
}

/* KPI Cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 6px;
}

.kpi-value.danger { color: var(--danger); }
.kpi-value.warning { color: var(--warning); }
.kpi-value.success { color: var(--success); }

/* Mapping modal */
.mapping-container {
    background: var(--bg-card);
    border: 2px solid var(--primary-light);
    border-radius: var(--radius);
    padding: 24px;
    margin: 16px 0;
}

.mapping-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}

.mapping-header h3 {
    margin: 0;
    font-size: 1.1rem;
    color: var(--text-primary);
}

.mapping-badge {
    background: #e3f2fd;
    color: var(--primary);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.col-preview {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Loading */
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
}

.loading-spinner {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    border: 3px solid #e8eaf6;
    border-top: 3px solid var(--primary);
    animation: spin 0.8s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-pulse {
    display: flex;
    gap: 6px;
    margin-top: 20px;
}

.loading-pulse span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    animation: pulse 1.4s ease-in-out infinite;
}

.loading-pulse span:nth-child(2) { animation-delay: 0.2s; }
.loading-pulse span:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

.loading-text {
    margin-top: 16px;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.loading-subtext {
    margin-top: 6px;
    font-size: 0.82rem;
    color: #90a4ae;
}

.progress-bar-container {
    width: 240px;
    height: 3px;
    background: #e8eaf6;
    border-radius: 3px;
    margin-top: 20px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--primary-light), var(--primary));
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Section cards */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Alert badges */
.alert-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.alert-badge.rouge { background: #ffebee; color: var(--danger); }
.alert-badge.orange { background: #fff3e0; color: var(--warning); }
.alert-badge.normal { background: #e8f5e9; color: var(--success); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
}

/* ===== MODE SIMPLE ===== */

.simple-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    padding: 24px;
    border-radius: var(--radius);
    margin-bottom: 24px;
    color: white;
    text-align: center;
}

.simple-header h1 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 700;
}

.simple-header p {
    margin: 8px 0 0;
    font-size: 0.95rem;
    opacity: 0.9;
}

/* Calendrier visuel */
.day-card {
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 12px;
    transition: transform 0.2s;
    border: 2px solid transparent;
}

.day-card:hover {
    transform: scale(1.02);
}

.day-card.vert {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-color: #a5d6a7;
}

.day-card.orange {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    border-color: #ffcc80;
}

.day-card.rouge {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    border-color: #ef9a9a;
}

.day-icon {
    font-size: 2.8rem;
    margin-bottom: 8px;
}

.day-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.day-date {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 10px;
}

.day-message {
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.4;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(255,255,255,0.7);
}

.day-action {
    margin-top: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
}

.day-action.vert { background: #c8e6c9; color: #1b5e20; }
.day-action.orange { background: #ffe0b2; color: #e65100; }
.day-action.rouge { background: #ffcdd2; color: #b71c1c; }

/* Résumé simple */
.simple-summary {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin: 20px 0;
    text-align: center;
}

.simple-summary h2 {
    font-size: 1.2rem;
    margin: 0 0 12px;
    color: var(--text-primary);
}

.simple-summary .big-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--primary);
}

.simple-summary .caption {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# COMPOSANTS UI
# =============================================================================

def show_header():
    st.markdown("""
        <div class="dashboard-header">
            <h1>Prevision des Jours de Forte Vente</h1>
            <p>Anticipez vos pics de vente et optimisez votre stock, personnel et logistique</p>
        </div>
    """, unsafe_allow_html=True)


def show_kpi(label, value, style=""):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {style}">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def show_loading(message, detail=""):
    st.markdown(f"""
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


def show_simple_view(df_forecast, use_case, config):
    """
    Mode Simple — Vue visuelle pour commerçants.
    Calendrier avec feux tricolores, icônes, messages courts.
    """
    # Header simple
    st.markdown("""
        <div class="simple-header">
            <h1>Vos Ventes Cette Semaine</h1>
            <p>Les jours en rouge = beaucoup de clients. Preparez-vous !</p>
        </div>
    """, unsafe_allow_html=True)

    # Résumé en gros
    nb_jours_forts = int((df_forecast["niveau_alerte"] != "NORMAL").sum())
    total_jours = len(df_forecast)

    if nb_jours_forts == 0:
        resume_icon = "&#9996;"
        resume_text = "Semaine tranquille"
        resume_detail = "Pas de jour exceptionnel prevu. Fonctionnez normalement."
    elif nb_jours_forts <= 2:
        resume_icon = "&#9888;"
        resume_text = f"{nb_jours_forts} jour(s) fort(s)"
        resume_detail = "Preparez du stock supplementaire pour ces jours."
    else:
        resume_icon = "&#128293;"
        resume_text = f"{nb_jours_forts} jours forts !"
        resume_detail = "Semaine chargee. Appelez du personnel en renfort."

    st.markdown(f"""
        <div class="simple-summary">
            <div style="font-size: 3rem;">{resume_icon}</div>
            <h2>{resume_text}</h2>
            <div class="caption">{resume_detail}</div>
        </div>
    """, unsafe_allow_html=True)

    # Calendrier visuel — une carte par jour
    # Afficher 7 jours max dans une grille
    jours_a_afficher = min(7, len(df_forecast))
    cols = st.columns(jours_a_afficher)

    for i, (_, row) in enumerate(df_forecast.head(jours_a_afficher).iterrows()):
        alerte = row["niveau_alerte"]

        if alerte == "ROUGE":
            css_class = "rouge"
            icon = "&#128680;"  # sirène
            message = "Beaucoup de monde !"
            action = "Preparez plus"
        elif alerte == "ORANGE":
            css_class = "orange"
            icon = "&#128200;"  # graphique hausse
            message = "Plus que d'habitude"
            action = "Verifiez le stock"
        else:
            css_class = "vert"
            icon = "&#9989;"  # check vert
            message = "Journee normale"
            action = "Tout est bon"

        with cols[i]:
            st.markdown(f"""
                <div class="day-card {css_class}">
                    <div class="day-icon">{icon}</div>
                    <div class="day-name">{row['jour']}</div>
                    <div class="day-date">{row['date']}</div>
                    <div class="day-message">{message}</div>
                    <div class="day-action {css_class}">{action}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Conseils concrets en dessous
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Ce que vous devez faire")

    alertes = df_forecast[df_forecast["niveau_alerte"] != "NORMAL"]

    if alertes.empty:
        st.markdown("""
            <div style="text-align: center; padding: 20px; color: var(--success);">
                <p style="font-size: 1.1rem;">Rien de special a preparer cette semaine.</p>
                <p>Gardez votre stock habituel.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        uc_config = config["use_cases"].get(use_case, {})

        for _, row in alertes.iterrows():
            reco = row.get("recommandations", {})
            if isinstance(reco, str):
                import ast
                try:
                    reco = ast.literal_eval(reco)
                except:
                    reco = {}

            conseils = []

            if use_case == "supermarche":
                conseils.append("Commandez plus de marchandise pour ce jour")
                if reco.get("personnel_extra", 0) > 0:
                    conseils.append(f"Appelez {reco['personnel_extra']} personnes en plus")
                if reco.get("caisses_recommandees"):
                    conseils.append(f"Ouvrez {reco['caisses_recommandees']} caisses")

            elif use_case == "restaurant":
                conseils.append("Preparez plus de nourriture")
                if reco.get("couverts_prevus"):
                    conseils.append(f"Prevoyez {reco['couverts_prevus']} couverts")
                if reco.get("personnel_extra", 0) > 0:
                    conseils.append(f"Appelez {reco['personnel_extra']} extras")

            elif use_case == "mobile_money":
                conseils.append("Gardez plus d'argent en caisse (float)")
                if reco.get("float_recommande"):
                    conseils.append(f"Float recommande : {reco['float_recommande']:,.0f} FCFA")

            elif use_case == "grossiste":
                conseils.append("Prevoyez plus de livraisons")
                if reco.get("camions_recommandes"):
                    conseils.append(f"Utilisez {reco['camions_recommandes']} camions")

            if not conseils:
                conseils.append("Preparez plus de stock que d'habitude")

            badge_color = "#ffebee" if row["niveau_alerte"] == "ROUGE" else "#fff3e0"
            emoji = "&#128308;" if row["niveau_alerte"] == "ROUGE" else "&#128992;"

            st.markdown(f"""
                <div style="background: {badge_color}; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="font-weight: 700; font-size: 1rem;">
                        {emoji} {row['jour']} {row['date']}
                    </div>
                    <ul style="margin: 8px 0 0; padding-left: 20px;">
                        {''.join(f'<li style="margin: 4px 0; font-size: 0.95rem;">{c}</li>' for c in conseils)}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def show_column_mapping(df_raw):
    """
    Affiche un dialog interactif de mapping des colonnes.
    L'utilisateur peut associer ses colonnes aux champs attendus.
    Retourne le DataFrame renommé ou None si annulé.
    """
    st.markdown("""
        <div class="mapping-container">
            <div class="mapping-header">
                <h3>Configuration des colonnes</h3>
                <span class="mapping-badge">Mapping requis</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.info(
        "Le systeme n'a pas pu detecter automatiquement vos colonnes. "
        "Associez vos colonnes aux champs requis ci-dessous."
    )

    cols_fichier = list(df_raw.columns)

    # Aperçu des données
    st.markdown("**Apercu de votre fichier :**")
    st.dataframe(df_raw.head(5), width="stretch", hide_index=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Vos colonnes**")
        for c in cols_fichier:
            sample = df_raw[c].dropna().head(3).tolist()
            st.markdown(f"""
                <div class="col-preview">
                    <strong>{c}</strong> — ex: {', '.join(str(s) for s in sample[:3])}
                </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("**Champs requis par le modele**")

        date_col = st.selectbox(
            "Colonne DATE (obligatoire)",
            options=["-- Auto-detection --"] + cols_fichier,
            help="Colonne contenant les dates de vente"
        )

        montant_col = st.selectbox(
            "Colonne MONTANT / VENTES (obligatoire)",
            options=["-- Auto-detection --"] + cols_fichier,
            help="Colonne contenant le chiffre d'affaires ou montant total"
        )

        trans_col = st.selectbox(
            "Colonne NB TRANSACTIONS (optionnel)",
            options=["-- Ignorer --", "-- Auto-detection --"] + cols_fichier,
            help="Nombre de transactions par jour (sera estime si absent)"
        )

    # Bouton de validation
    if st.button("Valider le mapping", type="primary", use_container_width=True):
        rename_map = {}

        if date_col != "-- Auto-detection --":
            rename_map[date_col] = "date"
        if montant_col != "-- Auto-detection --":
            rename_map[montant_col] = "montant_total"
        if trans_col not in ("-- Ignorer --", "-- Auto-detection --"):
            rename_map[trans_col] = "nb_transactions"

        if rename_map:
            df_renamed = df_raw.rename(columns=rename_map)
            st.session_state["manual_mapping"] = rename_map
            return df_renamed
        else:
            return df_raw

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Mode d'affichage")
        mode = st.radio(
            "Choisir le mode",
            ["Simple", "Expert"],
            horizontal=True,
            help="Simple = vue visuelle pour commercants | Expert = tableaux et graphiques detailles",
        )

        st.markdown("---")
        st.markdown("### Donnees")
        uploaded_file = st.file_uploader(
            "Importer vos donnees",
            type=["csv", "xlsx", "xls", "json"],
            help="CSV, Excel ou JSON. Detection automatique des colonnes.",
        )

        if uploaded_file:
            st.success(f"{uploaded_file.name}")

        st.markdown("---")
        st.markdown("### Parametres")

        use_case = st.selectbox(
            "Type de commerce",
            ["supermarche", "restaurant", "mobile_money", "grossiste"],
            format_func=lambda x: {
                "supermarche": "Supermarche",
                "restaurant": "Restaurant",
                "mobile_money": "Mobile Money",
                "grossiste": "Grossiste",
            }[x],
            help="Adapte les recommandations (personnel, stock, caisses...)"
        )

        horizon_jours = st.slider("Horizon (jours)", 7, 30, 14)

        st.markdown("---")
        st.markdown("### Actions")

        if st.button("Reentrainer le modele", use_container_width=True):
            with st.spinner("Entrainement..."):
                _, metrics, _ = train_pipeline(use_case=use_case)
                st.success(f"MAPE = {metrics['mape']:.1f}%")

    # --- Header ---
    show_header()

    # --- Chargement des données ---
    df_uploaded = None

    if uploaded_file is not None:
        uploaded_file.seek(0)
        try:
            df_uploaded = load_ventes(filepath=uploaded_file)
        except Exception as e:
            # Détection automatique échouée → afficher le mapping interactif
            log_error("import_auto", e)

            uploaded_file.seek(0)
            try:
                df_raw = _try_read_file(uploaded_file)
            except Exception as e2:
                log_error("import_read", e2)
                st.error(f"Impossible de lire le fichier : {e2}")
                return

            st.warning(
                f"Detection automatique impossible : {e}. "
                f"Configurez le mapping manuellement ci-dessous."
            )

            df_mapped = show_column_mapping(df_raw)

            if df_mapped is None:
                return

            # Retenter avec le mapping manuel
            try:
                df_uploaded = load_ventes(filepath=None)
                # Construire manuellement à partir du mapping
                result = pd.DataFrame()
                if "date" in df_mapped.columns:
                    result["date"] = pd.to_datetime(df_mapped["date"], errors="coerce")
                if "montant_total" in df_mapped.columns:
                    result["montant_total"] = pd.to_numeric(df_mapped["montant_total"], errors="coerce")
                if "nb_transactions" in df_mapped.columns:
                    result["nb_transactions"] = pd.to_numeric(df_mapped["nb_transactions"], errors="coerce")
                else:
                    result["nb_transactions"] = (result["montant_total"] / 3500).clip(lower=1).astype(int)

                result = result.dropna(subset=["date", "montant_total"])
                result = result[result["montant_total"] > 0]
                result = result.sort_values("date").reset_index(drop=True)

                # Agréger si transactionnel
                nb_dates = result["date"].dt.date.nunique()
                if len(result) > nb_dates * 1.5:
                    result = result.groupby(result["date"].dt.date).agg(
                        montant_total=("montant_total", "sum"),
                        nb_transactions=("montant_total", "count"),
                    ).reset_index()
                    result["date"] = pd.to_datetime(result["date"])

                df_uploaded = result
            except Exception as e3:
                log_error("import_manual", e3)
                st.error(f"Erreur apres mapping : {e3}")
                return

    # --- Vérifier qu'on a un modèle ou des données ---
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists() and df_uploaded is None:
        st.markdown("""
            <div class="section-card" style="text-align: center; padding: 60px;">
                <h3 style="color: var(--text-secondary);">Bienvenue</h3>
                <p style="color: var(--text-secondary); max-width: 500px; margin: 12px auto;">
                    Importez un fichier de ventes (CSV, Excel, JSON) via la barre laterale
                    pour commencer. Le systeme detecte automatiquement vos colonnes et
                    s'entraine sur vos donnees.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    # --- Loading + Prédiction ---
    loading_placeholder = st.empty()

    with loading_placeholder.container():
        if df_uploaded is not None:
            show_loading(
                "Adaptation du modele a vos donnees",
                f"{len(df_uploaded)} jours detectes — entrainement automatique..."
            )
        else:
            show_loading(
                "Calcul des previsions",
                f"{horizon_jours} jours pour {use_case.replace('_', ' ')}"
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
        log_error("prediction", e)
        loading_placeholder.empty()
        st.error(f"Erreur de prediction : {e}")
        return

    loading_placeholder.empty()

    # --- Aiguillage : Mode Simple ou Expert ---
    if mode == "Simple":
        show_simple_view(df_forecast, use_case, config)
        return

    # ===== MODE EXPERT (ci-dessous) =====

    # --- KPIs ---
    nb_alertes_rouge = int((df_forecast["niveau_alerte"] == "ROUGE").sum())
    nb_alertes_orange = int((df_forecast["niveau_alerte"] == "ORANGE").sum())
    ca_total_prevu = df_forecast["prediction"].sum()
    variation_moy = df_forecast["variation_pct"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_kpi("CA Total Prevu", f"{ca_total_prevu:,.0f} FCFA")
    with col2:
        style = "success" if variation_moy >= 0 else "danger"
        show_kpi("Variation Moyenne", f"{variation_moy:+.1f}%", style)
    with col3:
        style = "danger" if nb_alertes_rouge > 0 else ""
        show_kpi("Alertes Rouges", str(nb_alertes_rouge), style)
    with col4:
        style = "warning" if nb_alertes_orange > 0 else ""
        show_kpi("Alertes Oranges", str(nb_alertes_orange), style)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Graphique + Alertes ---
    col_graph, col_alertes = st.columns([3, 1])

    with col_graph:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Previsions")

        color_map = {"NORMAL": "#4caf50", "ORANGE": "#ff9800", "ROUGE": "#f44336"}

        fig = px.bar(
            df_forecast,
            x="date", y="prediction",
            color="niveau_alerte",
            color_discrete_map=color_map,
            hover_data=["jour", "variation", "message"],
            labels={"prediction": "CA Prevu (FCFA)", "date": "Date", "niveau_alerte": "Alerte"},
        )
        fig.update_layout(
            xaxis_tickformat="%a %d/%m",
            yaxis_tickformat=",",
            showlegend=True,
            height=380,
            margin=dict(t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(gridcolor="#f0f0f0")
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_alertes:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Alertes")
        alertes = df_forecast[df_forecast["niveau_alerte"] != "NORMAL"]
        if alertes.empty:
            st.markdown("""
                <div style="text-align:center; padding: 30px 10px; color: var(--success);">
                    <div style="font-size: 2rem;">&#10003;</div>
                    <p>Aucune alerte<br>Semaine normale</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            for _, row in alertes.iterrows():
                badge_class = "rouge" if row["niveau_alerte"] == "ROUGE" else "orange"
                st.markdown(f"""
                    <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px;
                         background: {'#ffebee' if badge_class == 'rouge' else '#fff3e0'};">
                        <span class="alert-badge {badge_class}">{row['niveau_alerte']}</span>
                        <div style="margin-top: 6px; font-weight: 600; font-size: 0.9rem;">
                            {row['jour']} {row['date']}
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                            {row['message']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tableau ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Historique ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Historique des ventes")

    df_hist = df_uploaded if df_uploaded is not None else load_ventes(use_case=use_case)
    max_hist = min(len(df_hist), 365)
    jours_hist = st.slider("Jours d'historique", 30, max_hist, min(90, max_hist))
    df_hist_recent = df_hist.tail(jours_hist).copy()

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_hist_recent["date"],
        y=df_hist_recent["montant_total"],
        mode="lines",
        name="Ventes reelles",
        line=dict(color="#1565c0", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(21,101,192,0.05)",
    ))

    df_hist_recent["moy_7j"] = df_hist_recent["montant_total"].rolling(7).mean()
    fig_hist.add_trace(go.Scatter(
        x=df_hist_recent["date"],
        y=df_hist_recent["moy_7j"],
        mode="lines",
        name="Moyenne 7 jours",
        line=dict(color="#ff9800", dash="dash", width=2),
    ))

    fig_hist.update_layout(
        xaxis_title="", yaxis_title="Montant (FCFA)",
        yaxis_tickformat=",",
        height=320,
        margin=dict(t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_hist.update_xaxes(gridcolor="#f0f0f0")
    fig_hist.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig_hist, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Export ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Exporter")

    col_excel, col_csv = st.columns(2)
    with col_excel:
        buffer = BytesIO()
        df_display.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Telecharger Excel",
            data=buffer.getvalue(),
            file_name=f"previsions_{use_case}_{horizon_jours}j.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_csv:
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Telecharger CSV",
            data=csv_data,
            file_name=f"previsions_{use_case}_{horizon_jours}j.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
