"""Fonctions utilitaires partagées."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    """Charge la configuration depuis config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_seuils():
    """Retourne les seuils d'alerte."""
    config = load_config()
    return config["seuils_alerte"]


def niveau_alerte(variation_pct, seuils=None):
    """Détermine le niveau d'alerte à partir d'une variation en %."""
    if seuils is None:
        seuils = get_seuils()
    if variation_pct >= seuils["critique"]:
        return "ROUGE"
    elif variation_pct >= seuils["attention"]:
        return "ORANGE"
    return "NORMAL"
