"""
Tests pour le pipeline ML de prédiction des ventes.
Couvre : génération de données, features, modèle, prédictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_ventes, load_evenements
from src.feature_engineering import (
    add_features_temporelles,
    add_features_evenements,
    add_features_lag,
    build_features,
    get_feature_columns,
)
from src.utils import niveau_alerte
from data.synthetic.generate_data import generate_sales_data, generate_calendrier_evenements


class TestGenerationDonnees:
    """Tests pour la génération de données synthétiques."""

    def test_generation_supermarche(self):
        df = generate_sales_data(use_case="supermarche", start_date="2023-01-01", end_date="2023-12-31")
        assert len(df) == 365
        assert "date" in df.columns
        assert "montant_total" in df.columns
        assert df["montant_total"].min() > 0

    def test_generation_tous_use_cases(self):
        for uc in ["supermarche", "restaurant", "mobile_money", "grossiste"]:
            df = generate_sales_data(use_case=uc, start_date="2023-06-01", end_date="2023-06-30")
            assert len(df) == 30
            assert (df["use_case"] == uc).all()

    def test_effet_debut_mois(self):
        df = generate_sales_data(use_case="supermarche", start_date="2023-01-01", end_date="2023-12-31")
        df["date"] = pd.to_datetime(df["date"])
        df["jour_mois"] = df["date"].dt.day

        moy_debut = df[df["jour_mois"] <= 5]["montant_total"].mean()
        moy_milieu = df[(df["jour_mois"] > 5) & (df["jour_mois"] < 26)]["montant_total"].mean()

        # Le début de mois doit être significativement plus élevé
        assert moy_debut > moy_milieu * 1.2

    def test_calendrier_evenements(self):
        df = generate_calendrier_evenements()
        assert len(df) > 0
        assert "evenement" in df.columns
        assert "tabaski" in df["evenement"].values


class TestFeatureEngineering:
    """Tests pour le feature engineering."""

    @pytest.fixture
    def df_sample(self):
        df = generate_sales_data(
            use_case="supermarche",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_features_temporelles(self, df_sample):
        df = add_features_temporelles(df_sample)
        assert "jour_semaine" in df.columns
        assert "est_weekend" in df.columns
        assert "mois_sin" in df.columns
        assert df["jour_semaine"].between(0, 6).all()
        assert df["mois"].between(1, 12).all()

    def test_features_lag(self, df_sample):
        df = add_features_lag(df_sample)
        # Les 7 premières lignes doivent avoir des NaN pour ventes_j_7
        assert df["ventes_j_7"].isna().sum() == 7
        # Après les NaN, les valeurs doivent être positives
        assert (df["ventes_j_7"].dropna() > 0).all()

    def test_features_lag_pas_de_fuite(self, df_sample):
        """Vérifie qu'il n'y a pas de data leakage (pas de valeurs futures)."""
        df = add_features_lag(df_sample)
        # ventes_j_1 à l'index i doit être le montant à l'index i-1
        for i in range(1, min(10, len(df))):
            expected = df_sample.iloc[i - 1]["montant_total"]
            actual = df.iloc[i]["ventes_j_1"]
            assert abs(actual - expected) < 0.01

    def test_build_features_complet(self, df_sample):
        df = build_features(df_sample)
        feature_cols = get_feature_columns()
        # Vérifier que la majorité des features attendues sont présentes
        present = [c for c in feature_cols if c in df.columns]
        assert len(present) >= 20

    def test_pas_de_nan_apres_build(self, df_sample):
        df = build_features(df_sample)
        feature_cols = [c for c in get_feature_columns() if c in df.columns]
        # Après dropna dans build_features, pas de NaN dans les features principales
        assert df[feature_cols].isna().sum().sum() == 0


class TestUtils:
    """Tests pour les utilitaires."""

    def test_niveau_alerte_normal(self):
        assert niveau_alerte(0.10) == "NORMAL"

    def test_niveau_alerte_orange(self):
        assert niveau_alerte(0.35) == "ORANGE"

    def test_niveau_alerte_rouge(self):
        assert niveau_alerte(0.70) == "ROUGE"

    def test_niveau_alerte_seuils_custom(self):
        seuils = {"attention": 0.20, "critique": 0.50}
        assert niveau_alerte(0.25, seuils) == "ORANGE"
        assert niveau_alerte(0.55, seuils) == "ROUGE"


class TestDataLoader:
    """Tests pour le chargement des données (nécessite les CSV générés)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Crée un CSV temporaire pour les tests."""
        self.csv_path = tmp_path / "test_ventes.csv"
        df = generate_sales_data(
            use_case="supermarche",
            start_date="2023-01-01",
            end_date="2023-03-31"
        )
        df.to_csv(self.csv_path, index=False)

    def test_load_csv(self):
        df = load_ventes(filepath=self.csv_path)
        assert len(df) == 90
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_load_csv_colonnes_manquantes(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            load_ventes(filepath=bad_csv)

    def test_dates_triees(self):
        df = load_ventes(filepath=self.csv_path)
        assert (df["date"].diff().dropna() >= pd.Timedelta(0)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
