# Phase 7 — Sécurité du modèle ML

## Vue d'ensemble

Système de sécurité multi-couches pour protéger le modèle en production contre :
- **Injections** (SQL, code)
- **Anomalies** (données suspectes, prédictions impossibles)
- **Abus** (DoS, spam)
- **Violations de conformité** (audit trail complet)

---

## 1. **security.py** — Validation et filtering

### InputValidator

Valide les features avant prédiction :

```python
from src.security import InputValidator

validator = InputValidator("supermarche")

# Valider les features
is_valid, error = validator.validate_features({
    "jour_semaine": 3,
    "jour_mois": 15,
    "mois": 6,
})

if not is_valid:
    print(f"Erreur: {error}")
    return
```

**Vérifications** :
- Types des features (doit être numérique)
- Ranges autorisés (jour_semaine 0-6, jour_mois 1-31, mois 1-12)
- Injection de code/SQL (blacklist : SELECT, INSERT, DELETE, EXEC, IMPORT)

### Détection d'anomalies

Identifie les données suspectes ou les tentatives d'exploitation :

```python
anomaly_detected, reason = validator.detect_anomaly({
    "temperature_max": 65,  # Impossible en Afrique
    "precipitation_mm": 600,  # Anomalement élevé
    "ventes_j_1": 50_000_000,  # 10x la moyenne
})

if anomaly_detected:
    log_security_event("anomaly", "warning", reason)
```

**Seuils** :
- Température : [-50, 60]°C
- Précipitation : [0, 500]mm
- Ventes : moyenne ±3σ

### OutputFilter

Valide les prédictions avant exposition :

```python
from src.security import OutputFilter

filter = OutputFilter("supermarche", historical_mean=1_000_000)

prediction, is_valid, warning = filter.filter_prediction(pred_value)

if not is_valid:
    st.error(f"Prédiction invalide: {warning}")
elif warning:
    st.warning(f"Attention: {warning}")
```

**Sécurité** :
- Reject : NaN, Inf, valeurs négatives
- Clamp : min=1% moyenne, max=10x moyenne
- Alerte : si clamping appliqué

---

## 2. **audit.py** — Audit trail et conformité

Enregistre TOUS les accès et prédictions pour audit légal :

```python
from src.audit import log_prediction_audit, log_security_event

# Avant chaque prédiction
log_prediction_audit(
    use_case="supermarche",
    prediction=1_500_000,
    validation_status="validated",
    security_check_passed=True,
    anomaly_detected=False,
    model_version="v2",
    user_id="user_123",
)

# En cas d'événement de sécurité
log_security_event(
    event_type="injection_attempt",
    severity="critical",
    description="SQL injection detected in features",
    user_id="attacker_ip",
    context={"endpoint": "/predict"},
)
```

### Tables d'audit

**prediction_audit** :
- Quand : timestamp
- Quoi : use_case, model_version
- Qui : user_id
- Résultat : prediction, validation_status, security_check_passed

**security_events** :
- event_type : rate_limit_exceeded, injection_attempt, anomaly_detected, etc
- severity : info, warning, critical
- Complète traçabilité pour conformité RGPD/CCPA

### Export de conformité

```python
from src.audit import export_compliance_report

report_path = export_compliance_report()
# Génère : compliance_report_20260624.json avec tous les logs
```

---

## 3. **rate_limit.py** — Protection contre les abus

Rate limiting pour chaque endpoint :

```python
from src.rate_limit import RATE_LIMITER_PREDICT

# Dans l'API
user_id = request.user_id or request.ip
allowed, reason = RATE_LIMITER_PREDICT.check_rate_limit(user_id, "predict")

if not allowed:
    return {"error": reason}, 429  # HTTP 429 Too Many Requests
```

### Configurations pré-définies

| Limiter | Limite | Fenêtre |
|---------|--------|---------|
| **RATE_LIMITER_API** | 1000 req | 1 heure |
| **RATE_LIMITER_PREDICT** | 100 req | 1 minute |
| **RATE_LIMITER_ADMIN** | 50 req | 1 heure |

### Comportement

- **Fenêtre glissante** : reset toutes les N secondes
- **Blocage progressif** : chaque dépassement logger comme événement sécurité
- **Cleanup auto** : vieux records supprimés après 30 jours

---

## 4. **Test de sécurité complet**

```bash
python3 test_security.py
```

Résultats :

```
✅ Validation des entrées : 5/5 tests passés
✅ Détection d'anomalies : 5/5 tests passés
✅ Filtering de sortie : 5/5 tests passés
✅ Rate limiting : 100/100 autorisées, 50/50 bloquées
✅ Audit logging : 1 audit log, 51 security events
```

---

## 5. Intégration dans le pipeline de production

### Étape 1 : Réception de la requête

```python
# API endpoint
@app.post("/predict")
def predict(request):
    # Rate limiting
    allowed, reason = RATE_LIMITER_PREDICT.check_rate_limit(
        request.user_id or request.client_ip,
        "predict"
    )
    if not allowed:
        log_security_event("rate_limit_exceeded", "warning", reason)
        return {"error": reason}, 429

    log_api_access("/predict", "POST", 200, response_time_ms, user_id=request.user_id)
```

### Étape 2 : Validation des entrées

```python
    validator = InputValidator(use_case)

    # Vérifier les features
    is_valid, error = validator.validate_features(features)
    if not is_valid:
        log_security_event("invalid_input", "warning", error, user_id=request.user_id)
        return {"error": error}, 400

    # Détecter les anomalies
    anomaly, reason = validator.detect_anomaly(features)
    if anomaly:
        log_security_event("anomaly_detected", "warning", reason)
        # Peut retourner une alerte ou rejeter la prédiction
```

### Étape 3 : Prédiction

```python
    prediction = model.predict(features)
```

### Étape 4 : Validation de la sortie

```python
    output_filter = OutputFilter(use_case)
    prediction, is_valid, warning = output_filter.filter_prediction(prediction)

    if not is_valid:
        log_security_event("invalid_prediction", "critical", "Model produced invalid output")
        return {"error": "Prédiction invalide"}, 500

    if warning:
        log_security_event("prediction_warning", "info", warning)
```

### Étape 5 : Audit logging

```python
    log_prediction_audit(
        use_case=use_case,
        prediction=prediction,
        validation_status="validated",
        security_check_passed=True,
        anomaly_detected=anomaly,
        model_version="v2",
        user_id=request.user_id,
    )

    return {"prediction": prediction, "unit": "FCFA", "confidence": 0.85}
```

---

## 6. Seuils de sécurité

| Menace | Seuil | Action |
|--------|-------|--------|
| **Injection SQL/Code** | 1 tentative | Bloc immédiat + log critique |
| **Anomalie de données** | Détection | Log + alerte UI |
| **Prédiction invalide** | 1 cas | Log critique + rejeter prédiction |
| **Rate limit** | 100 req/min | 429 Too Many Requests |
| **Abus de rate limit** | 5 violations/jour | IP blacklist temporaire |

---

## 7. Conformité légale

### RGPD (protection des données)

- ✅ Audit trail complet (quand, qui, quoi)
- ✅ Export de données possible (`export_compliance_report()`)
- ✅ Droit à l'oubli : supprimer les logs anciens de > 30 jours
- ✅ Pas de stockage de données sensibles

### CCPA (transparence)

- ✅ User_id enregistré à chaque prédiction
- ✅ Raison des décisions loggée
- ✅ Rapport d'audit générable

---

## 8. Monitoring de sécurité

### Dashboard sécurité (à implémenter dans monitoring_dashboard.py)

```
Anomalies détectées : 3 (7j)
Rate limits dépassées : 2 (24h)
Tentatives injection : 0 (30j)
Prédictions bloquées : 1 (24h)
```

### Alertes

Notifier les admins si :
- Anomalies > 10/jour
- Rate limits dépassées > 5/jour
- Injection attempt → alerte immédiate
- Model output invalide → alerte immédiate

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `src/security.py` | InputValidator, OutputFilter |
| `src/audit.py` | Logging audit, compliance |
| `src/rate_limit.py` | Rate limiting & protection DoS |
| `test_security.py` | Test du système |
| `data/processed/audit.db` | Base audit SQLite |

---

## Checklist Phase 7 ✓

- [x] Validation des entrées
- [x] Détection d'anomalies
- [x] Filtering de sortie
- [x] Rate limiting
- [x] Audit trail
- [x] Export de conformité
- [x] Tests de sécurité
- [ ] Dashboard de monitoring sécurité (optionnel)
- [ ] Notifications Slack en cas d'alerte (optionnel)
- [ ] Blacklist d'IP pour abus répétés (optionnel)
