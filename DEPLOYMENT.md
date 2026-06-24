# Guide de Déploiement VPS

Déploiement complet du système de prédiction ML en production sur VPS.

---

## 1. Architecture de production

```
┌─────────────────┐
│    Utilisateur  │
└────────┬────────┘
         │ HTTP/HTTPS
         ▼
┌──────────────────┐
│  Nginx (reverse) │  Port 80/443
├──────────────────┤  • Rate limiting
│  API (Uvicorn)   │  • SSL/TLS
│  Port 8000       │  • Load balancing
└──────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌──────────┐
│ Dash   │ │ ML Model │
│ 8501   │ │ SQLite   │
└────────┘ └──────────┘

Données:
├── data/raw/             # Données brutes (input)
├── data/processed/       # Features + DB audit
├── models/               # Modèles entraînés
└── logs/                 # Application logs
```

---

## 2. Prérequis

### Sur votre VPS (Ubuntu 22.04+)

```bash
# Mise à jour système
sudo apt-get update && sudo apt-get upgrade -y

# Dépendances
sudo apt-get install -y \
    python3.11 python3.11-venv \
    docker.io docker-compose \
    nginx \
    curl wget git \
    build-essential libssl-dev libffi-dev

# Utilisateur pour l'app
sudo useradd -m -s /bin/bash mlapp
sudo usermod -aG docker mlapp
```

---

## 3. Déploiement avec Docker

### Option A : Docker Compose (SIMPLE - Recommandé)

```bash
# 1. Clone le projet
cd /home/mlapp
git clone https://github.com/sokevinjonas/Pr-diction-des-Jours-de-Forte-Vente.git
cd Pr-diction-des-Jours-de-Forte-Vente

# 2. Configuration
cp .env.example .env
nano .env  # Modifier les clés secrètes

# 3. Build & démarrer
docker-compose up -d

# 4. Vérifier
docker-compose ps
curl http://localhost/api/
```

**Accès** :
- API : http://votre-domaine.com/api/
- Dashboard : http://votre-domaine.com/

### Option B : Déploiement manuel

#### Étape 1 : Cloner et préparer

```bash
cd /home/mlapp
git clone https://github.com/sokevinjonas/Pr-diction-des-Jours-de-Forte-Vente.git prediction
cd prediction

# Virtual env
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt

# Configuration
cp .env.example .env
nano .env
```

#### Étape 2 : Systemd services

**API REST** — Créer `/etc/systemd/system/prediction-api.service` :

```ini
[Unit]
Description=Prediction API ML
After=network.target

[Service]
Type=notify
User=mlapp
WorkingDirectory=/home/mlapp/prediction
Environment="PATH=/home/mlapp/prediction/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=/home/mlapp/prediction/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --access-logfile /home/mlapp/prediction/logs/api.log \
    --error-logfile /home/mlapp/prediction/logs/api_error.log \
    api.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Dashboard** — Créer `/etc/systemd/system/prediction-dashboard.service` :

```ini
[Unit]
Description=Prediction Dashboard
After=network.target

[Service]
Type=simple
User=mlapp
WorkingDirectory=/home/mlapp/prediction
Environment="PATH=/home/mlapp/prediction/venv/bin"
ExecStart=/home/mlapp/prediction/venv/bin/streamlit run \
    app/dashboard_v2.py \
    --server.port=8501 \
    --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Activer et démarrer
sudo systemctl daemon-reload
sudo systemctl enable prediction-api
sudo systemctl enable prediction-dashboard
sudo systemctl start prediction-api
sudo systemctl start prediction-dashboard

# Vérifier
sudo systemctl status prediction-api
```

#### Étape 3 : Nginx

Éditer `/etc/nginx/sites-available/prediction` :

```nginx
upstream prediction_api {
    server 127.0.0.1:8000;
}

upstream prediction_dashboard {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name votre-domaine.com;

    # Redirecte vers HTTPS en production
    # return 301 https://$server_name$request_uri;

    # API
    location /api/ {
        proxy_pass http://prediction_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
    }

    # Dashboard
    location / {
        proxy_pass http://prediction_dashboard/;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check
    location /health {
        proxy_pass http://prediction_api/;
        access_log off;
    }
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
```

```bash
# Activer site
sudo ln -s /etc/nginx/sites-available/prediction /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 4. SSL/TLS avec Let's Encrypt

```bash
# Installer Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Générer certificat
sudo certbot certonly --nginx -d votre-domaine.com

# Renouvellement auto
sudo systemctl enable certbot.timer
```

Modifier nginx.conf pour HTTPS (décommenter la section HTTPS dans `nginx.conf`).

---

## 5. Monitoring et maintenance

### Logs

```bash
# API logs
tail -f /home/mlapp/prediction/logs/api.log

# Audit logs
sqlite3 /home/mlapp/prediction/data/processed/audit.db \
    "SELECT * FROM security_events ORDER BY timestamp DESC LIMIT 10;"

# Prédictions
sqlite3 /home/mlapp/prediction/data/processed/predictions.db \
    "SELECT * FROM predictions WHERE date_prediction='2026-06-24';"
```

### Health check

```bash
# Test API
curl http://localhost/api/

# Test health endpoint
curl http://localhost/api/health/supermarche

# Test prédiction
curl -X POST http://localhost/api/predict \
    -H "Content-Type: application/json" \
    -d '{
        "use_case": "supermarche",
        "date_prediction": "2026-06-25",
        "features": {
            "jour_semaine": 3,
            "jour_mois": 25,
            "mois": 6
        }
    }'
```

### Ré-entraînement automatique (Cron)

```bash
# Éditer crontab
crontab -e

# Ajouter ligne pour ré-entraîner quotidiennement à 02:00
0 2 * * * cd /home/mlapp/prediction && \
    source venv/bin/activate && \
    python3 train_all.py >> logs/retrain.log 2>&1
```

---

## 6. Sauvegardes

### Script de sauvegarde

```bash
# /home/mlapp/prediction/backup.sh
#!/bin/bash

BACKUP_DIR="/mnt/backups"
mkdir -p $BACKUP_DIR

# Sauvegarder les données & modèles
tar -czf $BACKUP_DIR/prediction-backup-$(date +%Y%m%d).tar.gz \
    data/ models/ logs/

# Garder 30 jours
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed"
```

```bash
# Ajouter en cron (quotidien à 03:00)
0 3 * * * /home/mlapp/prediction/backup.sh
```

---

## 7. Performance et scaling

### Tuning Nginx

```nginx
worker_processes auto;
worker_connections 2048;
keepalive_timeout 65;

# Gzip compression
gzip on;
gzip_types text/plain application/json;
gzip_comp_level 6;
```

### Tuning Gunicorn

```bash
gunicorn -w 8 -k uvicorn.workers.UvicornWorker \
    --worker-class sync \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    api.main:app
```

### Load balancing (Nginx)

Si besoin d'instances multiples :

```nginx
upstream api_backend {
    least_conn;  # Load balancing
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}
```

---

## 8. Monitoring avec Prometheus (optionnel)

Ajouter à `requirements-prod.txt` :

```
prometheus-client==0.19.0
```

Ajouter à `api/main.py` :

```python
from prometheus_client import Counter, Histogram, start_http_server

# Métriques
predictions_total = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.middleware("http")
async def record_metrics(request, call_next):
    with prediction_latency.time():
        response = await call_next(request)
    predictions_total.inc()
    return response

# Démarrer Prometheus
if __name__ == "__main__":
    start_http_server(8888)
```

---

## 9. Troubleshooting

### Erreur "Address already in use"

```bash
# Trouver le processus
lsof -i :8000
# Tuer le processus
kill -9 <PID>
```

### Erreur "Permission denied" sur logs

```bash
sudo chown -R mlapp:mlapp /home/mlapp/prediction/logs
chmod 755 /home/mlapp/prediction/logs
```

### Modèle absent ou périmé

```bash
# Réentraîner manuellement
cd /home/mlapp/prediction
source venv/bin/activate
python3 train_all.py

# Ou via API
curl -X POST http://localhost/api/retrain/supermarche \
    -H "X-API-Key: admin_secret_key"
```

---

## 10. Checklist de déploiement

- [ ] Cloner repo et installer dépendances
- [ ] Copier .env.example → .env et configurer
- [ ] Générer clés secrètes (`openssl rand -hex 32`)
- [ ] Configurer domaine DNS
- [ ] Configurer Nginx et SSL
- [ ] Démarrer services (Docker ou systemd)
- [ ] Vérifier health check: `/api/health/supermarche`
- [ ] Tester prédiction: `/api/predict`
- [ ] Configurer backups automatiques
- [ ] Configurer ré-entrainement quotidien (cron)
- [ ] Monitorer logs: `tail -f logs/api.log`
- [ ] Configurer alertes (optionnel: Slack, email)

---

## Contacts d'urgence

- **API down** : `sudo systemctl restart prediction-api`
- **OOM** : Augmenter RAM VPS
- **Modèle périmé** : Lancer `python3 train_all.py`
- **Abus** : Vérifier `audit.db` et blacklister IP

---

**C'est prêt pour la production !** 🚀
