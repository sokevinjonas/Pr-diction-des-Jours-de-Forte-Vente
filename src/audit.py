"""
Audit logging — Enregistre tous les accès, prédictions, erreurs pour conformité et debugging.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils import PROJECT_ROOT

AUDIT_DB = PROJECT_ROOT / "data" / "processed" / "audit.db"


def init_audit_db():
    """Crée la base d'audit et retourne le chemin."""
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    # Log des accès API
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_access (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            user_id TEXT,
            ip_address TEXT,
            method TEXT,
            status_code INTEGER,
            response_time_ms REAL,
            error_message TEXT
        )
    """)

    # Log des prédictions (audit trail)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            use_case TEXT NOT NULL,
            user_id TEXT,
            input_hash TEXT,
            prediction REAL,
            validation_status TEXT,
            security_check_passed INTEGER,
            anomaly_detected INTEGER,
            anomaly_reason TEXT,
            model_version TEXT
        )
    """)

    # Log des sécurité (violations, abus)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT,
            severity TEXT,
            description TEXT,
            user_id TEXT,
            ip_address TEXT,
            context_json TEXT
        )
    """)

    # Configuration de l'accès (rate limiting)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            ip_address TEXT,
            endpoint TEXT,
            request_count INTEGER,
            window_start TEXT,
            last_request TEXT
        )
    """)

    conn.commit()
    conn.close()

    return AUDIT_DB


def log_api_access(endpoint: str, method: str, status_code: int,
                   response_time_ms: float, user_id: Optional[str] = None,
                   ip_address: Optional[str] = None, error_message: Optional[str] = None):
    """Enregistre un accès API."""
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO api_access
        (timestamp, endpoint, method, status_code, response_time_ms, user_id, ip_address, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        endpoint,
        method,
        status_code,
        response_time_ms,
        user_id,
        ip_address,
        error_message,
    ))

    conn.commit()
    conn.close()


def log_prediction_audit(use_case: str, prediction: float, validation_status: str,
                        security_check_passed: bool, anomaly_detected: bool = False,
                        anomaly_reason: str = None, model_version: str = "v2",
                        user_id: Optional[str] = None, input_hash: Optional[str] = None):
    """Enregistre une prédiction pour audit."""
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_audit
        (timestamp, use_case, prediction, validation_status, security_check_passed,
         anomaly_detected, anomaly_reason, model_version, user_id, input_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        use_case,
        float(prediction),
        validation_status,
        int(security_check_passed),
        int(anomaly_detected),
        anomaly_reason,
        model_version,
        user_id,
        input_hash,
    ))

    conn.commit()
    conn.close()


def log_security_event(event_type: str, severity: str, description: str,
                      user_id: Optional[str] = None, ip_address: Optional[str] = None,
                      context: Optional[dict] = None):
    """Enregistre un événement de sécurité (violation, abus, etc)."""
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    context_json = json.dumps(context) if context else None

    cursor.execute("""
        INSERT INTO security_events
        (timestamp, event_type, severity, description, user_id, ip_address, context_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        event_type,
        severity,
        description,
        user_id,
        ip_address,
        context_json,
    ))

    conn.commit()
    conn.close()


def get_audit_log(days: int = 7, event_type: Optional[str] = None) -> list:
    """Récupère l'audit log."""
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    query = """
        SELECT * FROM prediction_audit
        WHERE timestamp > datetime('now', '-' || ? || ' days')
    """
    params = [days]

    if event_type:
        query += " AND validation_status = ?"
        params.append(event_type)

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def get_security_events(days: int = 7, severity: Optional[str] = None) -> list:
    """Récupère les événements de sécurité."""
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    query = """
        SELECT * FROM security_events
        WHERE timestamp > datetime('now', '-' || ? || ' days')
    """
    params = [days]

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def export_compliance_report(output_path: Optional[str] = None) -> str:
    """Exporte un rapport de conformité (audit trail complet)."""
    if output_path is None:
        output_path = PROJECT_ROOT / f"compliance_report_{datetime.now().strftime('%Y%m%d')}.json"

    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    # Récupérer les 1000 derniers événements
    cursor.execute("SELECT * FROM prediction_audit ORDER BY timestamp DESC LIMIT 1000")
    predictions = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]

    cursor.execute("SELECT * FROM security_events ORDER BY timestamp DESC LIMIT 100")
    security = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]

    conn.close()

    report = {
        "generated_at": datetime.now().isoformat(),
        "prediction_audit": predictions,
        "security_events": security,
        "summary": {
            "total_predictions": len(predictions),
            "total_security_events": len(security),
            "critical_events": sum(1 for e in security if e.get("severity") == "critical"),
        }
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return str(output_path)
