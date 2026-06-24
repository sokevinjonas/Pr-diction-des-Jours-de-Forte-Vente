"""
Rate limiting — Protège contre les abus et les DoS.
Limite les requêtes par utilisateur/IP.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, Optional

from src.audit import init_audit_db, log_security_event


class RateLimiter:
    """Implémente un rate limiting configurable."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Args:
            max_requests : nombre de requêtes autorisées
            window_seconds : fenêtre de temps en secondes (ex: 3600 = 1h)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.db_path = init_audit_db()

    def check_rate_limit(self, identifier: str, endpoint: str = "general") -> Tuple[bool, str]:
        """
        Vérifie si la limite est dépassée.
        Retourne (allowed, reason).

        Args:
            identifier : user_id ou IP address
            endpoint : quel endpoint (pour des limites différentes)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Récupérer les requêtes récentes pour cet identifiant
        cursor.execute("""
            SELECT request_count, window_start FROM rate_limits
            WHERE (user_id = ? OR ip_address = ?) AND endpoint = ?
            ORDER BY window_start DESC LIMIT 1
        """, (identifier, identifier, endpoint))

        row = cursor.fetchone()

        if row is None:
            # Premier accès - créer un nouvel entry
            cursor.execute("""
                INSERT INTO rate_limits (user_id, endpoint, request_count, window_start, last_request)
                VALUES (?, ?, ?, ?, ?)
            """, (identifier, endpoint, 1, now.isoformat(), now.isoformat()))
            conn.commit()
            conn.close()
            return True, "First request"

        request_count, window_str = row
        window_start_dt = datetime.fromisoformat(window_str)

        # Si la fenêtre est expirée, reset
        if window_start_dt + timedelta(seconds=self.window_seconds) < now:
            cursor.execute("""
                UPDATE rate_limits
                SET request_count = 1, window_start = ?, last_request = ?
                WHERE (user_id = ? OR ip_address = ?) AND endpoint = ?
            """, (now.isoformat(), now.isoformat(), identifier, identifier, endpoint))
            conn.commit()
            conn.close()
            return True, "Window reset"

        # Fenêtre active - vérifier la limite
        if request_count >= self.max_requests:
            conn.close()

            # Logger l'abus
            log_security_event(
                event_type="rate_limit_exceeded",
                severity="warning",
                description=f"Rate limit exceeded: {request_count}/{self.max_requests} requests",
                user_id=identifier,
                context={"endpoint": endpoint, "window_seconds": self.window_seconds}
            )

            return False, f"Rate limit exceeded: {request_count}/{self.max_requests} in {self.window_seconds}s"

        # Incrémenter le compteur
        cursor.execute("""
            UPDATE rate_limits
            SET request_count = request_count + 1, last_request = ?
            WHERE (user_id = ? OR ip_address = ?) AND endpoint = ?
        """, (now.isoformat(), identifier, identifier, endpoint))
        conn.commit()
        conn.close()

        return True, f"OK ({request_count + 1}/{self.max_requests})"

    def cleanup_old_records(self, days: int = 30):
        """Nettoie les anciens records de rate limiting."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("DELETE FROM rate_limits WHERE window_start < ?", (cutoff_date,))

        conn.commit()
        conn.close()


# Instances pré-configurées
RATE_LIMITER_API = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 req/hour
RATE_LIMITER_PREDICT = RateLimiter(max_requests=100, window_seconds=60)  # 100 req/minute
RATE_LIMITER_ADMIN = RateLimiter(max_requests=50, window_seconds=3600)   # 50 req/hour pour admin
