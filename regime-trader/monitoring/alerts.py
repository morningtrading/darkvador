"""
alerts.py -- Email and webhook alerts for critical trading events.

Rate-limits outbound alerts so that a cascading drawdown does not flood
the operator's inbox.  All alert methods are safe to call frequently --
duplicates within the rate-limit window are silently dropped.

Credentials are loaded from environment variables:
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD
  ALERT_RECIPIENT, ALERT_WEBHOOK_URL
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Webhook colour map (Slack attachment colour field)
_LEVEL_COLOURS = {
    "INFO":     "#36a64f",   # green
    "WARNING":  "#ff9800",   # orange
    "CRITICAL": "#e53935",   # red
}


class AlertManager:
    """
    Send email and/or webhook alerts for critical events, with rate limiting.

    Parameters
    ----------
    smtp_host           : SMTP server hostname.
    smtp_port           : SMTP port (default 587, STARTTLS).
    smtp_user           : SMTP login username.
    smtp_password       : SMTP login password.
    recipient           : Default email recipient.
    webhook_url         : Slack / Teams / Discord webhook URL.
    rate_limit_minutes  : Min gap between repeated alerts of the same key.
    """

    def __init__(
        self,
        smtp_host:          Optional[str] = None,
        smtp_port:          int           = 587,
        smtp_user:          Optional[str] = None,
        smtp_password:      Optional[str] = None,
        recipient:          Optional[str] = None,
        webhook_url:        Optional[str] = None,
        rate_limit_minutes: int           = 15,
    ) -> None:
        # Prefer constructor args; fall back to environment variables
        self.smtp_host     = smtp_host     or os.environ.get("ALERT_SMTP_HOST")
        self.smtp_port     = smtp_port
        self.smtp_user     = smtp_user     or os.environ.get("ALERT_SMTP_USER")
        self.smtp_password = smtp_password or os.environ.get("ALERT_SMTP_PASSWORD")
        self.recipient     = recipient     or os.environ.get("ALERT_RECIPIENT")
        self.webhook_url   = webhook_url   or os.environ.get("ALERT_WEBHOOK_URL")
        self.rate_limit_minutes = rate_limit_minutes

        self._last_sent: Dict[str, dt.datetime] = {}

    # ======================================================================= #
    # Public API                                                               #
    # ======================================================================= #

    def alert(
        self,
        title:     str,
        message:   str,
        level:     str           = "WARNING",
        alert_key: Optional[str] = None,
    ) -> bool:
        """
        Send an alert via all configured channels, subject to rate limiting.

        Returns True if sent, False if suppressed.
        """
        key = alert_key or title

        if not self.check_rate_limit(key):
            logger.debug(
                "Alert suppressed (rate limit): key=%s  next_allowed=%s",
                key,
                (self._last_sent[key] + dt.timedelta(minutes=self.rate_limit_minutes))
                .strftime("%H:%M:%S"),
            )
            return False

        sent = False

        if self.smtp_host and self.recipient:
            sent |= self.send_email(title, message, self.recipient)

        if self.webhook_url:
            sent |= self.send_webhook(title, message, level)

        if not self.smtp_host and not self.webhook_url:
            # No channels configured -- log locally so the event is not lost
            log_fn = getattr(logger, level.lower(), logger.warning)
            log_fn("ALERT [%s] %s: %s", level, title, message)
            sent = True

        if sent:
            self._update_rate_limit(key)

        return sent

    def send_email(
        self,
        subject:   str,
        body:      str,
        recipient: Optional[str] = None,
    ) -> bool:
        """
        Send an email alert via SMTP with STARTTLS.

        Returns True on success, False on any error.
        """
        to = recipient or self.recipient
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, to]):
            logger.debug("send_email: SMTP not fully configured -- skipping")
            return False

        msg            = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[RegimeTrader] {subject}"
        msg["From"]    = self.smtp_user
        msg["To"]      = to

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.smtp_user, self.smtp_password)
                smtp.sendmail(self.smtp_user, [to], msg.as_string())
            logger.info("Alert email sent to %s | subject=%s", to, subject)
            return True
        except Exception as exc:
            logger.error("send_email failed: %s", exc)
            return False

    def send_webhook(
        self,
        title:   str,
        message: str,
        level:   str = "WARNING",
    ) -> bool:
        """
        POST a formatted message to the configured webhook URL.

        Returns True if the POST returned 2xx.
        """
        if not self.webhook_url:
            return False

        payload = self._format_webhook_payload(title, message, level)
        try:
            resp = requests.post(
                self.webhook_url,
                json    = payload,
                timeout = 10,
            )
            resp.raise_for_status()
            logger.info("Webhook alert sent | title=%s | status=%d", title, resp.status_code)
            return True
        except Exception as exc:
            logger.error("send_webhook failed: %s", exc)
            return False

    # ======================================================================= #
    # Convenience methods                                                      #
    # ======================================================================= #

    def alert_drawdown_halt(
        self,
        equity:        float,
        drawdown_pct:  float,
    ) -> bool:
        """Fire a CRITICAL alert when the peak-drawdown halt threshold fires."""
        return self.alert(
            title   = "TRADING HALTED: Peak Drawdown Breached",
            message = (
                f"Peak drawdown reached {drawdown_pct:.2%}.\n"
                f"Current equity: ${equity:,.2f}\n"
                f"All trading has been halted.\n"
                f"Delete trading_halted.lock to resume."
            ),
            level     = "CRITICAL",
            alert_key = "drawdown_halt",
        )

    def alert_regime_change(
        self,
        previous_regime: str,
        new_regime:      str,
        confidence:      float,
    ) -> bool:
        """Fire an INFO alert on regime transitions."""
        return self.alert(
            title   = f"Regime Change: {previous_regime} -> {new_regime}",
            message = (
                f"HMM detected a regime transition.\n"
                f"Previous: {previous_regime}\n"
                f"New     : {new_regime}\n"
                f"Confidence: {confidence:.1%}"
            ),
            level     = "INFO",
            alert_key = f"regime_{previous_regime}_{new_regime}",
        )

    def alert_order_error(
        self,
        symbol:        str,
        error_message: str,
    ) -> bool:
        """Fire a WARNING alert on order submission failure."""
        return self.alert(
            title   = f"Order Error: {symbol}",
            message = (
                f"An order for {symbol} failed.\n"
                f"Error: {error_message}"
            ),
            level     = "WARNING",
            alert_key = f"order_error_{symbol}",
        )

    # ======================================================================= #
    # Rate limiting                                                            #
    # ======================================================================= #

    def check_rate_limit(self, alert_key: str) -> bool:
        """
        Return True if enough time has elapsed since the last alert with
        this key (i.e. it is OK to send again).
        """
        last = self._last_sent.get(alert_key)
        if last is None:
            return True
        elapsed = (dt.datetime.utcnow() - last).total_seconds() / 60.0
        return elapsed >= self.rate_limit_minutes

    def _update_rate_limit(self, alert_key: str) -> None:
        """Record the current UTC time as the last send time for alert_key."""
        self._last_sent[alert_key] = dt.datetime.utcnow()

    # ======================================================================= #
    # Private helpers                                                          #
    # ======================================================================= #

    def _format_webhook_payload(
        self,
        title:   str,
        message: str,
        level:   str,
    ) -> Dict:
        """
        Build a Slack-compatible attachment payload.

        Compatible with Slack incoming webhooks; also works with many other
        webhook services that accept a generic JSON body.
        """
        colour = _LEVEL_COLOURS.get(level.upper(), "#607d8b")
        return {
            "attachments": [
                {
                    "color":     colour,
                    "title":     f"[{level}] {title}",
                    "text":      message,
                    "footer":    "Regime Trader",
                    "ts":        int(dt.datetime.utcnow().timestamp()),
                }
            ]
        }
