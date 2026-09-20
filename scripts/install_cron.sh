#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_LINE="0 16 * * * $PROJECT_DIR/scripts/run_daily.sh >> $PROJECT_DIR/logs/cron.log 2>&1"

existing_cron="$(crontab -l 2>/dev/null || true)"
if grep -Fqx "$CRON_LINE" <<< "$existing_cron"; then
  echo "Daily 10:00 cron job is already installed."
  exit 0
fi

{
  printf '%s\n' "$existing_cron"
  printf '%s\n' "$CRON_LINE"
} | crontab -
echo "Installed daily 16:00 cron job for $PROJECT_DIR"
