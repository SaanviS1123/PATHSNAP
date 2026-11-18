#!/usr/bin/env bash
# Run on either laptop. No Python required; checks the tunneled dashboard.
# Usage: bash scripts/check-dashboard.sh https://YOUR-LINK.trycloudflare.com
set -euo pipefail
url="${1:?Pass the dashboard HTTPS URL}"
url="${url%/}"
printf 'Dashboard health: '
curl --fail --silent --show-error --max-time 30 "$url/_stcore/health"
printf '\nOpen this dashboard in a browser: %s\n' "$url"
printf 'Run a prediction and check the host API terminal for its trip/model log.\n'
