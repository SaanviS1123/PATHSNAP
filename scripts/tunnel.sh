#!/usr/bin/env bash
# Expose only the dashboard. Run on the host laptop; keep this terminal open.
# Usage: scripts/tunnel.sh
# Send the printed https://...trycloudflare.com link to the second laptop.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$ROOT/scripts/_demo-env.sh"
command -v cloudflared >/dev/null || die "install cloudflared first: macOS: brew install cloudflared; other OS: https://developers.cloudflare.com/tunnel/downloads/"
curl -fsS --max-time 5 "http://127.0.0.1:${UI_PORT:-8501}/_stcore/health" >/dev/null || die "start scripts/serve.sh ui first"
say "Share the HTTPS dashboard URL printed below. Ctrl-C stops public access. API stays local."
exec cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${UI_PORT:-8501}" --protocol "${TUNNEL_PROTOCOL:-auto}"
