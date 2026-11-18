#!/usr/bin/env bash
# Source after _common.sh: only known network settings are read from .env.
need_venv
while IFS='=' read -r name value; do
  export "$name=$value"
done < <("$PY" - <<'PY'
import os
from dotenv import dotenv_values
settings=dotenv_values('.env')
for key,default in [('API_PORT','8000'),('UI_PORT','8501'),('API_HOST','127.0.0.1'),('UI_HOST','127.0.0.1')]:
    value=os.getenv(key,settings.get(key) or default)
    if '\n' in value or '\r' in value: raise SystemExit('Network settings must use a single line')
    print(f'{key}={value}')
PY
)
