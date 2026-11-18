"""Read .env consistently before choosing host/port for either service."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
if sys.argv[1]=='api':
    import uvicorn
    uvicorn.run('api.main:app',host=os.getenv('API_HOST','127.0.0.1'),port=int(os.getenv('API_PORT','8000')),log_level='info')
else:
    from streamlit.web import cli
    os.environ.setdefault('API_URL',f"http://127.0.0.1:{os.getenv('API_PORT','8000')}")
    sys.argv=['streamlit','run',str(ROOT/'ui/app.py'),'--server.address',os.getenv('UI_HOST','127.0.0.1'),
              '--server.port',os.getenv('UI_PORT','8501'),'--server.headless','true',
              '--server.enableWebsocketCompression','false','--browser.gatherUsageStats','false']
    sys.exit(cli.main())
