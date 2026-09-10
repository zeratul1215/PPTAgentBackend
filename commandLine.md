frontend:
cd agent_frontend && npm run dev

backend:
python -m uvicorn agent_backend.server.app:app \
 --host 127.0.0.1 --port 8765

clear:
.venv/bin/python -m agent_backend.server.reset_state --yes
