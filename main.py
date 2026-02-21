import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

AGENT_DIR = "agents/"

api = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri="sqlite+aiosqlite:///database/dialog/dialog.db",
    allow_origins=["*"],
    web=True,
)

if __name__ == "__main__":
    uvicorn.run("main:api", host="0.0.0.0", port=8002, reload=True)
