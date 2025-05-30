import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import models, chat, conversations
from utils.config_loader import ConfigManager


config_manager = ConfigManager()


os.makedirs("models", exist_ok=True)


app = FastAPI(
    title="Local SLM Server",
    description="API server for SLM A.I running in local mode",
    version="1.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(models.router)
app.include_router(chat.router)
app.include_router(conversations.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    print("Starting Local AI Server...")
    print("Reading model configurations from config/models.yaml")
    
    port = config_manager.get_server_port()
    host = config_manager.get_server_host()
    
    print(f"Server will run at http://{host}:{port}")
    print(f"Open browser and visit http://localhost:{port}/docs to view API documentation")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("Server stopped.")