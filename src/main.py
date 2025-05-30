import uvicorn
from app import app
from config import settings


def run_service():
    """Run the FastAPI service"""
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    run_service()
    # import torch
    #
    # print(torch.cuda.is_available())  # Should print: True
    # print(torch.cuda.get_device_name(0))  # Should print: NVIDIA RTX 3090
