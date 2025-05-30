"""
# Offline SLM Package

## Setup and Usage

### 1. Install dependencies
```
uv install
```

### 2. Download the model
```
uv run .\src\download_model.py
```
This will download the model files to ./model_files

### 3. Test the service locally
```
uv run .\src\main.py
```
Access the API at http://localhost:8000/docs

### 4. Package as executable
```
python package.py
```
This creates an executable in ./dist/SLM_Server/

### 5. Run the packaged application
Navigate to ./dist/SLM_Server/ and run SLM_Server.exe
"""


# Known Issue:
```powershell
$Env:UV_HTTP_TIMEOUT=10

```
