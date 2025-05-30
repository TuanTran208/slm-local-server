# Local SLM Server - User Guide

## Introduction

Local SLM Server is an API server application that allows you to run Large Language Models (LLM) on your personal computer. The server supports GGUF format models and provides APIs to interact with these models.

## System Requirements

- Python 3.10 or higher
- Storage: enough space to store GGUF models (typically 4GB to 60GB depending on the model)
- CUDA-compatible GPU (optional, but recommended for faster processing)

## Installation Process

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Configuration and Model Directories

```bash
mkdir -p config models
```

### 3. Download GGUF Models

Download GGUF models from Hugging Face or other sources. Models must be in `.gguf` format.

Popular model examples:
- [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [Llama-3-8B-Instruct-GGUF](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)
- [Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)

Save models to the `models` directory following this structure:

```
models/
  ├── llama-2-7b-chat.Q4_K_M.gguf
  ├── mistral-7b-instruct-v0.2.Q4_K_M.gguf
  ├── llama-3-8b-instruct.Q4_K_M.gguf
  └── llama-3.2-3b-instruct-q8_0.gguf
```

### 4. Server Configuration

Create a `config/models.yaml` file with the following content:

```yaml
models:
  - id: llama2-7b
    name: Llama 2 7B
    model_path: models/llama-2-7b-chat.Q4_K_M.gguf
    model_family: llama
    context_length: 4096
    local: true
    default_parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 2048
      repeat_penalty: 1.1
    provider_parameters:
      n_gpu_layers: -1  # -1: maximum GPU usage, 0: CPU only
      n_ctx: 4096
      use_mlock: false
      use_mmap: true
  
  - id: llama3-3b-q8
    name: Llama 3.2 3B Instruct Q8
    model_path: models/llama-3.2-3b-instruct-q8_0.gguf
    model_family: llama
    context_length: 8192
    local: true
    default_parameters:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 2048
      repeat_penalty: 1.1
    provider_parameters:
      n_gpu_layers: -1  # -1: maximum GPU usage, 0: CPU only
      n_ctx: 8192
      use_mlock: false
      use_mmap: true
```
``
Create a `config/.env` file with:

`````````
LLM_SERVER_HOST=0.0.0.0
LLM_SERVER_PORT=8888
```

## Starting the Server

```bash
python main.py
```

The server will run at http://localhost:8888 (or the port you configured in .env)

## Web User Interface

The project now includes a web user interface in the `index.html` file located in the root directory. To use it:

1. Start the server with `python main.py`
2. Open the `index.html` file in your browser
3. The UI will connect to the server running at http://localhost:8888

Key features of the web interface:
- Model management (view, load, select models)
- Chat with models with or without streaming
- Conversation management
- Customizable generation parameters (temperature, max tokens, top_p)

Note: The web UI works directly from the file system and doesn't need to be served.

## API Workflow

### 1. Check and Load Models

#### List All Models
```
GET http://localhost:8888/api/models
```

#### Load Model into Memory
```
POST http://localhost:8888/api/models/{model_id}/load
```

#### Load Model in Background (non-blocking)
```
POST http://localhost:8888/api/models/{model_id}/load-async
```

#### Check Loading Progress
```
GET http://localhost:8888/api/models/{model_id}/loading-status
```

### 2. Chat with Models

#### Simple Chat (without context)
```
POST http://localhost:8888/api/chat/completions

Body:
{
  "model": "llama2-7b",
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

#### Streaming Chat (receive results token by token)
```
POST http://localhost:8888/api/chat/completions

Body:
{
  "model": "llama2-7b",
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "stream": true
}
```

#### Chat with Context Saving
```
POST http://localhost:8888/api/chat/completions/with_context?model=llama2-7b&message=Hello%2C%20who%20are%20you%3F
```

### 3. Conversation Management

#### Create New Conversation
```
POST http://localhost:8888/api/conversations?model=llama2-7b
```

#### List Conversations
```
GET http://localhost:8888/api/conversations
```

#### Add Message to Conversation
```
POST http://localhost:8888/api/conversations/{conv_id}/messages?role=user&content=Hello
```

## Important Notes

1. **Model Loading**: Each model only needs to be loaded once and will stay in memory for all subsequent requests until the server is restarted or the model is explicitly unloaded.

2. **Conversation Storage**: Conversations are stored in RAM and will be lost when the server is restarted. The system doesn't currently have persistent storage for conversations.

3. **Multiple Conversations**: The server supports multiple conversations, but the current web UI only interacts with one conversation at a time.

## Notes for Systems without CUDA (CPU only)

If your computer doesn't have a NVIDIA GPU with CUDA support:

1. In the `config/models.yaml` file, set `n_gpu_layers: 0` to use CPU only.
2. Use smaller models with higher quantization levels (like Q4_K_M) to reduce RAM usage.
3. Response times will be significantly slower compared to using a GPU.
4. Reduce `context_length` and `max_tokens` to increase speed and reduce RAM usage.

## Supported Model Formats

The server currently only supports models in GGUF format (`.gguf`). These formats are quantized versions of LLM models, helping to reduce size and increase processing speed.

Format typically follows: `model-name.quantize-level.gguf`
Examples:
- `llama-2-7b-chat.Q4_K_M.gguf`
- `llama-3.2-3b-instruct-q8_0.gguf`

Common quantization levels:
- Q4_K_M: Small, fast, good quality for CPU
- Q5_K_M: Good balance between size and quality
- Q8_0: Higher quality, larger size

## Recommendations

1. Use 7B models or smaller if using CPU only
2. Ensure RAM is sufficient (at least double the model size)
3. Reduce `max_tokens` to increase response speed
4. Use stream=true to receive partial responses while the model is processing

## Common Error Troubleshooting

- **Insufficient RAM**: Choose a smaller model or higher quantization level (Q4 instead of Q8)
- **Model not found**: Check that the `model_path` correctly points to the .gguf file location
- **Slow server**: If no GPU is available, consider reducing context_length and using a smaller model