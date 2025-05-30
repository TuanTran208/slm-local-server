import os
import yaml
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, dotenv_values
import time


class ModelParameters(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    repeat_penalty: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None


class ModelConfig(BaseModel):
    id: str
    name: str
    model_path: Optional[str] = None
    repo_id: Optional[str] = None
    model_family: str
    context_length: int
    local: bool = True
    default_parameters: ModelParameters
    provider_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def get_provider_param(self, param_name: str, default_value: Any = None) -> Any:
        """Get a specific parameter for the provider or default value if it doesn't exist"""
        if not self.provider_parameters:
            return default_value
        return self.provider_parameters.get(param_name, default_value)


class ConfigManager:
    def __init__(self, models_config_path: str = "config/models.yaml", env_path: str = "config/.env"):
        self.models_config_path = models_config_path
        self.env_path = env_path
        self.models: List[ModelConfig] = []
        self.env_config: Dict[str, str] = {}
        self.last_config_load_time = 0

        os.makedirs(os.path.dirname(self.models_config_path), exist_ok=True)

        self.load_models_config()
        self.load_env_config()

    def load_models_config(self) -> None:
        """Load models configuration from YAML file"""
        try:
            if not os.path.exists(self.models_config_path):
                self._create_default_models_config()

            self.last_config_load_time = time.time()
            with open(self.models_config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            self.models = []
            for model_data in config_data.get('models', []):
                params = model_data.pop('default_parameters', {})
                model_data['default_parameters'] = ModelParameters(**params)
                self.models.append(ModelConfig(**model_data))

            print(f"Loaded {len(self.models)} models from config")
        except Exception as e:
            print(f"Error loading models configuration: {str(e)}")
            self.models = []

    def load_env_config(self) -> None:
        """Load configuration from .env file"""
        try:
            if not os.path.exists(self.env_path):
                self._create_default_env_config()

            self.env_config = dotenv_values(self.env_path)

            load_dotenv(dotenv_path=self.env_path)

        except Exception as e:
            print(f"Error loading environment configuration: {str(e)}")
            self.env_config = {}
    
    def _create_default_models_config(self) -> None:
        """Create default models configuration file if it doesn't exist"""
        default_config = {
            'models': [
                {
                    'id': 'llama2-7b',
                    'name': 'Llama 2 7B',
                    'provider': 'huggingface',
                    'model_path': 'models/llama-2-7b-chat-hf',
                    'model_family': 'llama',
                    'context_length': 4096,
                    'local': True,
                    'default_parameters': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_tokens': 2048,
                        'repeat_penalty': 1.1
                    }
                }
            ]
        }

        with open(self.models_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, sort_keys=False)

    def _create_default_env_config(self) -> None:
        """Create default .env file if it doesn't exist"""
        default_env = """LLM_SERVER_HOST=0.0.0.0
LLM_SERVER_PORT=8888
"""
        with open(self.env_path, 'w', encoding='utf-8') as f:
            f.write(default_env)

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def get_all_models(self) -> List[ModelConfig]:
        """Return list of all models with auto-reload if config file has changed"""
        if os.path.exists(self.models_config_path):
            mod_time = os.path.getmtime(self.models_config_path)
            if mod_time > self.last_config_load_time:
                print("Config file has changed, reloading models...")
                self.load_models_config()

        return self.models

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable value from configuration"""
        return os.environ.get(key, default)

    def get_server_host(self) -> str:
        """Get host for server"""
        return self.get_env("LLM_SERVER_HOST", "0.0.0.0")

    def get_server_port(self) -> int:
        """Get port for server"""
        return int(self.get_env("LLM_SERVER_PORT", 8000))