from typing import Dict, Any, Optional

config_ = {
    "redis_url": 'redis://localhost:6379/0',
    "inputs_path": 'model_inputs',
    "config_dir": 'config',
    "templates_dir": 'templates',
    "default_parameters": {},
    "client": 'unknown',
    "event_listeners": {}
}


def update_config(update: Dict[str, Any]):
    global config_
    config_ = {**config_, **update}


def config(key: Optional[str] = None) -> Any:
    return config_[key] if key else config_
