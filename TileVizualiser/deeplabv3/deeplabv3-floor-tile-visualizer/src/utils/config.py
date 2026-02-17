import yaml

def load_config(config_path='config.yaml'):
    """Load configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model_config(config):
    """Get model configuration settings."""
    return config.get('model', {})

def get_tile_config(config):
    """Get tile configuration settings."""
    return config.get('tiles', {})

def get_processing_config(config):
    """Get processing configuration settings."""
    return config.get('processing', {})