# models/model_registry.py
MODEL_REGISTRY = {}

def register_model(name, network_class, default_config):
    MODEL_REGISTRY[name] = {
        'class': network_class,
        'config': default_config
    }

def create_model(model_name, **overrides):
    model_info = MODEL_REGISTRY[model_name]
    config = {**model_info['config'], **overrides}
    return model_info['class'](config)

# models/preset_models.py
# Registra combinações prontas
register_model(
    name="ff_snn_small",
    network_class=FeedForwardSNN,
    default_config={
        'input_size': 784,
        'hidden_size': 256,
        'output_size': 10,
        'beta': 0.9
    }
)

register_model(
    name="conv_snn_medium", 
    network_class=ConvSNN,
    default_config={
        'filters1': 12,
        'filters2': 64,
        'beta': 0.95
    }
)