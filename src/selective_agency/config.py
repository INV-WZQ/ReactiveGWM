"""Attention mode used by the model's training parameter configuration."""


def configured_self_attention_mode(config):
    return config.get("model", {}).get("self_attention_mode", "frame_causal")
