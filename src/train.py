import tensorflow as tf

import .models.base_model as models

def model_init():
    model_set = models.Transformer(
        num_layers=8, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000)
