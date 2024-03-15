import sys

PATH = "/workspaces/Experiments/fedem/"

try:
    sys.path.index(PATH)
except ValueError:
    sys.path.append(PATH)


# from fedem.client import Seshu

# model = Seshu(
#     hf_model_path="mlsquare/pico_seshu_test",
#     hf_tokenizer_path="google/byt5-large",
#     target_modules=["model.layers.3.dt_proj"],
#     hf_adapter_path="mlsquare/mamba_pico_small_dt_proj",
#     hf_data_path="mlsquare/CLIENT_samantar_mixed_train_val",
#     hf_token="<hf_token_here>",
# )

# model.train_lora(debug=True)

# model.push_to_hub()


import os

from fedem.server import Seshu

adapters_path = os.path.join(PATH, "scripts", "adapters.json")
model_parameters_path = os.path.join(PATH, "scripts", "model_parameters.json")

model = Seshu(
    adapters=adapters_path,
    config_file=model_parameters_path,
)

model.model_merge_eval(
    model_path="mlsquare/pico_seshu_test",
    type_config="large",
    data="mlsquare/SERVER_samantar_mixed_val",
)
