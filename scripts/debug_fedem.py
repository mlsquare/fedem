import sys

PATH = "/workspaces/Experiments/fedem/"

try:
    sys.path.index(PATH)
except ValueError:
    sys.path.append(PATH)


from fedem.client import Seshu

model = Seshu(
    hf_model_path="mlsquare/pico_seshu_test",
    hf_tokenizer_path="google/byt5-large",
    target_modules=["model.layers.3.dt_proj"],
    hf_adapter_path="mlsquare/mamba_pico_small_dt_proj",
    hf_data_path="mlsquare/CLIENT_samantar_mixed_train_val",
    hf_token="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",
)

model.train_lora(debug=True)

model.push_to_hub()
