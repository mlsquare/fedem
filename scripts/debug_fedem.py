# import sys

# PATH = "/workspaces/Experiments/fedem/"

# try:
#     sys.path.index(PATH)
# except ValueError:
#     sys.path.append(PATH)


from fedem.client import Seshu

model = Seshu(
    hf_model_path="mlsquare/pico_mamba",
    hf_tokenizer_path="google/byt5-large",
    target_modules=["out_proj"],
    hf_adapter_path="mlsquare/exp-lora-ada-1",
    hf_data_path="mlsquare/samantar1per_cent_merged_with_train_val",
    hf_token="<hf_token>",
)

model.train_lora(debug=True)

model.push_to_hub()
