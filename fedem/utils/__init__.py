# type: ignore

import json
import os

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from ..configurations.mamba import MambaConfig
from ..models.mamba import MambaForCausalLM


def load_json(json_path):
    with open(json_path, "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_data(data_path):
    data = load_dataset(data_path).shuffle()
    return DatasetDict(
        {
            "train": data["train"].select(list(range(int(len(data["train"]) * 0.5)))),
            "valid": data["valid"].select(list(range(int(len(data["valid"]) * 0.5)))),
        }
    )


def load_model_pretrained(config):
    return MambaForCausalLM.from_pretrained(config)


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)


def make_config(json):
    config = MambaConfig(
        vocab_size=json["vocab_size"],
        d_model=json["d_model"],
        d_conv=json["d_conv"],
        expand=json["expand"],
        conv_bias=json["conv_bias"],
        bias=json["bias"],
        n_layer=json["n_layer"],
        dt_rank=json["dt_rank"],
        pad_vocab_size_multiple=json["pad_vocab_size_multiple"],
        initializer_range=json["initializer_range"],
    )
    return config


def split_data(data):
    train_size = int(len(data) * 0.8)
    valid_size = len(data) - train_size

    ds_train = data.select(list(range(train_size)))
    ds_valid = data.select(list(range(train_size, train_size + valid_size)))

    return DatasetDict({"train": ds_train, "valid": ds_valid})


def load_model(config):
    config = make_config(config)
    return MambaForCausalLM(config)


def load_model_with_LoRA(model, target_modules, local_path):
    config = LoraConfig(target_modules=target_modules)
    m1 = get_peft_model(model, config)
    m1.print_trainable_parameters()
    m1.save_pretrained(os.path.join(local_path, "adapter"))
    return m1


def get_checkpoint_model(model_name):
    def get_models_by_organization(org_id, model_name):
        api = HfApi()
        new_filter = ModelFilter(tags="mamba")
        models = api.list_models(filter=new_filter)

        for i in models:
            if org_id in i.modelId:
                print(i)
                if model_name in i.modelId:
                    return i.modelId
        return False

    org_id = "mlsquare"
    return get_models_by_organization(org_id, model_name)


def make_config(json):
    config = MambaConfig(
        vocab_size=json["vocab_size"],
        d_model=json["d_model"],
        d_conv=json["d_conv"],
        expand=json["expand"],
        conv_bias=json["conv_bias"],
        bias=json["bias"],
        n_layer=json["n_layer"],
        dt_rank=json["dt_rank"],
        pad_vocab_size_multiple=json["pad_vocab_size_multiple"],
        initializer_range=json["initializer_range"],
    )
    return config
