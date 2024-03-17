# type: ignore

import json
import os

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from ..configurations.mamba import MambaConfig
from ..models.mamba import MambaForCausalLM


def load_json(json_path):
    """
    Load JSON data from a file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.

    """
    with open(json_path, "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: Model to print trainable parameters for.

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
    """
    Load dataset from a given path and split it into train and validation sets.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        DatasetDict: Dictionary containing train and validation datasets.

    """
    data = load_dataset(data_path).shuffle()
    return DatasetDict(
        {
            "train": data["train"].select(list(range(int(len(data["train"]) * 0.5)))),
            "valid": data["valid"].select(list(range(int(len(data["valid"]) * 0.5)))),
        }
    )


def load_model_pretrained(config):
    """
    Load a pre-trained model based on the provided configuration.

    Args:
        config: Model configuration.

    Returns:
        MambaForCausalLM: Loaded pre-trained model.

    """
    return MambaForCausalLM.from_pretrained(config)


def load_tokenizer(path):
    """
    Load tokenizer from a given path.

    Args:
        path (str): Path to the tokenizer.

    Returns:
        AutoTokenizer: Loaded tokenizer.

    """
    return AutoTokenizer.from_pretrained(path)


def make_config(json):
    """
    Create a MambaConfig object based on the provided JSON data.

    Args:
        json (dict): JSON data containing configuration parameters.

    Returns:
        MambaConfig: Created configuration object.

    """
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
    """
    Split dataset into train and validation sets.

    Args:
        data (Dataset): Dataset to split.

    Returns:
        DatasetDict: Dictionary containing train and validation datasets.

    """
    train_size = int(len(data) * 0.8)
    valid_size = len(data) - train_size

    ds_train = data.select(list(range(train_size)))
    ds_valid = data.select(list(range(train_size, train_size + valid_size)))

    return DatasetDict({"train": ds_train, "valid": ds_valid})


def load_model(config):
    """
    Load a model based on the provided configuration.

    Args:
        config: Model configuration.

    Returns:
        MambaForCausalLM: Loaded model.

    """
    config = make_config(config)
    return MambaForCausalLM(config)


def load_model_with_LoRA(model, target_modules, local_path):
    """
    Load a model with LoRA (Low-Rank Adaptation) applied.

    Args:
        model: Base model to apply LoRA to.
        target_modules: List of target modules.
        local_path (str): Local path to save the adapter.

    Returns:
        MambaForCausalLM: Model with LoRA applied.

    """
    config = LoraConfig(target_modules=target_modules)
    m1 = get_peft_model(model, config)
    m1.print_trainable_parameters()
    m1.save_pretrained(os.path.join(local_path, "adapter"))
    return m1


def get_checkpoint_model(model_name):
    """

    Get the checkpoint model based on the model name and organization ID.

    Args:
        model_name (str): Name of the model.

    Returns:
        str | bool: Model ID if found, False otherwise.
    """
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
    """
    Make Config

    """
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
