# type: ignore

import json

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, ModelFilter
from peft import PeftMixedModel
from tqdm import tqdm
from transformers import AutoTokenizer

from .models.mamba import MambaForCausalLM


def get_models_by_organization(org_id):
    api = HfApi()
    new_filter = ModelFilter(tags="mamba")
    models = api.list_models(filter=new_filter)
    models_list = []
    for i in models:
        print(i.modelId)
        if org_id in i.modelId:
            models_list.append(i.modelId)
    return models_list


# org_id = "mlsquare"
# models = get_models_by_organization(org_id)
# models


def compute_loss(model, inputs, return_outputs=False):
    lm_logits = model(inputs)[0]
    labels = inputs.to(lm_logits.device)

    shift_logits = lm_logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    return lm_loss


def evaluation(data, model, tokenizer):
    val = 0
    for i in tqdm(data, desc="Evaluating"):
        value = tokenizer.encode(i['tgt'], return_tensors="pt")
        val += compute_loss(model, value)

    avg_loss = val / len(data)
    print("LOSS: ", avg_loss)
    return avg_loss


def model_merge_large(adapters, model_path, data, tokenizer):

    model = MambaForCausalLM.from_pretrained(model_path)
    print("model loaded")

    model.load_adapter(adapters["large"][0])
    print("adapter merged")

    result = evaluation(data, model, tokenizer)
    return result


def model_merge_small(adapters, model_path, data, tokenizer):

    base_model = MambaForCausalLM.from_pretrained(model_path)
    print("model loaded")

    peft_model = PeftMixedModel.from_pretrained(base_model, adapters["small"][0])
    peft_model.load_adapter(adapters["small"][1], adapter_name="1")
    peft_model.load_adapter(adapters["small"][2], adapter_name="2")
    peft_model.set_adapter(["default", "1", "2"])
    print("adapter merged")

    result = evaluation(data, peft_model, tokenizer)
    return result


def create_JSON(value):
    json_data = json.dumps(value, indent=4)
    with open(f"{value}", "w") as json_file:
        json_file.write(json_data)


def get_data(data_path, fraction=0.01):
    data = load_dataset(data_path)['train'].shuffle()
    data = data.select(list(range(int(len(data) * fraction))))
    print("data fetched")
    return data


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)
