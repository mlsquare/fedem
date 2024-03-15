import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, ModelFilter
from peft import PeftMixedModel  # type: ignore
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..configurations.mamba import MambaConfig
from ..models.mamba import MambaForCausalLM
from ..utils.huggingface import get_client_details, verify_user_with_org


class Seshu:
    def __init__(
        self,
        adapters: dict | str,
        config_file: dict | str,
        hf_token: str | None = None,
        org_id: str = "mlsquare",
        train_args=False,
    ):

        self.hf_token = hf_token
        self.api, self.client_details = get_client_details(hf_token=self.hf_token)

        self.username: str = self.client_details['name']
        self.fullname: str = self.client_details['fullname']

        self.org_id = org_id
        self.org_details = verify_user_with_org(
            self.client_details, self.org_id, access_level=['admin', 'write']
        )
        print(
            f"{self.fullname} is part of the organization {self.org_id} with write access."
        )

        if isinstance(adapters, str):
            self.adapters = load_json(adapters)
        else:
            self.adapters = adapters

        if isinstance(config_file, str):
            self.config_data = load_json(config_file)
        else:
            self.config_data = config_file

        if train_args:
            self.train_args = train_args
        else:
            self.train_args = TrainingArguments(
                output_dir="mamba",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                num_train_epochs=4,
                weight_decay=0.1,
                lr_scheduler_type="cosine",
                learning_rate=5e-4,
                fp16=False,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_data["tokenizer_path"])

    def tokenize(self, data):
        outputs = self.tokenizer(
            data["tgt"],
            truncation=True,
            max_length=1024,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):  # type: ignore
            if length != 0:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    def pretrain(self, cpt_hours: int | None = None, debug: bool = False):
        if get_checkpoint_model(self.config_data["upload_path"]):
            model = MambaForCausalLM.from_pretrained(
                self.config_data["upload_path"], token=self.hf_token
            )
        else:
            model = MambaForCausalLM(
                MambaConfig(
                    vocab_size=self.config_data["vocab_size"],
                    d_model=self.config_data["d_model"],
                    d_conv=self.config_data["d_conv"],
                    expand=self.config_data["expand"],
                    conv_bias=self.config_data["conv_bias"],
                    bias=self.config_data["bias"],
                    n_layer=self.config_data["n_layer"],
                    dt_rank=self.config_data["dt_rank"],
                    pad_vocab_size_multiple=self.config_data["pad_vocab_size_multiple"],
                    initializer_range=self.config_data["initializer_range"],
                )
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        data = load_data(self.config_data["data"])
        tokenized_data = data.map(
            self.tokenize, batched=True, remove_columns=data["train"].column_names
        )

        trainer = MambaTrainer(
            model=model,  # type: ignore
            tokenizer=self.tokenizer,
            args=self.train_args,  # type: ignore
            data_collator=data_collator,
            train_dataset=tokenized_data["train"],  # type: ignore
            eval_dataset=tokenized_data["valid"],  # type: ignore
        )

        if cpt_hours:
            start_time = time.time()
            while True:
                if time.time() - start_time > cpt_hours * 3600:
                    break

                if debug:
                    print("Trainer function will be called!")
                else:
                    trainer.train()
        else:

            if debug:
                print("Trainer function will be called!")

            else:
                trainer.train()

        # trainer.save_model(os.path.join(self.local_path, "local_copy"))

        try:
            response = model.push_to_hub(self.config_data["upload_path"])  # type: ignore
            print("Model was uploaded to the hub.")
            print(f"Commit link: {response.commit_url}")  # type: ignore
        except:
            print("Model was not uploaded to the hub due to an error.")

    def model_merge_eval(
        self, model_path, type_config="small", data="mlsquare/SERVER_samantar_mixed_val"
    ):
        adapters = self.adapters[type_config]
        data = get_data(data)
        tokenizer = self.tokenizer
        result = model_merge(adapters, model_path, data, tokenizer)
        return result


def compute_loss(model, inputs, return_outputs=False):
    lm_logits = model(inputs)[0]
    labels = inputs.to(lm_logits.device)

    shift_logits = lm_logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    return lm_loss


def evaluation(data, model, tokenizer, batch_size=32, max_length=1024):
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_loss = 0

    with torch.no_grad():
        model.eval()
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_data = data['tgt'][start_idx:end_idx]
            inputs = [
                tokenizer.encode(
                    datum,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                )
                for datum in batch_data
            ]
            input_ids = torch.cat(inputs, dim=0)

            loss = compute_loss(model, input_ids)
            total_loss += loss.item() * (end_idx - start_idx)

    avg_loss = total_loss / num_samples
    return avg_loss


def model_merge(adapters, model_path, data, tokenizer):
    base_model = MambaForCausalLM.from_pretrained(model_path)
    print("model loaded")
    ls_count = 0
    names = ["default"]
    peft_model = PeftMixedModel.from_pretrained(base_model, adapters[ls_count])  # type: ignore
    ls_count += 1
    while ls_count < len(adapters):
        peft_model.load_adapter(adapters[ls_count], adapter_name=str(ls_count))
        names.append(str(ls_count))
        ls_count += 1

    peft_model.set_adapter(names)
    peft_model = peft_model.merge_and_unload()
    print("adapter merged")

    result = evaluation(data, peft_model, tokenizer)
    return result


def create_JSON(value):
    json_data = json.dumps(value, indent=4)
    with open(f"{value}", "w") as json_file:
        json_file.write(json_data)


def get_data(data_path, fraction=0.01):
    data = load_dataset(data_path)['train'].shuffle()  # type: ignore
    data = data.select(list(range(int(len(data) * fraction))))
    print("data fetched")
    return data


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
            "train": data["train"].select(list(range(int(len(data["train"]) * 0.5)))),  # type: ignore
            "valid": data["valid"].select(list(range(int(len(data["valid"]) * 0.5)))),  # type: ignore
        }
    )


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


def get_checkpoint_model(model_name):
    def get_models_by_organization(org_id, model_name):
        api = HfApi()
        new_filter = ModelFilter(tags="mamba")
        models = api.list_models(filter=new_filter)

        for i in models:
            if org_id in i.modelId:  # type: ignore
                print(i)
                if model_name in i.modelId:  # type: ignore
                    return i.modelId  # type: ignore
        return False

    org_id = "mlsquare"
    return get_models_by_organization(org_id, model_name)


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids)[0]
        labels = input_ids.to(lm_logits.device)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss
