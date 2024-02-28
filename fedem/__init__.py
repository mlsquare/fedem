# type: ignore

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from .utils import (
    MambaTrainer,
    load_data,
    load_json,
    load_model_pretrained,
    load_model_with_LoRA,
    load_tokenizer,
)

# from utils import get_checkpoint_model, load_model


class Seshu:
    def __init__(self, config_file, train_args=False):
        self.config_data = load_json(config_file)
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
        self.tokenizer = load_tokenizer(self.config_data["tokenizer_path"])

    def tokenize(self, data):
        outputs = self.tokenizer(
            data["tgt"],
            truncation=True,
            max_length=1024,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length != 0:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    def train_lora(self):
        avail = False
        try:
            model = AutoModelForCausalLM.from_pretrained(self.config_data["model_path"])
            model.enable_input_require_grads()
            model.load_adapter(self.config_data["adapter_path"])
            avail = True
        except Exception:
            print("Adapter not valid!! creating new.")
        if not avail:
            model = load_model_pretrained(self.config_data["model_path"])
            model = load_model_with_LoRA(model, self.config_data["target_modules"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        data = load_data(self.config_data["data"])
        tokenized_data = data.map(
            self.tokenize, batched=True, remove_columns=data["train"].column_names
        )
        trainer = MambaTrainer(
            model=model,
            tokenizer=self.tokenizer,
            args=self.train_args,
            data_collator=data_collator,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["valid"],
        )
        trainer.train()
        # model.push_to_hub(self.config_data["upload_path"])

    # TODO: Add pretrain method
    # def pretrain(self):
    #     # model_config = make_config(self.config_data)
    #     if get_checkpoint_model(self.config_data["model_path"]):
    #         model = load_model_pretrained(config)
    #     else:
    #         model = load_model(self.config_data)

    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
    #     data = load_data(self.config_data["data"])
    #     tokenized_data = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)
    #     trainer = MambaTrainer( model=model, tokenizer=self.tokenizer, args=self.train_args, data_collator=data_collator,
    #                             train_dataset=tokenized_data["train"], eval_dataset=tokenized_data["valid"])
    #     trainer.train()
    #     # model.push_to_hub(self.config_data["upload_path"])
