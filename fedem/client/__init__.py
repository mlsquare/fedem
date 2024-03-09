import os

from huggingface_hub import CommitInfo, RepoUrl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from ..models.mamba import MambaForCausalLM
from ..trainer import MambaTrainer
from ..utils import load_data, load_model_with_LoRA
from ..utils.huggingface import get_client_details, verify_user_with_org


class Seshu:

    def __init__(
        self,
        hf_model_path: str,
        hf_tokenizer_path: str,
        target_modules: list[str],
        hf_adapter_path: str,
        hf_data_path: str,
        training_args: TrainingArguments | None = None,
        org_id: str = "mlsquare",
        hf_token: str | None = None,
    ):

        self.hf_token = hf_token
        self.api, self.client_details = get_client_details(hf_token=self.hf_token)

        self.username: str = self.client_details['name']
        self.fullname: str = self.client_details['fullname']

        self.org_id = org_id
        self.org_details = verify_user_with_org(self.client_details, self.org_id)
        print(
            f"{self.fullname} is part of the organization {self.org_id} as a contributor."
        )

        self.hf_model_path = hf_model_path
        self.hf_tokenizer_path = hf_tokenizer_path
        self.target_modules = target_modules
        self.hf_adapter_path = hf_adapter_path
        self.hf_data_path = hf_data_path

        # create a new model repo if the model does not exist on the Hugging Face Hub
        self.repo_name: str = (
            f"{self.org_id}/{self.hf_adapter_path.split('/')[-1]}_{self.username}"
        )

        try:
            response: RepoUrl = self.api.create_repo(self.repo_name, repo_type="model")
            print(f"New model repo created at {response.url}")

        except:
            print(f"Model repo {self.repo_name} already exists on HF.")

        # create a local path to store the model
        self.local_path: str = os.path.join(os.getcwd(), self.repo_name)
        os.makedirs(self.local_path, exist_ok=True)

        if training_args is None:
            self.training_args = TrainingArguments(
                output_dir=os.path.join(self.local_path, "outputs"),
                num_train_epochs=1,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.local_path, "logs"),
            )
        else:
            self.training_args = training_args

        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)

    def tokenize(self, data_to_tokenize):

        outputs = self.tokenizer(
            data_to_tokenize["tgt"],
            truncation=True,
            max_length=1024,
            return_overflowing_tokens=True,
            return_length=True,
        )

        if "length" not in outputs and "input_ids" not in outputs:
            raise ValueError(
                "The tokenizer did not return the expected outputs. Please check the inputs."
            )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):  # type: ignore
            if length != 0:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    def train_lora(
        self,
        debug: bool = False,
    ):

        avail = False
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.hf_model_path, token=self.hf_token
            )
            model.enable_input_require_grads()
            model.load_adapter(self.hf_adapter_path)
            avail = True
        except Exception:
            print("Creating new adapter as the previous one is not valid.")

        if not avail:
            model = MambaForCausalLM.from_pretrained(self.hf_model_path)
            model = load_model_with_LoRA(model, self.target_modules, self.local_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        data_to_tokenize = load_data(self.hf_data_path)

        tokenized_data = data_to_tokenize.map(
            self.tokenize,
            batched=True,
            remove_columns=data_to_tokenize["train"].column_names,
        )

        trainer = MambaTrainer(
            model=model,  # type: ignore
            tokenizer=self.tokenizer,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=tokenized_data["train"],  # type: ignore
            eval_dataset=tokenized_data["valid"],  # type: ignore
        )

        if not debug:
            trainer.train()
        else:
            print("trainer.train() will be called in non debug mode")

            # storing the model as a class variable
            self.model = model

        trainer.save_model(os.path.join(self.local_path, "local_copy"))

    def push_to_hub(self):
        response: CommitInfo = self.api.upload_folder(
            folder_path=os.path.join(self.local_path, "local_copy"),
            repo_id=self.repo_name,
            repo_type="model",
        )

        print(f"Model uploaded to {response.commit_url} successfully.")
