import json
import os
import shutil
from datetime import datetime

from huggingface_hub import CommitInfo, RepoUrl
from transformers import (
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
        org_id: str = "mlsquare",
        hf_token: str | None = None,
    ):

        self.hf_model_path = hf_model_path
        self.hf_tokenizer_path = hf_tokenizer_path
        self.target_modules = target_modules
        self.hf_adapter_path = hf_adapter_path
        self.hf_data_path = hf_data_path

        self.hf_token = hf_token
        self.api, self.client_details = get_client_details(hf_token=self.hf_token)

        self.username: str = self.client_details['name']
        self.fullname: str = self.client_details['fullname']

        self.org_id = org_id
        self.org_details = verify_user_with_org(self.client_details, self.org_id)
        print(
            f"{self.fullname} is part of the organization {self.org_id} as a contributor."
        )

        try:
            # check if the status json exists in the root adapter repo
            os.makedirs(self.org_id, exist_ok=True)

            response = self.api.hf_hub_download(
                repo_id=self.hf_adapter_path,
                filename="status.json",
                local_dir=self.org_id,
            )

            raise Exception(
                """The adapter is being used by another user.
                Please use a different adapter or wait for couple of hours for it to be available."""
            )

        except:
            print(
                f"The adapter {self.hf_adapter_path} is available for use by {self.username}."
            )

        # create a new model repo if the model does not exist on the Hugging Face Hub
        self.repo_name: str = (
            f"{self.org_id}/{self.hf_adapter_path.split('/')[-1]}_{self.username}"
        )

        try:
            response = self.api.model_info(self.repo_name)
            print(f"Model repo {self.repo_name} already exists on HF.")

            response = self.api.delete_repo(repo_id=self.repo_name)
            print(f"Model repo {self.repo_name} deleted successfully.")
        except:
            print(f"Model repo {self.repo_name} does not exist on HF.")

        response: RepoUrl = self.api.create_repo(self.repo_name, repo_type="model")
        print(f"New model repo created at {response.url}")

        # create a local path to store the model
        self.local_path: str = os.path.join(os.getcwd(), self.repo_name)

        if os.path.exists(self.local_path):
            shutil.rmtree(self.local_path, ignore_errors=True)
            print(
                f"Local path already exists. Deleting the existing path. - {self.local_path}"
            )

        os.makedirs(self.local_path)
        print(f"Local path created - {self.local_path}")

        # create a status json file
        status = {
            "username": self.username,
            "fullname": self.fullname,
            "org_id": self.org_id,
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "hf_model_path": self.hf_model_path,
            "hf_tokenizer_path": self.hf_tokenizer_path,
            "target_modules": self.target_modules,
            "hf_adapter_path": self.hf_adapter_path,
            "hf_data_path": self.hf_data_path,
        }

        os.makedirs(os.path.join(self.local_path, "local_copy"), exist_ok=True)
        json_path = os.path.join(self.local_path, "local_copy", "status.json")

        with open(json_path, "w") as f:
            json.dump(status, f, indent=4)

        print(f"Status json created at {json_path}")

        self.push_to_hub()

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
        training_args: TrainingArguments | None = None,
        debug: bool = False,
    ):

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

        try:
            model = MambaForCausalLM.from_pretrained(self.hf_model_path)
            model.enable_input_require_grads()  # type: ignore
            model.load_adapter(self.hf_adapter_path)  # type: ignore
        except Exception:
            model = MambaForCausalLM.from_pretrained(self.hf_model_path)
            model = load_model_with_LoRA(model, self.target_modules, self.local_path)
            print("Creating new adapter as the previous one is not valid.")

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

        trainer.save_model(os.path.join(self.local_path, "local_copy"))

    def push_to_hub(self):
        response: CommitInfo = self.api.upload_folder(
            folder_path=os.path.join(self.local_path, "local_copy"),
            repo_id=self.repo_name,
            repo_type="model",
        )

        print(f"File(s) uploaded to {response.commit_url} successfully.")
