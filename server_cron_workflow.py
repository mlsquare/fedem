import json
import os
import shutil
from datetime import datetime

from huggingface_hub import HfApi

# path for adapters json file
adapters_json_path = os.path.join(os.getcwd(), 'server_cron_adapters.json')

# load the adapters json file
try:
    with open(adapters_json_path, 'r') as f:
        adapters = json.load(f)

except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {adapters_json_path}")


# create a HfApi object
api = HfApi()

output_adapters: dict = {}

for adapter_size in adapters:
    for adapter in adapters[adapter_size]:
        org_id, model_id = adapter.split("/")
        # print(f"org_id: {org_id}, model_id: {model_id}")

        try:
            response = api.list_models(search=model_id, author=org_id)
        except:
            raise Exception(f"Error in listing models for {org_id}/{model_id}")

        for model in response:
            if org_id in model.modelId and model_id in model.modelId:  # type: ignore
                # print(model.modelId)  # type: ignore

                if model.modelId.startswith(adapter + "_"):  # type: ignore
                    output_adapter: list = output_adapters.get(adapter, [])
                    output_adapter.append(model.modelId)  # type: ignore
                    output_adapters[adapter] = output_adapter


# we now analyse the output adapters and delete unnecessary ones
for root_adapter, client_adapters in output_adapters.items():

    oldest_adapter = None

    shutil.rmtree("mlsquare", ignore_errors=True)

    try:
        response = api.hf_hub_download(repo_id=root_adapter, filename="status.json")
        root_json_config = json.load(open(response, 'r'))

        client_time_object = datetime.strptime(
            root_json_config['time'], "%Y-%m-%d %H:%M:%S"
        )

        # if the time difference is more than 3 hours, delete the root status json and client adapters
        if (datetime.utcnow() - client_time_object).total_seconds() > 3 * 60 * 60:
            print(f"Deleting the root status json for {root_adapter}")
            api.delete_file(repo_id=root_adapter, path_in_repo="status.json")

            print(f"Deleting the client adapters for {root_adapter}")
            for client_adapter in client_adapters:
                try:
                    response = api.delete_repo(repo_id=client_adapter)
                except:
                    print(f"Error in deleting {client_adapter}")

        else:
            oldest_adapter = root_adapter + "_" + root_json_config['username']

    except:
        adapters_with_time: dict = {}

        for client_adapter in client_adapters:
            print(f"{root_adapter}: {client_adapter}")

            try:
                response = api.hf_hub_download(
                    repo_id=client_adapter, filename="status.json"
                )
            except:
                print(Exception(f"Error in downloading status.json for {client_adapter}"))
                continue

            client_json = json.load(open(response, 'r'))

            date_object = datetime.strptime(client_json['time'], "%Y-%m-%d %H:%M:%S")
            adapters_with_time[client_adapter] = date_object

        if adapters_with_time == {}:
            continue

        oldest_adapter = min(adapters_with_time, key=lambda k: adapters_with_time[k])

    if oldest_adapter is not None:

        print(
            f"\n\nStage: Root Adapter - {root_adapter} | Oldest Adapter - {oldest_adapter}\n\n"
        )

        try:

            response = api.hf_hub_download(
                repo_id=oldest_adapter, filename="adapter_config.json"
            )

            os.makedirs(oldest_adapter, exist_ok=True)

            try:
                response = api.snapshot_download(
                    repo_id=oldest_adapter, local_dir=oldest_adapter
                )
            except:
                print(f"Error in downloading {oldest_adapter}")
                continue

            os.rename(
                os.path.join(oldest_adapter, "status.json"),
                os.path.join(oldest_adapter, "config.json"),
            )

            try:
                api.upload_folder(
                    folder_path=oldest_adapter,
                    repo_id=root_adapter,
                    commit_message="Update adapter from client",
                )
            except:
                print(f"Error in uploading {oldest_adapter}")
                continue

            try:
                api.delete_file(repo_id=root_adapter, path_in_repo="status.json")
            except:
                # print(f"Error in deleting status.json from {root_adapter}")
                pass

            for client_adapter in client_adapters:
                try:
                    api.delete_repo(repo_id=client_adapter)
                except:
                    print(f"Error in deleting {client_adapter}")
                    continue

        except:

            print(f"Stage 2: Exception Block: {root_adapter}")

            try:
                response = api.hf_hub_download(
                    repo_id=oldest_adapter, filename="status.json"
                )
                api.upload_file(
                    repo_id=root_adapter,
                    path_or_fileobj=response,
                    path_in_repo="status.json",
                )
            except:
                print(f"Error in uploading status.json for {root_adapter}")

            for client_adapter in client_adapters:
                if client_adapter != oldest_adapter:
                    api.delete_repo(repo_id=client_adapter)


print("Pipeline completed successfully!")
