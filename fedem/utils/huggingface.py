from huggingface_hub import HfApi
from huggingface_hub.utils._headers import LocalTokenNotFoundError
from requests import HTTPError


def get_client_details(hf_token: str | None = None) -> tuple:
    """
    Get the client details from the Hugging Face API.

    Args:
        hf_token (str, optional): The Hugging Face API token. Defaults to None.

    Returns:
        tuple | None: The Hugging Face API client and the user details.
    """

    try:
        api: HfApi = HfApi(token=hf_token)
        whoami: dict = api.whoami()
        return api, whoami

    except LocalTokenNotFoundError:
        raise LocalTokenNotFoundError(
            "You need to be logged in to use this feature. Please send hf_token as a parameter or run `huggingface-cli login` in cli."
        )
    except (HTTPError, Exception) as e:
        raise Exception(
            f"Something went wrong while trying to fetch your user details. {e}"
        )


def verify_user_with_org(
    client_details: dict, org_id: str, access_level: list = ['contributor']
) -> dict:
    """
    Verify if the user is part of the organization.

    Args:
        client_details (dict): The client details.
        org_id (str): The organization id.

    Returns:
        dict | None: The org details if the user is part of the organization, else None.
    """

    for hf_org in client_details['orgs']:
        if (hf_org['name'] == org_id) and (hf_org['roleInOrg'] in access_level):
            return hf_org

    raise Exception(
        f"{client_details['fullname']} is not part of the {org_id} organization. Please join as a {access_level}."
    )
