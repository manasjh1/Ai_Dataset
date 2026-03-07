import os
import logging
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def create_azure_chat_openai(
    azure_deployment: str = None,
    api_version: str = None,
    api_key: str = None,
    temperature: float = 0.1,
    max_tokens: int = 4000
) -> AzureChatOpenAI:
    """
    Creates and returns an instance of AzureChatOpenAI.
    """
    try:
        # Fallback to environment variables if parameters are not provided
        deployment = azure_deployment or os.environ.get("AZURE_OPENAI_MODEL_TD")
        version = api_version or os.environ.get("OPENAI_API_VERSION_TD")
        key = api_key or os.environ.get("AZURE_OPENAI_KEY_TD")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        if not all([deployment, version, key, endpoint]):
            raise ValueError("Missing required Azure OpenAI credentials in environment variables.")

        llm = AzureChatOpenAI(
            azure_deployment=deployment,
            api_version=version,
            api_key=key,
            azure_endpoint=endpoint,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI: {str(e)}")
        raise e