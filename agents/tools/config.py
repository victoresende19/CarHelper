from google.adk.models.lite_llm import LiteLlm

def configure_model():
    MODEL_NAME = "openai/gpt-4o-mini"
    MODEL = LiteLlm(model=MODEL_NAME)
    return MODEL