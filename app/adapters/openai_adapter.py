import openai
from app.core import ports


class OpenAIAdapter(ports.OpenAIPort):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_text(self, prompt: str) -> str:
        print(prompt)
        # response = openai.Completion.create(engine="davinci", prompt=prompt)
        # return response.choices[0].text.strip()
        return "This is a test answer"
