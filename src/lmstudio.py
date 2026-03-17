import os
from openai import OpenAI
from aya import system_message


class LMStudioClient:
    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = os.environ["LMSTUDIO_BASE_URL"]
        self.client = OpenAI(base_url=base_url, api_key="")

    def __call__(self, model: str, msg: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message()},
                {"role": "user", "content": msg},
            ],
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    base_url = os.environ["LMSTUDIO_BASE_URL"]
    cl = LMStudioClient(base_url=base_url)
    print(cl("gemma-2-2b-it", "hello, world!"))
