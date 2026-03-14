import os

from cohere import ClientV2, UserChatMessageV2
from dotenv import load_dotenv


def format_msg(lemma: str, instance: str) -> str:
    return (
        f'Give a brief definition of the word "{lemma}" in the sentence given as '
        f'input. Generate only the definition.\n\nInput: "{instance}"'
    )


class AyaClient:
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = os.environ["COHERE_API_KEY"]
        self.client = ClientV2(api_key)

    def __call__(self, model: str, msg: str) -> str:
        response = self.client.chat(
            model=model, messages=[UserChatMessageV2(content=msg)]
        )
        return response.message.content[0].text


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ["COHERE_API_KEY"]
    aya = AyaClient(api_key)
    aya("tiny-aya-global", format_msg("mole", "I have a mole on my face"))
