import os
from cohere import ClientV2, SystemChatMessageV2, UserChatMessageV2


def system_message(lang: str = "en") -> str:
    return f"""You are an expert lexicographer and linguist specialising in {lang} semantics.

# TASK
You will be given
- a target token
- the original sentence it occurs in

Using this context, produce: a possible English dictionary-style definition of the target word in this context. The definition must be a phrase that fully captures and explains the sense. Example: "she ran all the way home" -> "move rapidly from one place to another"

# GUIDELINES
- Be specific and detailed enough to distinguish senses.
- Account for negation: define the words meaning, not its truth value. Example: "she didn't run" -> "move at a speed faster than a walk", not "stand "still".
- Account for metaphorical meanings: when in doubt, include both literal and metaphorical definitions. Example: "she saw a risk in his plan" -> "perceive a situation mentally" works better than "perceive by sight".
- Beware of distinguishing the actual meaning of a verb from those of its arguments.
    - Examples: "He shed a few tears" -> "let fall, emit" is the right choice, not "cry", which would absorb the meaning of "tears".
    - However, "She threw a party" -> "organize an event" is appropriate, just like "She caught a cold" -> "get struck by an illness", as those are actual meanings conveyed by the verbs.
- Avoid unnecessary contextual information: "She ate the cake gleefully" -> "take in food", not "take in food in a joyous manner".
    - However, "He devoured the cake" -> "eat quickly and hungrily" is correct, because the specific manner of eating is a core semantic feature inherently lexicalized in the verb itself.
- Keep outputs precise and consistent.

# OUTPUT:
Return ONLY valid definition as string.
No commentary, no markdown.
    """


def format_msg(lemma: str, instance: str) -> str:
    return f'TARGET TOKEN: {lemma}\n\nSENTENCE: "{instance}"'


class AyaClient:
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = os.environ["COHERE_API_KEY"]
        self.client = ClientV2(api_key)

    def __call__(self, model: str, msg: str) -> str:
        response = self.client.chat(
            model=model,
            messages=[
                SystemChatMessageV2(content=system_message()),
                UserChatMessageV2(content=msg),
            ],
        )
        return response.message.content[0].text


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ["COHERE_API_KEY"]
    aya = AyaClient(api_key)
    aya("tiny-aya-global", format_msg("mole", "I have a mole on my face"))
