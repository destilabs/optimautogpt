import toolz as tz
from finetoolformer.api import OpenAiChoice, OpenAiMessage

def get_assistant_messages(msgs:list[OpenAiChoice]) -> list[OpenAiMessage]:
    """
    Get assistant messages from a list of messages.

    Args:
        msgs (list): List of messages.

    Returns:
        list: List of assistant messages.
    """
    return list(tz.filter(lambda x: x.message.role == "assistant", msgs))