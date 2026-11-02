"""Lossless repeated-paragraph compaction, without deleting user repetitions."""
from langchain_core.messages import AIMessage


def compact_history(messages):
    output = []
    before = sum(len(str(message.content)) for message in messages)
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            paragraphs = message.content.split("\n\n")
            seen, kept = set(), []
            for paragraph in paragraphs:
                if paragraph not in seen or not paragraph.strip():
                    kept.append(paragraph)
                seen.add(paragraph)
            message = message.model_copy(update={"content": "\n\n".join(kept)})
        output.append(message)
    after = sum(len(str(message.content)) for message in output)
    return output, {"before_chars": before, "after_chars": after, "removed_chars": before - after}
