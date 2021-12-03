import re

__invalid_beginning = re.compile(r"^[_0-9]")
__invalid_chars = re.compile(r"[^a-zA-Z0-9_]")


def valid_identifier(identifier: str, allow_none=True):
    return (allow_none and identifier is None) or (
            identifier != "" and not __invalid_beginning.match(identifier) and not __invalid_chars.match(identifier)
    )


def sanitize_identifier(identifier: str) -> str:
    identifier = __invalid_chars.sub("_", identifier)
    if __invalid_beginning.match(identifier):
        identifier = f"t_{identifier}"

    return identifier
