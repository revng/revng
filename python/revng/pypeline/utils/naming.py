#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in a string by removing leading and trailing
    whitespace and replacing multiple spaces with a single space.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_flag(name: str) -> str:
    """
    Normalize a flag name by replacing spaces and underscores with
    hyphens and converting it to lowercase.
    """
    return normalize_whitespace(name).replace(" ", "-").replace("_", "-").lower()


def normalize_pos_arg_name(name: str) -> str:
    """
    Normalize a positional argument name by replacing spaces and underscores
    with hyphens and converting it to lowercase.
    This is used for positional arguments that are not flags.
    """
    return normalize_whitespace(name).replace(" ", "_").replace("-", "_").upper()


def normalize_kwarg_name(name: str) -> str:
    """
    Normalize the provided name to the convention used by click on naming
    command handler variable arguments.
    """
    return name.replace("-", "_").lower()
