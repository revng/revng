from typing import Dict, List


def clean_double_dict(dictionary: Dict[str, Dict[str, List]]):
    keys_to_delete = []
    for key in dictionary.keys():
        clean_dict(dictionary[key])
        if not dictionary[key]:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        dictionary.pop(key)


def clean_dict(dictionary: Dict[str, List]):
    keys_to_delete = []
    for key in dictionary.keys():
        if not dictionary[key]:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        dictionary.pop(key)
