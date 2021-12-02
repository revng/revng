#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Dict

from .definition import Definition


class EnumMember:
    def __init__(self, *, name, doc=None):
        self.name = name
        self.doc = doc

    @staticmethod
    def from_dict(source_dict: Dict):
        return EnumMember(**source_dict)


class EnumDefinition(Definition):
    def __init__(self, *, namespace, user_namespace, name, members, doc=None, tag=None):
        super(EnumDefinition, self).__init__(
            namespace,
            user_namespace,
            name,
            f"{namespace}::Values",
            f"{user_namespace}::Values",
            doc=doc,
            tag=tag,
        )
        self.members = members

    @staticmethod
    def from_dict(source_dict: Dict, default_base_namespace: str):
        args = {
            "namespace": f"{default_base_namespace}::{source_dict['name']}",
            "user_namespace": f"{default_base_namespace}::{source_dict['name']}",
            "name": source_dict["name"],
            "doc": source_dict.get("doc"),
            "members": [EnumMember.from_dict(d) for d in source_dict["members"]],
            "tag": source_dict.get("tag"),
        }
        return EnumDefinition(**args)
