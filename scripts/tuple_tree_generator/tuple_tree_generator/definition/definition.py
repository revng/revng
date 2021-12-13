#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Dict
from abc import abstractmethod


class Definition:
    def __init__(self, namespace, user_namespace, name, fullname, user_fullname, doc=None, tag=None):
        # Namespace where the generated code will reside
        self.namespace = namespace
        # Namespace to be used by the user
        self.user_namespace = user_namespace
        self.name = name
        self.fullname = fullname
        self.user_fullname = user_fullname
        self.doc = doc
        self.filename = f"{self.name}.h"
        self.impl_filename = f"{self.name}.cpp"
        if tag:
            self.tag = tag
        else:
            self.tag = name

        # Names of types on which this definition depends on. Not necessarily names defined by the user
        self.dependencies = set()
        # Name of required headers
        self.includes = set()

    @staticmethod
    @abstractmethod
    def from_dict(dict: Dict, default_namespace: str):
        raise NotImplementedError()

    def resolve_references(self, generator):
        return
