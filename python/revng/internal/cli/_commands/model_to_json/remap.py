#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import yaml


def is_metaaddress(value):
    return (
        isinstance(value, str)
        and value.startswith("0x")
        and (":Code_" in value or ":Generic" in value)
    )


class MetaAddressRemapper:
    def __init__(self):
        self.addresses = set()
        self.replacements = {}

    def handle(self, value):
        if is_metaaddress(value):
            self.addresses.add(value.split(":")[0])

    def collect(self, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.handle(k)
                self.collect(v)

        elif isinstance(value, list):
            for v in value:
                self.collect(v)

        else:
            self.handle(value)

    def apply_replacement(self, value):
        if isinstance(value, str):
            splits = value.split(":")
            if is_metaaddress(value) and splits[0] in self.replacements:

                return self.replacements[splits[0]]
        return value

    def replace(self, value):
        if isinstance(value, dict):
            new_values = {
                self.apply_replacement(k): self.apply_replacement(v) for k, v in value.items()
            }
            value.clear()
            value.update(new_values)

            for k, v in value.items():
                self.replace(v)

        elif isinstance(value, list):
            new_values = [self.apply_replacement(v) for v in value]
            value.clear()
            for new_value in new_values:
                value.append(new_value)

            for v in value:
                self.replace(v)

    def rewrite(self, value):
        self.replacements = {v: str(i + 1) for i, v in enumerate(sorted(self.addresses))}
        self.replace(value)

        return value


def remap_metaaddress(model):
    mar = MetaAddressRemapper()
    mar.collect(model)
    return mar.rewrite(model)


def parse_model(text_model):
    return yaml.load(text_model, Loader=yaml.SafeLoader)
