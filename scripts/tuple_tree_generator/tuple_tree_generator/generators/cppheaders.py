#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import graphlib
from collections import defaultdict
from typing import Dict, Optional

from .jinja_utils import environment
from ..schema import Schema
from ..schema import StructDefinition
from ..schema import EnumDefinition
from ..schema import SequenceStructField

enum_template = environment.get_template("enum.h.tpl")
struct_template = environment.get_template("struct.h.tpl")
struct_late_template = environment.get_template("struct_late.h.tpl")
struct_impl_template = environment.get_template("struct_impl.cpp.tpl")
struct_forward_decls_template = environment.get_template("struct_forward_decls.h.tpl")
class_forward_decls_template = environment.get_template("class_forward_decls.h.tpl")
all_types_variant_template = environment.get_template("all_types_variant_decls.h.tpl")


class CppHeadersGenerator:
    def __init__(self, schema: Schema, root_type: str, user_include_path: Optional[str] = None):
        self.schema = schema
        self.root_type = root_type

        if not user_include_path:
            user_include_path = ""
        elif not user_include_path.endswith("/"):
            user_include_path = user_include_path + "/"

        # Path where user-provided headers are assumed to be located
        # Prepended to include statements (e.g. #include "<user_include_path>/Class.h")
        self.user_include_path = user_include_path

        self.depgraph = self._build_depgraph()
        self._inverse_depgraph = self._build_inverse_depgraph()

    def emit(self) -> Dict[str, str]:
        sources = {
            "ForwardDecls.h": self._emit_forward_decls(),
            "AllTypesVariant.h": self._emit_all_types_variant_decl(),
        }

        early_definitions = self._emit_early_type_definitions()
        late_definitions = self._emit_late_type_definitions()
        impl_definitions = self._emit_impl()
        sources.update(early_definitions)
        sources.update(late_definitions)
        sources.update(impl_definitions)
        return sources

    def _build_depgraph(self):
        depgraph = {}
        for definition in self.schema.definitions.values():
            depgraph[definition.fullname] = set()
            for dependency_name in definition.dependencies:
                dependency_type = self.schema.get_definition_for(dependency_name)
                if dependency_type:
                    depgraph[definition.fullname].add(dependency_type.fullname)
        return depgraph

    def _build_inverse_depgraph(self):
        inverse_depgraph = defaultdict(set)
        for definition in self.schema.definitions.values():
            for dependency in definition.dependencies:
                inverse_depgraph[dependency].add(definition.fullname)
        return inverse_depgraph

    def _emit_early_type_definitions(self):
        definitions = {}
        toposorter = graphlib.TopologicalSorter(self.depgraph)
        order = list(toposorter.static_order())
        for type_name_to_emit in order:
            type_to_emit = self.schema.get_definition_for(type_name_to_emit)
            if not type_to_emit:
                # Should we emit a warning here?
                continue

            filename = f"Early/{type_to_emit.filename}"
            assert filename not in definitions
            if isinstance(type_to_emit, StructDefinition):
                upcastable_types = self.schema.get_upcastable_types(type_to_emit)
                definition = struct_template.render(
                    struct=type_to_emit, upcastable=upcastable_types, generator=self
                )
            elif isinstance(type_to_emit, EnumDefinition):
                definition = enum_template.render(enum=type_to_emit)
            else:
                raise ValueError()

            definitions[filename] = definition

        return definitions

    def _emit_late_type_definitions(self):
        definitions = {}
        for type_to_emit in self.schema.definitions.values():
            filename = f"Late/{type_to_emit.filename}"
            assert filename not in definitions

            if isinstance(type_to_emit, StructDefinition):
                upcastable_types = self.schema.get_upcastable_types(type_to_emit)
                definition = struct_late_template.render(
                    struct=type_to_emit,
                    upcastable=upcastable_types,
                    root_type=self.root_type,
                    namespace=self.schema.generated_namespace,
                )
            elif isinstance(type_to_emit, EnumDefinition):
                definition = ""
            else:
                raise ValueError()

            definitions[filename] = definition
        return definitions

    def _emit_impl(self):
        definitions = {}
        for type_to_emit in self.schema.definitions.values():
            filename = f"Impl/{type_to_emit.impl_filename}"
            assert filename not in definitions

            if isinstance(type_to_emit, StructDefinition):
                upcastable_types = self.schema.get_upcastable_types(type_to_emit)
                definition = struct_impl_template.render(
                    struct=type_to_emit,
                    upcastable=upcastable_types,
                    generator=self,
                    schema=self.schema,
                )
            elif isinstance(type_to_emit, EnumDefinition):
                definition = ""
            else:
                raise ValueError()

            definitions[filename] = definition

        return definitions

    def _emit_forward_decls(self):
        generated_ns_to_names = defaultdict(set)
        user_ns_to_names = defaultdict(set)
        generated_ns_to_names = defaultdict(set)
        for definition in self.schema.struct_definitions():
            # Forward declaration for autogenerated class
            generated_ns_to_names[definition.namespace].add(definition.name)
            # Forward declaration for user-defined derived class
            user_ns_to_names[definition.user_namespace].add(definition.name)

        abstract_names = [
            definition.name
            for definition in self.schema.struct_definitions()
            if definition.abstract
        ]

        return "\n".join(
            [
                struct_forward_decls_template.render(ns_to_names=generated_ns_to_names),
                class_forward_decls_template.render(
                    ns_to_names=user_ns_to_names, abstract_names=abstract_names
                ),
            ]
        )

    def _emit_all_types_variant_decl(self):
        all_known_types = set()

        for struct in self.schema.struct_definitions():
            all_known_types.add(struct.user_fullname)
            for field in struct.fields:
                all_known_types.add(field.type)
                if isinstance(field, SequenceStructField):
                    all_known_types.add(field.underlying_type)

        return all_types_variant_template.render(
            namespace=self.schema.generated_namespace, all_types=all_known_types
        )
