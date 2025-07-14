#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# mypy: disable-error-code="attr-defined,name-defined"

import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import idb
import idb.analysis
import idb.fileformat
import idb.typeinf
import idb.typeinf_flags

import revng.model as m
from revng.model.metaaddress import MetaAddressType
from revng.support import log_error

RevngTypeDefinitions = Union[
    m.UnionDefinition,
    m.StructDefinition,
    m.EnumDefinition,
    m.TypedefDefinition,
    m.CABIFunctionDefinition,
    # Omit RFDs, as this importer never emits them.
    # m.RawFunctionDefinition,
]

idb_procname_to_revng_arch = {
    "x86_64": m.Architecture.x86_64,
    "x86": m.Architecture.x86,
    "arm": m.Architecture.arm,
    "aarch64": m.Architecture.aarch64,
    "mips": m.Architecture.mips,
    "mipsb": m.Architecture.mips,
    "mipsl": m.Architecture.mipsel,
    "s390x": m.Architecture.systemz,
}

revng_arch_to_metaaddr_code_type = {
    m.Architecture.x86: MetaAddressType.Code_x86,
    m.Architecture.x86_64: MetaAddressType.Code_x86_64,
    m.Architecture.arm: MetaAddressType.Code_arm,
    m.Architecture.aarch64: MetaAddressType.Code_aarch64,
    m.Architecture.mips: MetaAddressType.Code_mips,
    m.Architecture.mipsel: MetaAddressType.Code_mipsel,
    m.Architecture.systemz: MetaAddressType.Code_systemz,
}

revng_arch_to_abi = {
    m.Architecture.arm: m.ABI.AAPCS,
    m.Architecture.aarch64: m.ABI.AAPCS64,
    m.Architecture.x86: m.ABI.SystemV_x86,
    m.Architecture.x86_64: m.ABI.SystemV_x86_64,
    m.Architecture.mips: m.ABI.SystemV_MIPS_o32,
    m.Architecture.mipsel: m.ABI.SystemV_MIPSEL_o32,
    m.Architecture.systemz: m.ABI.SystemZ_s390x,
}

revng_arch_to_abiname = {
    m.Architecture.arm: "AAPCS",
    m.Architecture.aarch64: "AAPCS64",
    m.Architecture.x86: "SystemV_x86",
    m.Architecture.x86_64: "SystemV_x86_64",
    m.Architecture.mips: "SystemV_MIPS_o32",
    m.Architecture.mipsel: "SystemV_MIPSEL_o32",
    m.Architecture.systemz: "SystemZ_s390x",
}


def get_pointer_size(architecture):
    match architecture:  # noqa: R503
        case m.Architecture.x86 | m.Architecture.arm | m.Architecture.mips | m.Architecture.mipsel:
            return 4
        case m.Architecture.x86_64 | m.Architecture.aarch64 | m.Architecture.systemz:
            return 8
        case default:  # noqa: F841
            return None


class IDBConverter:
    def __init__(self, input_idb: idb.fileformat.IDB, base_addr, verbose):
        self.idb: idb.fileformat.IDB = input_idb
        self.api = idb.IDAPython(self.idb)

        self.arch: m.Architecture = self._get_arch()
        self.is64bit: bool = self._is_64_bit()
        self.segments: List[m.Segment] = []
        self.revng_types_by_id: Dict[str, RevngTypeDefinitions] = {}
        self.idb_types_to_revng_types: Dict[int, m.Type] = {}
        self.functions: Set[m.Function] = set()
        self.dynamic_functions: List[m.DynamicFunction] = []
        self.imported_libraries: List[str] = []
        self.base_addr = base_addr
        self.verbose = verbose

        self._structs_to_fixup: Set[Tuple[m.StructDefinition, idb.typeinf.TInfo]] = set()
        self._ordinal_types_to_fixup: Set[Tuple[m.Type, int]] = set()
        self._unions_to_fixup: Set[Tuple[m.UnionDefinition, idb.typeinf.TInfo]] = set()

        self._import_types()
        self._fixup_structs()
        self._fixup_unions()
        self._fixup_ordinal_types()
        self._import_functions()
        self._collect_imports()

    def log(self, message):
        if self.verbose:
            sys.stderr.write(message + "\n")

    def _import_types(self):
        """Imports initial types from the IDB. The types will be incomplete and need to be fixed"""
        til = self.idb.til
        for type_definition in til.types.defs:
            assert isinstance(type_definition, idb.typeinf.TILTypeInfo)
            the_type = type_definition.type
            assert isinstance(the_type, idb.typeinf.TInfo)
            self._convert_idb_type_to_revng_type(the_type, ordinal=type_definition.ordinal)

    def _import_functions(self):
        metaaddr_type = revng_arch_to_metaaddr_code_type[self.arch]
        if self.arch == m.Architecture.arm and self.api.idc.ItemSize(0x0) == 2:
            # If the instructions are 16-bit long, it is a Thumb mode.
            metaaddr_type = MetaAddressType.Code_arm_thumb

        for function_start_addr in self.api.idautils.Functions():
            function = idb.analysis.Function(self.idb, function_start_addr)
            function_name = function.get_name()

            idb_function_type = function.get_signature()

            function_attributes: List[m.FunctionAttribute] = []

            if self.api.ida_nalt.is_noret(function_start_addr):
                function_attributes.insert(0, m.FunctionAttribute.NoReturn)

            function_start_addr += self.base_addr

            if idb_function_type is not None:
                prototype = self._convert_idb_type_to_revng_type(idb_function_type)
                revng_function = m.Function(
                    Name=function_name,
                    Entry=m.MetaAddress(Address=function_start_addr, Type=metaaddr_type),
                    Attributes=function_attributes,
                    Prototype=prototype,
                )
            else:
                self.log(f"warning: Function {function_name} without a signature.")
                revng_function = m.Function(
                    Name=function_name,
                    Entry=m.MetaAddress(Address=function_start_addr, Type=metaaddr_type),
                    Attributes=function_attributes,
                )

            self.functions.add(revng_function)

    def _get_arch(self):
        inf_structure = self.api.idaapi.get_inf_structure()
        procname = inf_structure.procname.lower()

        if procname == "arm" and inf_structure.is_64bit():
            procname = "aarch64"
        elif procname == "metapc" and inf_structure.is_64bit():
            procname = "x86_64"
        elif procname == "metapc" and inf_structure.is_32bit():
            procname = "x86"

        return idb_procname_to_revng_arch[procname]

    def _is_64_bit(self):
        inf_structure = self.api.idaapi.get_inf_structure()
        return inf_structure.is_64bit()

    def _import_names_helper(self, function_addr, function_name):
        function = idb.analysis.Function(self.idb, function_addr)
        if not function:
            log_error(f"Unable to find function {function_name}.")
            return True
        function_name = function.get_name()
        function_type = None
        try:
            function_type = function.get_signature()
        except Exception as exception:
            log_error(f"Unable to parse function type for {function_name}.")
            log_error(str(exception))
            function_type = None
            return True
        if function_type is not None:
            prototype = self._convert_idb_type_to_revng_type(function_type)
        else:
            prototype_definition = m.CABIFunctionDefinition(
                ABI=revng_arch_to_abiname[self.arch],
                Arguments=[],
            )
            self.revng_types_by_id[prototype_definition.key()] = prototype_definition
            prototype = self._type_for_definition(prototype_definition)
        dynamic_function = m.DynamicFunction(
            Name=function_name,
            Prototype=prototype,
        )
        self.dynamic_functions.append(dynamic_function)
        return True

    def _collect_imports(self):
        # NOTE: If @plt was used, the functions are in Functions() already, but the name starts
        # with ".", and we do import types for them during parsing of regular/local functions.

        # NOTE: We cannot use api.ida_nalt.get_import_module_qty() here, since it relies
        # on `library names`, but IDA prints `.dynsym` as library for each library in case of ELF.

        for mod_index in range(self.api.ida_nalt.get_import_module_qty() + 1):

            def import_names_callback(function_addr, function_name, ignored_always_none):
                assert ignored_always_none is None
                return self._import_names_helper(function_addr, function_name)

            try:
                mod_name = self.api.ida_nalt.get_import_module_name(mod_index)
            except KeyError as exception:
                # TODO: Due to the bug mentioned below, we try sometimes to
                # find imported modules that do not exist.
                if mod_index != 0:
                    # It would be very strange if the mod_index is not 0.
                    log_error(f"Unable to find module with index: {mod_index}.")
                    log_error(str(exception))
                continue
            # NOTE: IDA 7 has a bug when reporting imported library names - prints `.dynsym` for
            # each import.
            if mod_name and mod_name != ".dynsym":
                self.imported_libraries.append(mod_name)
            self.api.ida_nalt.enum_import_names(mod_index, import_names_callback)

    def _fixup_ordinal_types(self):
        """Fixes some types that were referring an ordinal that was not observed yet."""
        while self._ordinal_types_to_fixup:
            placeholder_type, idb_ordinal_type = self._ordinal_types_to_fixup.pop()
            assert idb_ordinal_type in self.idb_types_to_revng_types
            real_type = self.idb_types_to_revng_types[idb_ordinal_type]

            placeholder_definition = self.unwrap_definition(placeholder_type)
            real_definition = self.unwrap_definition(real_type)
            # TODO: Until now, we only ever saw this affecting structs.
            assert isinstance(placeholder_definition, m.StructDefinition) and isinstance(
                real_definition, m.StructDefinition
            )

            placeholder_definition.Fields = real_definition.Fields
            placeholder_definition.Name = real_definition.Name
            placeholder_definition.Size = real_definition.Size

    def _fixup_structs(self):
        while self._structs_to_fixup:
            revng_type, idb_type = self._structs_to_fixup.pop()

            assert isinstance(revng_type, m.StructDefinition)
            assert idb_type.is_decl_struct()

            fields = []
            committed_size = 0

            # TODO: For now, we are ignoring structs that contain bitfields.
            # We should also improve the python-idb package to pickup the struct size directly from
            # the IDB files (e.g. to populate the `type.type_details.storage_size` for structs).
            for member in idb_type.type_details.members:
                if member.type.is_decl_bitfield():
                    self.log(
                        f"warning: Ignoring {revng_type.Name} struct that contains a " "bitfield."
                    )
                    return

            for member in idb_type.type_details.members:
                underlying_type = self._convert_idb_type_to_revng_type(member.type)
                revng_member = m.StructField(
                    Name=member.name,
                    Type=underlying_type,
                    Offset=committed_size,
                )
                member_size = member.type.get_size()
                if member_size == 0:
                    self.log(
                        f"warning: Dropping zero-sized field {member.name} of struct "
                        f"{revng_type.Name}."
                    )
                else:
                    fields.append(revng_member)
                    committed_size += member_size

            revng_type.Fields = fields
            revng_type.Size = committed_size

    def _fixup_unions(self):
        while self._unions_to_fixup:
            revng_type, idb_type = self._unions_to_fixup.pop()

            assert isinstance(revng_type, m.UnionDefinition)
            assert idb_type.is_decl_union()

            fields = []
            for idx, member in enumerate(idb_type.type_details.members):
                qualified_type = self._convert_idb_type_to_revng_type(member.type)
                revng_member = m.UnionField(
                    Name=member.name,
                    Type=qualified_type,
                    Index=idx,
                )
                fields.append(revng_member)

            revng_type.Fields = fields

    def _convert_idb_type_to_revng_type(
        self,
        type: idb.typeinf.TInfo,  # noqa: A002
        ordinal=None,
    ) -> m.Type:
        """
        Converts the given TInfo obtained from python-idb to the corresponding revng type.
        If available, the integer identifying the type in the IDB (ordinal) should be supplied
        to allow handling circular references. If a type with the given ordinal was already
        converted the same instance is returned.
        """
        assert isinstance(type, idb.typeinf.TInfo)

        # Check if we already converted this type, and if so return the existing type.
        # Fundamental to handle circular dependencies.
        existing_revng_type = self.idb_types_to_revng_types.get(ordinal)
        if existing_revng_type is not None:
            return existing_revng_type

        type_name = type.get_name()

        resulting_definition = None
        if type.is_decl_typedef():
            aliased_type = type.get_final_tinfo()

            # IDA's types could be generated/identified in two ways: by ordinal and by names,
            # so we handle both ways here.
            aliased_type_ordinal = None
            if type.type_details.is_ordref:
                aliased_type_ordinal = type.type_details.ordinal
            else:
                aliased_tiltypeinfo = type.til.types.find_by_name(aliased_type.name)
                if aliased_tiltypeinfo:
                    aliased_type_ordinal = aliased_tiltypeinfo.ordinal

            underlying = self._convert_idb_type_to_revng_type(
                aliased_type, ordinal=aliased_type_ordinal
            )

            # IDA uses c-like type system in that for each struct/union a corresponding typedef
            # is introduced. In our model type system, such typedefs are already included
            # in the struct/union definitions (we emit them automatically for each aggregate, with
            # no need for an explicit typedef - de facto, always introducing their names into
            # the global namespace).
            #
            # As such, when importing from IDA, unless a typedef carries any extra information,
            # there's no point of importing it (it's harmful, actually, as it introduces multiple
            # types sharing the same name) - so we just skip it here.

            if (
                aliased_type.get_name() == type_name
                and type.is_decl_const() == aliased_type.is_decl_const()
                and isinstance(underlying, m.DefinedType)
            ):
                # skip unnecessary typedefs
                resulting_definition = self.unwrap_definition(underlying)
            else:
                resulting_definition = m.TypedefDefinition(
                    Name=type_name, UnderlyingType=underlying
                )

        elif type.is_decl_enum():
            underlying = m.PrimitiveType(
                PrimitiveKind=m.PrimitiveKind.Unsigned,
                Size=type.type_details.storage_size,
                IsConst=False,
            )
            entries = []
            for member in type.type_details.members:
                if member.value >= 2 ** (underlying.Size * 8):
                    self.log(
                        f"warning: Value {hex(member.value)} for enum member "
                        f"{type_name}.{member.name} out of range, ignoring it."
                    )
                    continue
                # TODO: We should keep the user comment which might exist in member.cmt.
                enum_entry = m.EnumEntry(
                    Name=member.name,
                    Value=member.value,
                )
                entries.append(enum_entry)

            if len(entries) == 0:
                self.log(f"warning: An empty enum type: {type_name}, emitting a typedef instead.")
                resulting_definition = m.TypedefDefinition(
                    Name=type_name, UnderlyingType=underlying
                )
            else:
                resulting_definition = m.EnumDefinition(
                    Name=type_name,
                    Entries=entries,
                    UnderlyingType=underlying,
                )

        elif type.is_decl_struct():
            if type.type_details.ref is not None and type.type_details.ref.type_details.is_ordref:
                if type.type_details.ref.type_details.ordinal not in self.idb_types_to_revng_types:
                    # This is just a forward declaration. Make a placeholder for now.
                    resulting_definition = m.StructDefinition(Name="", Size=0, Fields=[])
                    self.revng_types_by_id[resulting_definition.key()] = resulting_definition

                    wrapped = self._type_for_definition(resulting_definition, type.is_decl_const())

                    self._ordinal_types_to_fixup.add(
                        (wrapped, type.type_details.ref.type_details.ordinal)
                    )

                    return wrapped
                else:
                    return self.idb_types_to_revng_types[type.type_details.ref.type_details.ordinal]
            else:
                # Empty structs are not valid in the revng type system.
                if len(type.type_details.members) == 0:
                    self.log(f"warning: Ignoring empty struct {type_name}, typedef it to `void`")
                    typedef = m.TypedefDefinition(
                        Name=type_name,
                        UnderlyingType=m.PrimitiveType(
                            PrimitiveKind=m.PrimitiveKind.Void, Size=0, IsConst=False
                        ),
                    )
                    self.revng_types_by_id[typedef.key()] = typedef
                    return self._type_for_definition(typedef, type.is_decl_const())
                else:
                    # Struct members and size will be computed later.
                    resulting_definition = m.StructDefinition(Name=type_name, Size=0, Fields=[])
                    self._structs_to_fixup.add((resulting_definition, type))

        elif type.is_decl_union():
            if type.type_details.ref is not None and type.type_details.ref.type_details.is_ordref:
                if type.type_details.ref.type_details.ordinal not in self.idb_types_to_revng_types:
                    # This is just a forward declaration. Make a placeholder for now.
                    resulting_definition = m.UnionDefinition(Name="", Fields=[])
                    self.revng_types_by_id[resulting_definition.key()] = resulting_definition

                    wrapped = self._type_for_definition(resulting_definition, type.is_decl_const())
                    self._ordinal_types_to_fixup.add(
                        (wrapped, type.type_details.ref.type_details.ordinal)
                    )

                    return wrapped
                else:
                    return self.idb_types_to_revng_types[type.type_details.ref.type_details.ordinal]
            else:
                # Empty unions are not valid in the revng type system.
                if len(type.type_details.members) == 0:
                    self.log(f"warning: Ignoring empty union {type_name}, typedef it to `void`")
                    typedef = m.TypedefDefinition(
                        Name=type_name,
                        UnderlyingType=m.PrimitiveType(
                            PrimitiveKind=m.PrimitiveKind.Void, Size=0, IsConst=False
                        ),
                    )
                    self.revng_types_by_id[typedef.key()] = typedef
                    return self._type_for_definition(typedef, type.is_decl_const())
                else:
                    # Union members will be computed later.
                    resulting_definition = m.UnionDefinition(Name=type_name, Fields=[])
                    self._unions_to_fixup.add((resulting_definition, type))

        elif type.is_decl_ptr():
            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.obj_type)
            result = m.PointerType(
                PointeeType=underlying_type,
                PointerSize=type.get_size(),
                IsConst=type.is_decl_const(),
            )
            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        elif type.is_decl_array():
            if type.type_details.n_elems == 0:
                self.log(f"warning: Array {type_name} has invalid zero size.")

            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.obj_type)
            result = m.ArrayType(
                ElementType=underlying_type,
                ElementCount=type.type_details.n_elems,
                IsConst=type.is_decl_const(),
            )
            if type.is_decl_const():
                result.IsConst = True
            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        elif type.is_decl_bool():
            result = m.PrimitiveType(
                PrimitiveKind=m.PrimitiveKind.Unsigned,
                Size=type.get_size(),
                IsConst=type.is_decl_const(),
            )
            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        elif type.is_decl_int() or type.is_decl_floating():
            result = m.PrimitiveType(
                PrimitiveKind=get_primitive_kind(type),
                Size=type.get_size(),
                IsConst=type.is_decl_const(),
            )
            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        elif type.is_decl_void():
            result = m.PrimitiveType(
                PrimitiveKind=m.PrimitiveKind.Void, Size=0, IsConst=type.is_decl_const()
            )

            if type.get_name() != "":
                # Treat this as `typedef void someothername`.
                result = m.TypedefDefinition(
                    Name=type_name, UnderlyingType=result, IsConst=type.is_decl_const()
                )

            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        elif type.is_decl_func():
            # TODO: handle non C-ABI functions.

            # We cannot handle stack arguments at the moment.
            assert type.type_details.stkargs is None

            arguments = []
            for idx, argument in enumerate(type.type_details.args):
                arguments.append(
                    m.Argument(
                        Index=idx,
                        Type=self._convert_idb_type_to_revng_type(argument.type),
                        Name=argument.name,
                    )
                )

            rt = self._convert_idb_type_to_revng_type(type.get_rettype())
            if isinstance(rt, m.PrimitiveType) and rt.PrimitiveKind == m.PrimitiveKind.Void:
                rt = None

            resulting_definition = m.CABIFunctionDefinition(
                ABI=revng_arch_to_abi[self.arch],
                ReturnType=rt,
                Arguments=arguments,
            )

        elif type.is_decl_partial():
            # Represents an unknown or void type with a known size.
            assert type.get_size() != 0
            # The type should be compatible with being a primitive type.
            # NOTE: If we find a case where this is not satisfied, we can produce a char[].
            assert (
                type.get_size() == 1
                or type.get_size() == 2
                or type.get_size() == 4
                or type.get_size() == 8
                or type.get_size() == 10
                or type.get_size() == 16
            )

            result = m.PrimitiveType(
                PrimitiveKind=m.PrimitiveKind.Generic,
                Size=type.get_size(),
                IsConst=type.is_decl_const(),
            )
            if ordinal is not None:
                self.idb_types_to_revng_types[ordinal] = result
            return result

        else:
            # IDA does not know anything about this type.
            # TODO: In some cases we should emit a void type (e.g. when the type is only ever used
            # as a pointer).
            if type.get_size() == 0:
                result = m.PrimitiveType(
                    PrimitiveKind=m.PrimitiveKind.Void, Size=0, IsConst=type.is_decl_const()
                )
                if ordinal is not None:
                    self.idb_types_to_revng_types[ordinal] = result
                return result
            else:
                result = m.PrimitiveType(
                    PrimitiveKind=m.PrimitiveKind.PointerOrNumber,
                    Size=type.get_size(),
                    IsConst=type.is_decl_const(),
                )
                if ordinal is not None:
                    self.idb_types_to_revng_types[ordinal] = result
                return result

        assert resulting_definition is not None

        result = self._type_for_definition(resulting_definition, type.is_decl_const())
        if ordinal is not None:
            self.idb_types_to_revng_types[ordinal] = result

        if resulting_definition.key() not in self.revng_types_by_id:
            self.revng_types_by_id[resulting_definition.key()] = resulting_definition

        return result

    def _type_for_definition(self, definition: m.TypeDefinition, is_const: bool = False) -> m.Type:
        return m.DefinedType(
            Definition=m.Reference("/TypeDefinitions/" + definition.key()), IsConst=is_const
        )

    def get_model(self) -> m.Binary:
        return m.Binary(
            # NOTE: We assume that the EntryPoint can be obtained from binary itself, so we use an
            # invalid address for it here.
            EntryPoint=m.MetaAddress(Address=0x0, Type=MetaAddressType.Invalid),
            Functions=list(self.functions),
            ImportedDynamicFunctions=list(self.dynamic_functions),
            TypeDefinitions=list(self.revng_types_by_id.values()),
            Architecture=self.arch,
            Segments=self.segments,
            ImportedLibraries=self.imported_libraries,
        )

    def get_revng_type_by_name(self, name):
        for revng_type in self.revng_types_by_id.values():
            if revng_type.Name == name:
                return revng_type
        return None

    def unwrap_definition(self, defined_type: m.Type) -> Optional[RevngTypeDefinitions]:
        if not isinstance(defined_type, m.DefinedType):
            raise ValueError("Trying to unwrap a nested definition:\n" + str(defined_type))

        return self.revng_types_by_id.get(defined_type.Definition.id)


def get_primitive_kind(idb_type: idb.typeinf.TInfo) -> m.PrimitiveKind:
    if idb_type.is_decl_void():
        return m.PrimitiveKind.Void
    elif idb_type.is_decl_floating():
        return m.PrimitiveKind.Float
    elif idb.typeinf_flags.is_type_integral(idb_type.get_decltype()):
        if idb_type.is_signed():
            return m.PrimitiveKind.Signed
        elif idb_type.is_unsigned():
            return m.PrimitiveKind.Unsigned
        else:
            return m.PrimitiveKind.Number
    elif idb_type.is_decl_ptr():
        return m.PrimitiveKind.PointerOrNumber

    return m.PrimitiveKind.Generic
