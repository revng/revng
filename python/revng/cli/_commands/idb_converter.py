#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# type: ignore[attr-defined]
# type: ignore[name-defined]

import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import idb
import idb.analysis
import idb.fileformat
import idb.typeinf
import idb.typeinf_flags

import revng.model as m
from revng.cli.support import log_error
from revng.model.metaaddress import MetaAddressType

RevngTypes = Union[
    m.UnionType,
    m.StructType,
    m.PrimitiveType,
    m.EnumType,
    m.TypedefType,
    m.RawFunctionType,
    m.CABIFunctionType,
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

CONST_QUALIFIER = m.Qualifier(Kind=m.QualifierKind.Const)


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
        self.revng_types_by_id: Dict[int, RevngTypes] = {}
        self.idb_types_to_revng_types: Dict[int, m.QualifiedType] = {}
        self.functions: Set[m.Function] = set()
        self.dynamic_functions: List[m.DynamicFunction] = []
        self.imported_libraries: List[str] = []
        self.base_addr = base_addr
        self.verbose = verbose

        self._structs_to_fixup: Set[Tuple[m.StructType, idb.typeinf.TInfo]] = set()
        self._ordinal_types_to_fixup: Set[Tuple[m.QualifiedType, int]] = set()
        self._unions_to_fixup: Set[Tuple[m.UnionType, idb.typeinf.TInfo]] = set()

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
            # If the instrucitons are 16-bit long, it is a Thumb mode.
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
                qualified_revng_function_type = self._convert_idb_type_to_revng_type(
                    idb_function_type
                )
                revng_function_type = self.unwrap_qualified(qualified_revng_function_type)
                revng_function = m.Function(
                    OriginalName=function_name,
                    Entry=m.MetaAddress(Address=function_start_addr, Type=metaaddr_type),
                    Attributes=function_attributes,
                    Prototype=m.Reference.create(m.Binary, revng_function_type),
                )
            else:
                self.log(f"warning: Function {function_name} without a signature.")
                revng_function = m.Function(
                    OriginalName=function_name,
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
        revng_function_type = None
        if function_type is not None:
            qualified_revng_function_type = self._convert_idb_type_to_revng_type(function_type)
            revng_function_type = self.unwrap_qualified(qualified_revng_function_type)
        else:
            abi = revng_arch_to_abiname[self.arch]
            qualified_return_type = self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
            revng_function_type = m.CABIFunctionType(
                OriginalName=function_name,
                ABI=abi,
                ReturnType=qualified_return_type,
                Arguments=[],
            )
            self.revng_types_by_id[revng_function_type.ID] = revng_function_type
        dynamic_function = m.DynamicFunction(
            OriginalName=function_name,
            Prototype=m.Reference.create(m.Binary, revng_function_type),
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
                # TODO: Due to the bug mentioned bellow, we try sometimes to
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
            quialified_type, idb_ordinal_type = self._ordinal_types_to_fixup.pop()
            assert idb_ordinal_type in self.idb_types_to_revng_types
            real_quialified_type = self.idb_types_to_revng_types[idb_ordinal_type]

            the_revng_type = self.unwrap_qualified(real_quialified_type)
            revng_type_to_fix = self.unwrap_qualified(quialified_type)
            # TODO: For now, we found out that structs can be affected by this only.
            assert isinstance(the_revng_type, m.StructType) and isinstance(
                revng_type_to_fix, m.StructType
            )

            revng_type_to_fix.Fields = the_revng_type.Fields
            revng_type_to_fix.OriginalName = the_revng_type.OriginalName
            revng_type_to_fix.Size = the_revng_type.Size

    def _fixup_structs(self):
        while self._structs_to_fixup:
            revng_type, idb_type = self._structs_to_fixup.pop()

            assert isinstance(revng_type, m.StructType)
            assert idb_type.is_decl_struct()

            fields = []
            committed_size = 0

            # TODO: For now, we are ignoring structs that contain bitfields.
            # We should also improve the python-idb package to pickup the struct size directly from
            # the IDB files (e.g. to populate the `type.type_details.storage_size` for structs).
            for member in idb_type.type_details.members:
                if member.type.is_decl_bitfield():
                    self.log(
                        f"warning: Ignoring {revng_type.OriginalName} struct that contains a "
                        "bitfield."
                    )
                    return

            for member in idb_type.type_details.members:
                underlying_type = self._convert_idb_type_to_revng_type(member.type)
                revng_member = m.StructField(
                    OriginalName=member.name,
                    Type=underlying_type,
                    Offset=committed_size,
                )
                member_size = member.type.get_size()
                if member_size == 0:
                    self.log(
                        f"warning: Dropping zero-sized field {member.name} of struct "
                        f"{revng_type.OriginalName}."
                    )
                else:
                    fields.append(revng_member)
                    committed_size += member_size

            revng_type.Fields = fields
            revng_type.Size = committed_size

    def _fixup_unions(self):
        while self._unions_to_fixup:
            revng_type, idb_type = self._unions_to_fixup.pop()

            assert isinstance(revng_type, m.UnionType)
            assert idb_type.is_decl_union()

            fields = []
            for idx, member in enumerate(idb_type.type_details.members):
                qualified_type = self._convert_idb_type_to_revng_type(member.type)
                revng_member = m.UnionField(
                    OriginalName=member.name,
                    Type=qualified_type,
                    Index=idx,
                )
                fields.append(revng_member)

            revng_type.Fields = fields

    def _convert_idb_type_to_revng_type(
        self,
        type: idb.typeinf.TInfo,  # noqa: A002
        ordinal=None,
    ) -> m.QualifiedType:
        """Converts the given TInfo obtained from python-idb to the corresponding revng
        QualifiedType. If available, the integer identifying the type in the IDB (ordinal) should
        be supplied to allow handling circular references. If a type with the given ordinal was
        already converted the same instance is returned.
        """
        assert isinstance(type, idb.typeinf.TInfo)

        # Check if we already converted this type, and if so return the existing type.
        # Fundamental to handle circular dependencies.
        existing_revng_type = self.idb_types_to_revng_types.get(ordinal)
        if existing_revng_type is not None:
            return existing_revng_type

        type_name = type.get_name()
        revng_type_qualifiers: List[m.Qualifier] = []

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

            qualified_type = self._convert_idb_type_to_revng_type(
                aliased_type, ordinal=aliased_type_ordinal
            )

            revng_type = m.TypedefType(OriginalName=type_name, UnderlyingType=qualified_type)

        elif type.is_decl_enum():
            revng_underlying_type = self._get_primitive_type(
                m.PrimitiveTypeKind.Unsigned, type.type_details.storage_size
            )
            entries = []
            for member in type.type_details.members:
                member_value = member.value
                size_of_underlying_type_in_bits = type.type_details.storage_size * 8
                if member_value >= 2**size_of_underlying_type_in_bits:
                    self.log(
                        f"warning: Value {hex(member_value)} for enum member "
                        f"{type_name}.{member.name} out of range, ignoring it."
                    )
                    continue
                # TODO: We should keep the user comment which might exist in member.cmt.
                enum_entry = m.EnumEntry(
                    OriginalName=member.name,
                    Value=member_value,
                )
                entries.append(enum_entry)

            if len(entries) == 0:
                self.log(f"warning: An empty enum type: {type_name}.")
                revng_type = m.TypedefType(
                    OriginalName=type_name, UnderlyingType=revng_underlying_type
                )
            else:
                revng_type = m.EnumType(
                    OriginalName=type_name,
                    Entries=entries,
                    UnderlyingType=revng_underlying_type,
                )

        elif type.is_decl_struct():
            if type.type_details.ref is not None and type.type_details.ref.type_details.is_ordref:
                if type.type_details.ref.type_details.ordinal not in self.idb_types_to_revng_types:
                    # Make a placeholder for this, since we did not observe the type yet.
                    revng_type = m.StructType(OriginalName="", Size=0, Fields=[])
                    qualified_type = m.QualifiedType(
                        UnqualifiedType=m.Reference.create(m.Binary, revng_type),
                        Qualifiers=revng_type_qualifiers,
                    )
                    self.revng_types_by_id[revng_type.ID] = revng_type
                    self._ordinal_types_to_fixup.add(
                        (qualified_type, type.type_details.ref.type_details.ordinal)
                    )
                    return qualified_type
                else:
                    return self.idb_types_to_revng_types[type.type_details.ref.type_details.ordinal]
            else:
                # Empty structs will be considered as invalid types.
                if len(type.type_details.members) == 0:
                    self.log(f"warning: Ignoring empty struct {type_name}, typedef it to void*")
                    revng_void_type = self.unwrap_qualified(
                        self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
                    )
                    qualified_type = m.QualifiedType(
                        UnqualifiedType=m.Reference.create(m.Binary, revng_void_type),
                        Qualifiers={
                            m.Qualifier(
                                Kind=m.QualifierKind.Pointer, Size=get_pointer_size(self.arch)
                            )
                        },
                    )
                    revng_type = m.TypedefType(
                        OriginalName=type_name, UnderlyingType=qualified_type
                    )
                    self.revng_types_by_id[revng_type.ID] = revng_type
                    return qualified_type
                else:
                    # Struct members and size will be computed later.
                    revng_type = m.StructType(OriginalName=type_name, Size=0, Fields=[])
                    self._structs_to_fixup.add((revng_type, type))

        elif type.is_decl_union():
            if type.type_details.ref is not None and type.type_details.ref.type_details.is_ordref:
                if type.type_details.ref.type_details.ordinal not in self.idb_types_to_revng_types:
                    # Make a placeholder for this, since we did not observe the type yet.
                    revng_type = m.UnionType(OriginalName="", Size=0, Fields=[])
                    qualified_type = m.QualifiedType(
                        UnqualifiedType=m.Reference.create(m.Binary, revng_type),
                        Qualifiers=revng_type_qualifiers,
                    )
                    self.revng_types_by_id[revng_type.ID] = revng_type
                    self._ordinal_types_to_fixup.add(
                        (qualified_type, type.type_details.ref.type_details.ordinal)
                    )
                    return qualified_type
                else:
                    return self.idb_types_to_revng_types[type.type_details.ref.type_details.ordinal]
            else:
                revng_type = m.UnionType(OriginalName=type_name, Fields=[])
                if len(type.type_details.members) == 0:
                    self.log(f"warning: Ignoring empty union {type_name}, typedef it to void*")
                    revng_void_type = self.unwrap_qualified(
                        self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
                    )
                    qualified_type = m.QualifiedType(
                        UnqualifiedType=m.Reference.create(m.Binary, revng_void_type),
                        Qualifiers={
                            m.Qualifier(
                                Kind=m.QualifierKind.Pointer, Size=get_pointer_size(self.arch)
                            )
                        },
                    )
                    revng_type = m.TypedefType(
                        OriginalName=type_name, UnderlyingType=qualified_type
                    )
                    self.revng_types_by_id[revng_type.ID] = revng_type
                    return qualified_type
                else:
                    # Union members will be computed later.
                    self._unions_to_fixup.add((revng_type, type))
        elif type.is_decl_ptr():
            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.obj_type)
            revng_type = self.resolve_typeref(underlying_type.UnqualifiedType)
            revng_type_qualifiers = list(underlying_type.Qualifiers)

        elif type.is_decl_array():
            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.elem_type)
            revng_type = self.resolve_typeref(underlying_type.UnqualifiedType)
            revng_type_qualifiers = list(underlying_type.Qualifiers)

        elif type.is_decl_bool():
            size = type.get_size()
            revng_type = self.unwrap_qualified(
                self._get_primitive_type(m.PrimitiveTypeKind.Unsigned, size)
            )

        elif type.is_decl_int() or type.is_decl_floating():
            size = type.get_size()
            primitive_kind = get_primitive_kind(type)
            revng_type = self.unwrap_qualified(self._get_primitive_type(primitive_kind, size))

        elif type.is_decl_void():
            primitive_kind = m.PrimitiveTypeKind.Void
            size = 0
            if type.get_name() != "":
                # Treat this case as `typedef void someothername`.
                revng_void_type = self._get_primitive_type(primitive_kind, size)
                revng_type = m.TypedefType(OriginalName=type_name, UnderlyingType=revng_void_type)
            else:
                revng_type = self.unwrap_qualified(self._get_primitive_type(primitive_kind, size))

        elif type.is_decl_func():
            # TODO: handle non C-ABI functions.
            # We cannot handle stack arguments at the moment.
            assert type.type_details.stkargs is None

            idb_return_type = type.get_rettype()
            revng_return_type = self._convert_idb_type_to_revng_type(idb_return_type)

            arguments = []
            for idx, argument in enumerate(type.type_details.args):
                argument_qualified_type = self._convert_idb_type_to_revng_type(argument.type)
                revng_argument = m.Argument(
                    Index=idx,
                    Type=argument_qualified_type,
                    OriginalName=argument.name,
                )
                arguments.append(revng_argument)

            revng_type = m.CABIFunctionType(
                ABI=revng_arch_to_abi[self.arch],
                ReturnType=revng_return_type,
                Arguments=arguments,
                OriginalName=type_name,
            )

        elif type.is_decl_partial():
            # Represents an unknown or void type with a known size.
            assert type.get_size() != 0
            # The type should be compatible with being a primitive type.
            # NOTE: If we find a case where this is not satisifed, we can produce a char[].
            assert (
                type.get_size() == 1
                or type.get_size() == 2
                or type.get_size() == 4
                or type.get_size() == 8
                or type.get_size() == 10
                or type.get_size() == 16
            )
            revng_type = self.unwrap_qualified(
                self._get_primitive_type(
                    m.PrimitiveTypeKind.Generic,
                    type.get_size(),
                )
            )

        else:
            # IDA does not know anything about this type.
            # TODO: In some cases we should emit a void type (when the type is always used as a
            # pointer).
            size = type.get_size()
            if size == 0:
                revng_type = self.unwrap_qualified(
                    self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
                )
            else:
                kind = m.PrimitiveTypeKind.PointerOrNumber
                revng_type = self.unwrap_qualified(self._get_primitive_type(kind, size))

        existing_revng_type = self.revng_types_by_id.get(revng_type.ID)
        if existing_revng_type:
            # A type with this ID was already emitted, ensure we are returning the same instance.
            assert revng_type is existing_revng_type

        qualified_type = m.QualifiedType(
            UnqualifiedType=m.Reference.create(m.Binary, revng_type),
            Qualifiers=revng_type_qualifiers,
        )

        if type.is_decl_ptr():
            qualified_type.Qualifiers.insert(
                0, m.Qualifier(Kind=m.QualifierKind.Pointer, Size=type.get_size())
            )

        if type.is_decl_array():
            n_elements = type.type_details.n_elems
            if n_elements == 0:
                self.log(f"warning: Array {type_name} has invalid zero size.")
            qualified_type.Qualifiers.insert(
                0, m.Qualifier(Kind=m.QualifierKind.Array, Size=n_elements)
            )

        if type.is_decl_const():
            qualified_type.Qualifiers.insert(0, CONST_QUALIFIER)

        if ordinal is not None:
            self.idb_types_to_revng_types[ordinal] = qualified_type
        self.revng_types_by_id[revng_type.ID] = revng_type

        return qualified_type

    def _get_primitive_type(self, kind: m.PrimitiveTypeKind, size: int) -> m.QualifiedType:
        """Gets a primitive type, taking care to register it"""
        revng_type = m.PrimitiveType(PrimitiveKind=kind, Size=size)
        if revng_type.ID not in self.revng_types_by_id:
            self.revng_types_by_id[revng_type.ID] = revng_type

        return m.QualifiedType(UnqualifiedType=m.Reference.create(m.Binary, revng_type))

    def get_model(self) -> m.Binary:
        return m.Binary(
            # NOTE: We assume that the EntryPoint can be obtained from binary itself, so we use an
            # invalid address for it here.
            EntryPoint=m.MetaAddress(Address=0x0, Type=MetaAddressType.Invalid),
            Functions=list(self.functions),
            ImportedDynamicFunctions=list(self.dynamic_functions),
            Types=list(self.revng_types_by_id.values()),
            Architecture=self.arch,
            Segments=self.segments,
            ImportedLibraries=self.imported_libraries,
        )

    def resolve_typeref(self, typeref: m.Reference) -> Optional[RevngTypes]:
        return self.revng_types_by_id.get(typeref.id)

    def get_revng_type_by_name(self, name):
        for revng_type in self.revng_types_by_id.values():
            if revng_type.OriginalName == name:
                return revng_type
        return None

    def unwrap_qualified(self, qt: m.QualifiedType):
        if qt.Qualifiers:
            raise ValueError("Trying to unwrap qualified type with non empty qualifiers list!")

        return self.resolve_typeref(qt.UnqualifiedType)


def get_primitive_kind(idb_type: idb.typeinf.TInfo) -> m.PrimitiveTypeKind:
    if idb_type.is_decl_void():
        return m.PrimitiveTypeKind.Void
    elif idb_type.is_decl_floating():
        return m.PrimitiveTypeKind.Float
    elif idb.typeinf_flags.is_type_integral(idb_type.get_decltype()):
        if idb_type.is_signed():
            return m.PrimitiveTypeKind.Signed
        elif idb_type.is_unsigned():
            return m.PrimitiveTypeKind.Unsigned
        else:
            return m.PrimitiveTypeKind.Number
    elif idb_type.is_decl_ptr():
        return m.PrimitiveTypeKind.PointerOrNumber

    return m.PrimitiveTypeKind.Generic
