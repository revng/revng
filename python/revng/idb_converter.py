import re
from typing import Dict, List, Set, Tuple, Optional, Union

from loguru import logger

import idb
import idb.analysis
import idb.fileformat
import idb.typeinf
import idb.typeinf_flags
import revng.model.v1 as m

RevngTypes = Union[
    m.UnionType,
    m.StructType,
    m.PrimitiveType,
    m.EnumType,
    m.TypedefType,
    m.RawFunctionType,
    m.CABIFunctionType,
]

# TODO: map other idb procnames
idb_procname_to_revng_arch = {
    "metapc": m.Architecture.x86_64,
    "ARM": m.Architecture.arm,
    "mips": m.Architecture.mips,
    "mipsb": m.Architecture.mipsel,
}

revng_arch_to_metaaddr_code_type = {
    m.Architecture.x86: m.MetaAddressType.Code_x86,
    m.Architecture.x86_64: m.MetaAddressType.Code_x86_64,
    m.Architecture.arm: m.MetaAddressType.Code_arm,
    m.Architecture.aarch64: m.MetaAddressType.Code_aarch64,
    m.Architecture.mips: m.MetaAddressType.Code_mips,
    m.Architecture.mipsel: m.MetaAddressType.Code_mipsel,
    m.Architecture.systemz: m.MetaAddressType.Code_systemz,
}

revng_arch_to_abi = {
    m.Architecture.arm: m.ABI.Invalid,
    m.Architecture.aarch64: m.ABI.Invalid,
    m.Architecture.x86: m.ABI.SystemV_x86,
    m.Architecture.x86_64: m.ABI.SystemV_x86_64,
    m.Architecture.mips: m.ABI.SystemV_MIPS_o32,
    m.Architecture.mipsel: m.ABI.Invalid,
    m.Architecture.systemz: m.ABI.SystemZ_s390x,
}

CONST_QUALIFIER = m.Qualifier(Kind=m.QualifierKind.Const)


class IDBConverter:
    def __init__(self, input_idb: idb.fileformat.IDB):
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

        self._structs_to_fixup: Set[Tuple[m.StructType, idb.typeinf.TInfo]] = set()
        self._unions_to_fixup: Set[Tuple[m.UnionType, idb.typeinf.TInfo]] = set()

        self._import_types()
        self._fixup_structs()
        self._fixup_unions()
        self._import_functions()
        self._collect_imports()
        self._fixup_void_cabifunctiontype_arguments()

    def _import_types(self):
        """Imports initial types from the IDB. The types will be incomplete and need to be fixed"""
        til = self.idb.til
        for type_definition in til.types.defs:
            assert isinstance(type_definition, idb.typeinf.TILTypeInfo)
            type = type_definition.type
            assert isinstance(type, idb.typeinf.TInfo)
            self._convert_idb_type_to_revng_type(type, ordinal=type_definition.ordinal)

    def _import_functions(self):
        metaaddr_type = revng_arch_to_metaaddr_code_type[self.arch]

        for function_start_addr in self.api.idautils.Functions():
            function = idb.analysis.Function(self.idb, function_start_addr)
            function_name = function.get_name()

            # TODO: python-idb raises an exception when deserializing type info from some DBs, figure out why
            # (it tries to read a byte from an empty buffer)
            try:
                idb_function_type = function.get_signature()
            except Exception as e:
                idb_function_type = None

            if idb_function_type is not None:
                qualified_revng_function_type = self._convert_idb_type_to_revng_type(
                    idb_function_type
                )
                revng_function_type = self.unwrap_qualified(qualified_revng_function_type)
            else:
                # TODO: match ABI with arch
                abi = "SystemV_x86_64"
                qualified_return_type = self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
                revng_function_type = m.CABIFunctionType(
                    OriginalName=function_name,
                    ABI=abi,
                    ReturnType=qualified_return_type,
                    Arguments=[],
                )
                self.revng_types_by_id[revng_function_type.ID] = revng_function_type

            if self.api.ida_nalt.is_noret(function_start_addr):
                function_type = m.FunctionType.NoReturn
            else:
                function_type = m.FunctionType.Regular

            revng_function = m.Function(
                OriginalName=function_name,
                Entry=m.MetaAddress(Address=function_start_addr, Type=metaaddr_type),
                Type=function_type,
                Prototype=m.Reference.create(m.Binary, revng_function_type),
            )
            self.functions.add(revng_function)

    def _get_arch(self):
        inf_structure = self.api.idaapi.get_inf_structure()
        return idb_procname_to_revng_arch[inf_structure.procname]

    def _is_64_bit(self):
        inf_structure = self.api.idaapi.get_inf_structure()
        return inf_structure.is_64bit()

    def _collect_imports(self):
        # TODO: we're not getting type information for any imported symbol
        import_module_qty = self.api.ida_nalt.get_import_module_qty()
        if import_module_qty == 0:
            return

        for mod_index in range(self.api.ida_nalt.get_import_module_qty() + 1):

            def cb(function_addr, function_name, ignored_always_none):
                assert ignored_always_none is None
                function_type = self.get_revng_type_by_name(function_name)
                if function_type is None:
                    logger.warning(f"Type for imported function {function_name} not found")
                    return True

                assert isinstance(
                    function_type, (m.RawFunctionType, m.CABIFunctionType)
                ), f"Type mismatch for {function_name}: revng type for that symbol name is not a function"

                dynamic_function = m.DynamicFunction(
                    OriginalName=function_name,
                    Prototype=m.Reference.create(m.Binary, function_type),
                )
                self.dynamic_functions.append(dynamic_function)
                return True

            mod_name = self.api.ida_nalt.get_import_module_name(mod_index)
            if mod_name:
                self.imported_libraries.append(mod_name)
            self.api.ida_nalt.enum_import_names(mod_index, cb)

    def _fixup_void_cabifunctiontype_arguments(self):
        """Fixes CABIFunctionType objects with arguments of type void (functions with prototype `ret_type f(void, ...)`)
        by replacing the void argument with a generic primitive type.
        """
        for type_id, t in self.revng_types_by_id.items():
            if not isinstance(t, m.CABIFunctionType):
                continue

            for argument in t.Arguments:
                if len(argument.Type.Qualifiers) > 0:
                    continue

                argument_unqualified_type = self.resolve_typeref(argument.Type.UnqualifiedType)

                if not argument_unqualified_type:
                    continue

                if not isinstance(argument_unqualified_type, m.PrimitiveType):
                    continue

                if not argument_unqualified_type.PrimitiveKind == m.PrimitiveTypeKind.Void:
                    continue

                # TODO: pick appropriate size
                argument.Type = self._get_primitive_type(m.PrimitiveTypeKind.Generic, 8)

    def _fixup_structs(self):
        while self._structs_to_fixup:
            revng_type, idb_type = self._structs_to_fixup.pop()

            assert isinstance(revng_type, m.StructType)
            assert idb_type.is_decl_struct()

            fields = []
            committed_size = 0

            # Number of total aggregates
            aggregate_fields_count = 0
            # Size of the underlying type of the fields being aggregated
            aggregate_field_underlying_size = 0
            # Cumulative size of the fields pending aggregation
            accumulated_bitfield_width = 0

            def emit_aggregated_member(size, offset, field_number):
                underlying_primitive_type = self._get_primitive_type(
                    m.PrimitiveTypeKind.Generic, size
                )
                return m.StructField(
                    OriginalName=f"aggregate_{field_number}",
                    Type=underlying_primitive_type,
                    Offset=offset,
                )

            for member in idb_type.type_details.members:
                # We're dealing with a bitfield-type member
                if member.type.is_decl_bitfield():
                    # If we're not already aggregating start a new member
                    if aggregate_field_underlying_size == 0:
                        accumulated_bitfield_width = 0
                        aggregate_field_underlying_size = member.type.type_details.nbytes
                        aggregate_fields_count += 1

                    if (
                        accumulated_bitfield_width + member.type.type_details.width
                        > aggregate_field_underlying_size * 8
                    ):
                        # Emit aggregated member for the previous fields
                        aggregated_member = emit_aggregated_member(
                            aggregate_field_underlying_size,
                            committed_size,
                            aggregate_fields_count,
                        )
                        fields.append(aggregated_member)
                        committed_size += aggregate_field_underlying_size

                        # Start a new member
                        accumulated_bitfield_width = 0
                        aggregate_field_underlying_size = member.type.type_details.nbytes
                        aggregate_fields_count += 1

                    accumulated_bitfield_width += member.type.type_details.width
                    continue

                # We're dealing with a regular member
                else:
                    # If we were accumulating for one or more bitfields we must emit a field for them
                    if accumulated_bitfield_width > 0:
                        # Emit aggregated member for the previous fields
                        aggregated_member = emit_aggregated_member(
                            aggregate_field_underlying_size,
                            committed_size,
                            aggregate_fields_count,
                        )
                        fields.append(aggregated_member)
                        committed_size += aggregate_field_underlying_size

                        # Start a new member
                        accumulated_bitfield_width = 0
                        aggregate_field_underlying_size = 0
                        aggregate_fields_count += 1

                    underlying_type = self._convert_idb_type_to_revng_type(member.type)

                    revng_member = m.StructField(
                        OriginalName=member.name,
                        Type=underlying_type,
                        Offset=committed_size,
                    )
                    member_size = member.type.get_size()
                    if member_size == 0:
                        logger.warning(
                            f"Dropping zero-sized field {member.name} of struct {revng_type.OriginalName}"
                        )
                    else:
                        fields.append(revng_member)
                        committed_size += member_size

            # Emit final aggregated member
            if accumulated_bitfield_width > 0:
                aggregated_member = emit_aggregated_member(
                    aggregate_field_underlying_size,
                    committed_size,
                    aggregate_fields_count,
                )
                fields.append(aggregated_member)
                committed_size += aggregate_field_underlying_size

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
        type: idb.typeinf.TInfo,
        ordinal: Optional[int] = None,
    ) -> m.QualifiedType:
        """Converts the given TInfo obtained from python-idb to the corresponding revng QualifiedType.

        If available, the integer identifying the type in the IDB (ordinal) should be supplied to allow handling
        circular references. If a type with the given ordinal was already converted the same instance is returned.
        """
        assert isinstance(type, idb.typeinf.TInfo)

        # Check if we already converted this type, and if so return the existing type.
        # Fundamental to handle circular dependencies
        existing_revng_type = self.idb_types_to_revng_types.get(ordinal)
        if existing_revng_type is not None:
            return existing_revng_type

        type_name = type.get_name()
        revng_type_qualifiers = []

        if type.is_decl_typedef():
            aliased_type = type.get_next_tinfo()

            # Special case reserved typedefs
            match = re.fullmatch(r"(?P<unsigned>u)?int(?P<width>(8|16|32|64))_t", type_name)
            if match:
                unsigned = bool(match["unsigned"])
                size = int(match["width"]) // 8
                primitive_kind = (
                    m.PrimitiveTypeKind.Unsigned if unsigned else m.PrimitiveTypeKind.Signed
                )
                qualified_type = self._get_primitive_type(primitive_kind, size)
            elif type_name == "size_t":
                primitive_kind = m.PrimitiveTypeKind.Unsigned
                size = 8 if self.is64bit else 4
                qualified_type = self._get_primitive_type(primitive_kind, size)
            else:
                aliased_type_ordinal = None
                if type.type_details.is_ordref:
                    aliased_type_ordinal = type.type_details.ordinal
                else:
                    aliased_tiltypeinfo = type.til.types.find_by_name(aliased_type.name)
                    if aliased_tiltypeinfo:
                        aliased_type_ordinal = aliased_tiltypeinfo.ordinal

                if aliased_type_ordinal is None:
                    aliased_type = type.get_next_tinfo()

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
                if member_value >= 2**64:
                    logger.warning(
                        f"Value {hex(member_value)} for enum member {type_name}.{member.name} out of range, defaulting to 0"
                    )
                    member_value = 0
                # TODO: we should keep the user comment which might exist in member.cmt
                enum_entry = m.EnumEntry(
                    OriginalName=member.name,
                    Value=member_value,
                )
                entries.append(enum_entry)

            if len(entries) == 0:
                logger.warning(f"Inserting a dummy member into enum {type_name}")
                entries.append(
                    m.EnumEntry(
                        OriginalName="dummy",
                        Value=0,
                    )
                )

            revng_type = m.EnumType(
                OriginalName=type_name,
                Entries=entries,
                UnderlyingType=revng_underlying_type.UnqualifiedType,
            )

        elif type.is_decl_struct():
            if type.type_details.ref is not None and type.type_details.ref.type_details.is_ordref:
                return self.idb_types_to_revng_types[type.type_details.ref.type_details.ordinal]

            # Empty struct are illegal -- emit them as a typedef void <struct>
            elif len(type.type_details.members) == 0:
                logger.warning(
                    f"Found invalid empty struct {type_name}, replacing with a void* typedef"
                )
                underlying_type = self._get_primitive_type(m.PrimitiveTypeKind.Void, 0)
                underlying_type.Qualifiers.append(
                    m.Qualifier(
                        Kind=m.QualifierKind.Pointer,
                        Size=8 if self.is64bit else 4,
                    )
                )
                revng_type = m.TypedefType(OriginalName=type_name, UnderlyingType=underlying_type)
            else:
                # Struct members and size will be computed later
                revng_type = m.StructType(OriginalName=type_name, Size=0, Fields=[])
                self._structs_to_fixup.add((revng_type, type))

        elif type.is_decl_union():
            # Union members will be computed later
            revng_type = m.UnionType(OriginalName=type_name, Fields=[])
            self._unions_to_fixup.add((revng_type, type))

        elif type.is_decl_ptr():
            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.obj_type)
            revng_type = self.resolve_typeref(underlying_type.UnqualifiedType)
            revng_type_qualifiers = [q for q in underlying_type.Qualifiers]

        elif type.is_decl_array():
            underlying_type = self._convert_idb_type_to_revng_type(type.type_details.elem_type)
            revng_type = self.resolve_typeref(underlying_type.UnqualifiedType)
            revng_type_qualifiers = [q for q in underlying_type.Qualifiers]

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
                # Treat this case as `typedef void someothername`
                revng_void_type = self._get_primitive_type(primitive_kind, size)
                revng_type = m.TypedefType(OriginalName=type_name, UnderlyingType=revng_void_type)
            else:
                revng_type = self.unwrap_qualified(self._get_primitive_type(primitive_kind, size))

        elif type.is_decl_func():
            # TODO: handle non C-ABI functions
            # We cannot handle stack arguments at the moment
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

            # TODO: pick ABI properly
            revng_type = m.CABIFunctionType(
                ABI=revng_arch_to_abi[self.arch],
                ReturnType=revng_return_type,
                Arguments=arguments,
                OriginalName=type_name,
            )

        elif type.is_decl_partial():
            # Represents an unknown or void type with a known size
            revng_type = self.unwrap_qualified(
                self._get_primitive_type(
                    m.PrimitiveTypeKind.Generic,
                    type.get_size(),
                )
            )

        else:
            # IDA does not know anything about this type
            # TODO: In some cases we should emit a void type (when the type is always used as a pointer)
            kind = m.PrimitiveTypeKind.PointerOrNumber
            size = type.get_size()
            if size == 0:
                # Fallback
                size = 8 if self.is64bit else 4

            revng_type = self.unwrap_qualified(self._get_primitive_type(kind, size))

        existing_revng_type = self.revng_types_by_id.get(revng_type.ID)
        if existing_revng_type:
            # A type with this ID was already emitted, ensure we are returning the same instance
            assert revng_type is existing_revng_type

        qualified_type = m.QualifiedType(
            m.Reference.create(m.Binary, revng_type), Qualifiers=revng_type_qualifiers
        )

        if type.is_decl_ptr():
            qualified_type.Qualifiers.insert(
                0, m.Qualifier(Kind=m.QualifierKind.Pointer, Size=type.get_size())
            )

        if type.is_decl_array():
            n_elements = type.type_details.n_elems
            if n_elements == 0:
                logger.warning(f"Array {type_name} has invalid zero size")
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

        return m.QualifiedType(m.Reference.create(m.Binary, revng_type))

    def get_model(self) -> m.Binary:
        return m.Binary(
            # TODO: recover entrypoint from the IDB.
            # idb.analysis.enumerate_entrypoints returns a not very useful list
            EntryPoint=m.MetaAddress(Address=0x0, Type=m.MetaAddressType.Invalid),
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
        # TODO: can we ever reach this point?
        return m.PrimitiveTypeKind.PointerOrNumber
    elif idb_type.is_decl_bitfield():
        return m.PrimitiveTypeKind.Generic

    # TODO: should we raise an exception or return Invalid here instead?
    return m.PrimitiveTypeKind.Generic
