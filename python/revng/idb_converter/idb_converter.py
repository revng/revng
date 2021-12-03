import copy
import re
from typing import Dict, List, Set, Optional, Union

from loguru import logger

import idb
import idb.analysis
import idb.fileformat
import idb.typeinf
import idb.typeinf_flags
import revng.model as m

from .utils import sanitize_identifier


# TODO: emit const qualifiers (if possible)
RevngTypes = Union[
    m.Union,
    m.Struct,
    m.Primitive,
    m.Enum,
    m.Typedef,
    m.RawFunctionType,
    m.CABIFunctionType,
]


# TODO: map other idb procnames
idb_procname_to_revng_arch = {
    "metapc": m.Architecture.x86_64,
    "ARM": m.Architecture.arm,
}

revng_arch_to_metaaddr_code_type = {
    m.Architecture.x86_64: m.MetaAddressType.Code_x86_64,
    m.Architecture.arm: m.MetaAddressType.Code_arm,
}

revng_arch_to_abi = {
    m.Architecture.x86_64: m.ABI.SystemV_x86_64,
}


class IDBConverter:
    def __init__(self, input_idb: idb.fileformat.IDB, mangle_names=True):
        self.idb: idb.fileformat.IDB = input_idb
        self.mangle_names = mangle_names

        self.arch: m.Architecture = self._get_arch()
        self.is64bit: bool = self._is_64_bit()
        self.entrypoint = self._get_entrypoint()
        self.segments: List[m.Segment] = []
        self.name_to_revng_type: Dict[str, RevngTypes] = {}
        self.revng_types_by_id: Dict[int, RevngTypes] = {}
        self.name_to_idb_type: Dict[str, idb.typeinf.TInfo] = {}
        self.revng_types_to_idb_types: Dict[RevngTypes, idb.typeinf.TInfo] = {}
        self.idb_types_to_revng_types: Dict[idb.typeinf.TInfo, RevngTypes] = {}
        self.functions: Set[m.Function] = set()
        self.dynamic_functions: List[m.DynamicFunction] = []

        self._type_name_counter = 0
        self._structs_to_fixup = set()
        self._unions_to_fixup = set()

        self._import_types()
        self._fixup_structs()
        self._fixup_unions()
        self._import_functions()
        self._collect_imports()
        self._simplify_types()

    def _import_types(self):
        """Imports initial types from the IDB. The types will be incomplete and need to be fixed"""
        til = self.idb.til
        for type_definition in til.types.defs:
            assert isinstance(type_definition, idb.typeinf.TILTypeInfo)
            type = type_definition.type
            assert isinstance(type, idb.typeinf.TInfo)
            self._convert_idb_type_to_revng_type(type)

    def _import_functions(self):
        api = idb.IDAPython(self.idb)

        metaaddr_type = revng_arch_to_metaaddr_code_type[self.arch]

        for function_start_addr in api.idautils.Functions():
            function = idb.analysis.Function(self.idb, function_start_addr)
            function_name = function.get_name()

            # TODO: python-idb causes an exception when deserializing type info from some DBs, figure out why
            # (it tries to read a byte from an empty buffer)
            try:
                idb_function_type = function.get_signature()
            except Exception as e:
                idb_function_type = None

            if idb_function_type is not None:
                revng_function_type = self._convert_idb_type_to_revng_type(idb_function_type)
            else:
                # TODO: match ABI with arch
                abi = "SystemV_x86_64"
                primitive_kind = m.Primitive(PrimitiveKind=m.PrimitiveTypeKind.Void, Size=0)
                unqualified_return_type = self._get_or_declare(primitive_kind)
                unqualified_return_type_ref = m.Typeref.create(unqualified_return_type)
                qualified_return_type = m.QualifiedType(UnqualifiedType=unqualified_return_type_ref)
                revng_function_type = m.CABIFunctionType(
                    ABI=abi,
                    ReturnType=qualified_return_type,
                    Arguments=[],
                )
                self.revng_types_to_idb_types[revng_function_type] = None
                self.revng_types_by_id[revng_function_type.ID] = revng_function_type

            if api.ida_nalt.is_noret(function_start_addr):
                function_type = m.FunctionType.NoReturn
            else:
                function_type = m.FunctionType.Regular

            revng_function = m.Function(
                CustomName=function_name,
                Entry=m.MetaAddress(Address=function_start_addr, Type=metaaddr_type),
                Type=function_type,
                Prototype=m.Typeref.create(revng_function_type),
            )
            self.functions.add(revng_function)

    def _get_entrypoint(self):
        for entrypoint in idb.analysis.enumerate_entrypoints(self.idb):
            if entrypoint.name == "start" or entrypoint.name == "_start":
                metaaddr_type = revng_arch_to_metaaddr_code_type[self.arch]
                return m.MetaAddress(Address=entrypoint.address, Type=metaaddr_type)
        logger.warning("Could not get entrypoint address")
        return m.MetaAddress(Address=0x0, Type=m.MetaAddressType.Invalid)

    def _get_arch(self):
        api = idb.IDAPython(self.idb)
        inf_structure = api.idaapi.get_inf_structure()
        return idb_procname_to_revng_arch[inf_structure.procname]

    def _is_64_bit(self):
        api = idb.IDAPython(self.idb)
        inf_structure = api.idaapi.get_inf_structure()
        return inf_structure.is_64bit()

    def _collect_imports(self):
        # TODO: we're not getting type information for any imported symbol
        api = idb.IDAPython(self.idb)

        for mod_index in range(api.ida_nalt.get_import_module_qty() + 1):

            def cb(function_addr, function_name, ignored_always_none):
                assert ignored_always_none is None
                function_type = self.name_to_revng_type.get(function_name)
                if function_type is None:
                    logger.warning(f"Type for imported function {function_name} not found")
                    return True

                # TODO: this does not hold, because apparently IDBs contain type declarations for functions and structs
                #  with the same name.
                # assert isinstance(function_type, (t.RawFunctionType, t.CABIFunctionType))
                if not isinstance(function_type, (m.RawFunctionType, m.CABIFunctionType)):
                    logger.warning(
                        f"Type mismatch for {function_name}: revng type for that symbol name is not a function (known issue)"
                    )
                    return True

                function_type_ref = m.Typeref.create(function_type)
                dynamic_function = m.DynamicFunction(SymbolName=function_name, Prototype=function_type_ref)
                self.dynamic_functions.append(dynamic_function)
                return True

            mod_name = api.ida_nalt.get_import_module_name(mod_index)
            api.ida_nalt.enum_import_names(mod_index, cb)

    def _simplify_types(self):
        """Simplifies elements of CABIFunctionType.Argument (which is a list of QualifiedTypes) which have a reduntant
        layer of indirection (QualifiedType -> Typedef -> QualifiedType) to just a QualifiedType.
        This is possible if the Argument is a QualifiedType without any qualifiers.
        """

        # Fixed-point iteration
        keep_going = True
        while keep_going:
            keep_going = False

            for id, revng_type in self.revng_types_by_id.items():
                if not isinstance(revng_type, m.CABIFunctionType):
                    continue

                for argument_type in revng_type.Arguments:
                    qualified_argument_type = argument_type.Type
                    if qualified_argument_type.Qualifiers is not None and len(qualified_argument_type.Qualifiers) > 0:
                        continue

                    unqualified_argument_type = self.resolve_typeref(qualified_argument_type.UnqualifiedType)
                    assert unqualified_argument_type is not None

                    if not isinstance(unqualified_argument_type, m.Typedef):
                        continue

                    typedef_underlying_type = unqualified_argument_type.UnderlyingType

                    if not isinstance(typedef_underlying_type, m.QualifiedType):
                        continue

                    qualified_argument_type.Qualifiers = copy.deepcopy(typedef_underlying_type.Qualifiers)
                    qualified_argument_type.UnqualifiedType = copy.deepcopy(typedef_underlying_type.UnqualifiedType)
                    keep_going = True

    def _fixup_structs(self):
        while self._structs_to_fixup:
            revng_type = self._structs_to_fixup.pop()
            idb_type = self.revng_types_to_idb_types[revng_type]

            assert isinstance(revng_type, m.Struct)
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
                underlying_type = m.Primitive(
                    PrimitiveKind=m.PrimitiveTypeKind.Generic,
                    Size=size,
                )
                registered_underlying_type = self._get_or_declare(underlying_type)
                return m.StructField(
                    CustomName=f"aggregate_{field_number}",
                    Type=m.QualifiedType(
                        UnqualifiedType=m.Typeref.create(registered_underlying_type)
                    ),
                    Offset=offset
                )

            for member in idb_type.type_details.members:
                member_name = self.sanitize_identifier(member.name) or None

                # We're dealing with a bitfield-type member
                if member.type.is_decl_bitfield():
                    # If we're not already aggregating start a new member
                    if aggregate_field_underlying_size == 0:
                        accumulated_bitfield_width = 0
                        aggregate_field_underlying_size = member.type.type_details.nbytes
                        aggregate_fields_count += 1

                    if accumulated_bitfield_width + member.type.type_details.width > aggregate_field_underlying_size * 8:
                        # Emit aggregated member for the previous fields
                        aggregated_member = emit_aggregated_member(
                            aggregate_field_underlying_size,
                            committed_size,
                            aggregate_fields_count
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
                            aggregate_fields_count
                        )
                        fields.append(aggregated_member)
                        committed_size += aggregate_field_underlying_size

                        # Start a new member
                        accumulated_bitfield_width = 0
                        aggregate_field_underlying_size = 0
                        aggregate_fields_count += 1

                    unqualified_underlying_type = self._convert_idb_type_to_revng_type(member.type)
                    unqualified_underlying_type_ref = m.Typeref.create(unqualified_underlying_type)

                    qualifiers = []
                    if member.type.is_decl_const():
                        qualifiers.append(m.Qualifier(Kind=m.QualifierKind.Const))

                    underlying_type = m.QualifiedType(
                        UnqualifiedType=unqualified_underlying_type_ref,
                        Qualifiers=qualifiers
                    )
                    revng_member = m.StructField(
                        CustomName=member_name,
                        Type=underlying_type,
                        Offset=committed_size,
                    )
                    fields.append(revng_member)
                    committed_size += member.type.get_size()

            # Emit final aggregated member
            if accumulated_bitfield_width > 0:
                aggregated_member = emit_aggregated_member(
                    aggregate_field_underlying_size,
                    committed_size,
                    aggregate_fields_count
                )
                fields.append(aggregated_member)
                committed_size += aggregate_field_underlying_size

            revng_type.Fields = fields
            revng_type.Size = committed_size

    def _fixup_unions(self):
        while self._unions_to_fixup:
            revng_type = self._unions_to_fixup.pop()
            idb_type = self.revng_types_to_idb_types[revng_type]

            assert isinstance(revng_type, m.Union)
            assert idb_type.is_decl_union()

            fields = []
            for idx, member in enumerate(idb_type.type_details.members):
                member_name = self.sanitize_identifier(member.name) or None
                unqualified_underlying_type = self._convert_idb_type_to_revng_type(member.type)
                unqualified_underlying_type_ref = m.Typeref.create(unqualified_underlying_type)
                qualified_type = m.QualifiedType(UnqualifiedType=unqualified_underlying_type_ref)
                revng_member = m.UnionField(
                    CustomName=member_name,
                    Type=qualified_type,
                    Index=idx,
                )
                fields.append(revng_member)

            revng_type.Fields = fields

    def _convert_idb_type_to_revng_type(self, type: idb.typeinf.TInfo):
        assert isinstance(type, idb.typeinf.TInfo)

        if type in self.idb_types_to_revng_types:
            return self.idb_types_to_revng_types[type]

        type_name = self.sanitize_identifier(type.get_name()) or self.get_unique_type_name()

        # Some IDBs contain a type named X and then also a typedef X X
        existing_type = self.name_to_revng_type.get(type_name)
        if existing_type is not None:
            if not type.is_decl_typedef():
                logger.warning(f"{type_name} is defined twice in the IDB")
            return existing_type

        # Special case reserved typedefs
        match = re.fullmatch(r"(?P<unsigned>u)?int(?P<width>(8|16|32|64))_t", type_name)
        if match:
            unsigned = bool(match["unsigned"])
            size = int(match["width"]) // 8
            primitive_kind = m.PrimitiveTypeKind.Unsigned if unsigned else m.PrimitiveTypeKind.Signed
            revng_type = m.Primitive(PrimitiveKind=primitive_kind, Size=size)

        elif type.is_decl_typedef():
            aliased_type = type.get_next_tinfo()
            unqualified_aliased_type = self._convert_idb_type_to_revng_type(aliased_type)

            unqualified_aliased_type_ref = m.Typeref.create(unqualified_aliased_type)
            qualified_type = m.QualifiedType(UnqualifiedType=unqualified_aliased_type_ref)
            revng_type = m.Typedef(CustomName=type_name, UnderlyingType=qualified_type)

        elif type.is_decl_enum():
            revng_underlying_type = m.Primitive(
                PrimitiveKind=m.PrimitiveTypeKind.Unsigned,
                Size=type.type_details.storage_size,
            )
            underlying_type_ref = m.Typeref.create(revng_underlying_type)

            entries = []
            for member in type.type_details.members:
                # TODO: we should keep the user comment which might exist in member.cmt
                enum_entry = m.EnumEntry(
                    Aliases=[],
                    CustomName=member.name,
                    Value=member.value,
                )
                entries.append(enum_entry)

            revng_type = m.Enum(
                CustomName=type_name,
                Entries=entries,
                UnderlyingType=underlying_type_ref,
            )

        elif type.is_decl_struct():
            # Struct members and size will be computed later
            revng_type = m.Struct(CustomName=type_name, Size=0, Fields=[])
            self._structs_to_fixup.add(revng_type)

        elif type.is_decl_union():
            # Union members will be computed later
            revng_type = m.Union(CustomName=type_name, Fields=[])
            self._unions_to_fixup.add(revng_type)

        elif type.is_decl_ptr():
            unqualified_underlying_type = self._convert_idb_type_to_revng_type(type.type_details.obj_type)
            qualifiers = [m.Qualifier(Kind=m.QualifierKind.Pointer, Size=type.get_size())]
            unqualified_underlying_type_ref = m.Typeref.create(unqualified_underlying_type)
            underlying_type = m.QualifiedType(UnqualifiedType=unqualified_underlying_type_ref, Qualifiers=qualifiers)
            revng_type = m.Typedef(CustomName=type_name, UnderlyingType=underlying_type)

        elif type.is_decl_array():
            n_elements = type.type_details.n_elems
            unqualified_elem_type = self._convert_idb_type_to_revng_type(type.type_details.elem_type)
            qualifiers = [m.Qualifier(Kind=m.QualifierKind.Array, Size=n_elements)]
            unqualified_elem_type_ref = m.Typeref.create(unqualified_elem_type)
            underlying_type = m.QualifiedType(UnqualifiedType=unqualified_elem_type_ref, Qualifiers=qualifiers)
            revng_type = m.Typedef(CustomName=type_name, UnderlyingType=underlying_type)

        elif type.is_decl_bool():
            size = type.get_size()
            primitive_kind = get_primitive_kind(type)
            revng_type = m.Primitive(PrimitiveKind=primitive_kind, Size=size)

        elif type.is_decl_int() or type.is_decl_floating():
            size = type.get_size()
            primitive_kind = get_primitive_kind(type)
            revng_type = m.Primitive(PrimitiveKind=primitive_kind, Size=size)

        elif type.is_decl_void():
            primitive_kind = get_primitive_kind(type)
            size = 0
            if type.get_name() != "":
                # Treat this case as `typedef void someothername`
                revng_void_type = self._get_or_declare(m.Primitive(PrimitiveKind=primitive_kind, Size=size))
                revng_void_type_ref = m.Typeref.create(revng_void_type)
                revng_type = m.Typedef(
                    CustomName=type_name,
                    UnderlyingType=m.QualifiedType(UnqualifiedType=revng_void_type_ref)
                )
            else:
                revng_type = m.Primitive(PrimitiveKind=primitive_kind, Size=size)

        elif type.is_decl_func():
            # TODO: handle non C-ABI functions
            # We cannot handle stack arguments at the moment
            assert type.type_details.stkargs is None

            idb_return_type = type.get_rettype()
            revng_return_type = self._convert_idb_type_to_revng_type(idb_return_type)
            return_type_ref = m.Typeref.create(revng_return_type)

            arguments = []
            for idx, argument in enumerate(type.type_details.args):
                argument_name = self.sanitize_identifier(argument.name) or None
                argument_unqualified_type = self._convert_idb_type_to_revng_type(argument.type)
                argument_unqualified_type_ref = m.Typeref.create(argument_unqualified_type)
                argument_qualified_type = m.QualifiedType(UnqualifiedType=argument_unqualified_type_ref)
                revng_argument = m.Argument(Index=idx, Type=argument_qualified_type, CustomName=argument_name)
                arguments.append(revng_argument)

            # TODO: match arch and ABI in a proper way
            revng_type = m.CABIFunctionType(
                ABI=revng_arch_to_abi[self.arch],
                ReturnType=m.QualifiedType(UnqualifiedType=return_type_ref),
                Arguments=arguments,
                CustomName=type_name
            )

        elif type.is_decl_partial():
            # We don't know what this type is
            logger.warning(f"Unknown type '{type.get_typestr()}'")
            revng_type = m.Primitive(
                PrimitiveKind=m.PrimitiveTypeKind.Invalid,
                Size=type.get_size(),
            )

        else:
            # We don't know what this type is
            logger.error(f"Unsupported type '{type.get_typestr()}'")
            revng_type = m.Primitive(
                PrimitiveKind=m.PrimitiveTypeKind.Invalid,
                Size=0,
            )

        if revng_type.ID in self.revng_types_by_id:
            assert isinstance(revng_type, m.Primitive)
            return self.revng_types_by_id[revng_type.ID]

        self.revng_types_to_idb_types[revng_type] = type
        self.idb_types_to_revng_types[type] = revng_type
        self.name_to_revng_type[type_name] = revng_type
        self.name_to_idb_type[type_name] = type
        self.revng_types_by_id[revng_type.ID] = revng_type

        return revng_type

    def _get_or_declare(self, t):
        if t.ID in self.revng_types_by_id:
            return self.revng_types_by_id[t.ID]

        else:
            self.revng_types_by_id[t.ID] = t
            self.revng_types_to_idb_types[t] = None
            return t

    def get_model(self) -> m.Binary:
        return m.Binary(
            EntryPoint=self.entrypoint,
            Functions=list(self.functions),
            ImportedDynamicFunctions=[],
            Types=list(self.revng_types_by_id.values()),
            Architecture=self.arch,
            Segments=self.segments,
        )

    def get_unique_type_name(self):
        self._type_name_counter += 1
        return f"type_{self._type_name_counter}"

    def sanitize_identifier(self, identifier: str):
        if self.mangle_names:
            return sanitize_identifier(identifier)
        else:
            return identifier

    def resolve_typeref(self, typeref: m.Typeref) -> Optional[RevngTypes]:
        return self.revng_types_by_id.get(typeref.id)


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
