#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

namespace detail {

using TDK = model::TypeDefinition::Key;
using SFK = model::StructField::Key;
using UFK = model::UnionField::Key;
using CAK = model::Argument::Key;
using NRK = model::NamedTypedRegister::Key;

} // namespace detail

static_assert(HasScalarOrEnumTraits<MetaAddress>);
static_assert(HasScalarOrEnumTraits<BasicBlockID>);

/// This is the root rank, as such it has no key (there can only be one).
///
/// Example:
/// ```
/// /binary
/// ```
inline auto Binary = pipeline::defineRootRank<"binary">();

using pipeline::defineRank;

/// This is the rank for *isolated* functions.
///
/// The key is the model key of the function (entry point address).
///
/// Example:
/// ```
/// /function/0x1000:Code_aarch64
/// ```
inline auto
  Function = defineRank<"function", model::Function::Key, "Functions">(Binary);

/// This is the rank for a basic block of an *isolated* function.
///
/// The key is the basic block ID (its entry address + a deduplication index).
///
/// Examples:
/// ```
/// /basic-block/0x1000:Code_aarch64/0x1004:Code_aarch64
/// /basic-block/0x1000:Code_aarch64/0x1004:Code_aarch64-1
/// /basic-block/0x1000:Code_aarch64/0x1004:Code_aarch64-2
/// ```
inline auto BasicBlock = defineRank<"basic-block", BasicBlockID>(Function);

/// This is the rank for an instruction within an *isolated* function.
///
/// The key is the address of the instruction.
///
/// Note that the same instruction is considered two distinct instructions
/// when it's a part of separate (potentially inlined) basic blocks.
///
/// Examples:
/// ```
/// /instruction/0x1000:Code_aarch64/0x1004:Code_aarch64/0x1008:Code_aarch64
/// /instruction/0x1000:Code_aarch64/0x1004:Code_aarch64-1/0x1008:Code_aarch64
/// /instruction/0x1000:Code_aarch64/0x1004:Code_aarch64-2/0x1008:Code_aarch64
/// ```
inline auto Instruction = defineRank<"instruction", MetaAddress>(BasicBlock);

/// This is the rank for a type definition.
///
/// The key is the model key of the definition (its unique numeric ID and kind).
///
/// Examples:
/// ```
/// /type-definition/0-StructDefinition
/// /type-definition/1-UnionDefinition
/// /type-definition/2-EnumDefinition
/// ```
inline auto TypeDefinition = defineRank<"type-definition",
                                        detail::TDK,
                                        "TypeDefinitions">(Binary);

/// This is the rank for a struct field.
///
/// The key is the model key of the definition (offset from the beginning of
/// the struct).
///
/// Examples:
/// ```
/// /struct-field/3-StructDefinition/4
/// /struct-field/4-StructDefinition/16
/// ```
inline auto StructField = defineRank<"struct-field",
                                     detail::SFK,
                                     "Fields">(TypeDefinition);

/// This is the rank for a union field.
///
/// The key is the model key of the definition (numeric index within the union).
///
/// Examples:
/// ```
/// /union-field/5-UnionDefinition/0
/// /union-field/6-UnionDefinition/1
/// ```
inline auto UnionField = defineRank<"union-field", // formatting
                                    detail::UFK,
                                    "Fields">(TypeDefinition);

/// This is the rank for an enumeration entry.
///
/// The key is the model key of the definition (its numeric value).
///
/// Examples:
/// ```
/// /enum-entry/7-EnumDefinition/1
/// /enum-entry/8-EnumDefinition/42
/// /enum-entry/9-EnumDefinition/18446744073709551615
/// ```
inline auto EnumEntry = defineRank<"enum-entry",
                                   model::EnumEntry::Key,
                                   "Entries">(TypeDefinition);

/// This is the rank for a *C ABI* function argument.
///
/// The key is the model key of the definition (its index in the prototype).
///
/// Examples:
/// ```
/// /cabi-argument/10-CABIFunctionDefinition/0
/// /cabi-argument/11-CABIFunctionDefinition/1
/// /cabi-argument/12-CABIFunctionDefinition/2
/// ```
inline auto CABIArgument = defineRank<"cabi-argument",
                                      detail::CAK,
                                      "Arguments">(TypeDefinition);

/// This is the rank for a *raw* function argument.
///
/// The key is the model key of the definition (register used by this argument).
///
/// Examples:
/// ```
/// /raw-argument/13-RawFunctionDefinition/rdi_x86_64
/// /raw-argument/14-RawFunctionDefinition/r3_aarch64
/// /raw-argument/15-RawFunctionDefinition/r3_s390x
/// ```
inline auto RawArgument = defineRank<"raw-argument",
                                     detail::NRK,
                                     "Arguments">(TypeDefinition);

/// This is the rank for the return value of a *C ABI* function.
///
/// The key is the model key of the prototype.
///
/// Note that this is *not* a subrank of the prototype.
///
/// Examples:
/// ```
/// /return-register/18-RawFunctionDefinition/rax_x86_64
/// /return-register/19-RawFunctionDefinition/r2_aarch64
/// /return-register/20-RawFunctionDefinition/r2_s390x
/// ```
inline auto ReturnRegister = defineRank<"return-register",
                                        detail::NRK,
                                        "ReturnValues">(TypeDefinition);

/// This is the rank for *all of* stack arguments of a *raw* function.
///
/// The key is the model key of the prototype.
///
/// Note that this is *not* a subrank of the prototype.
///
/// Example:
/// ```
/// /raw-stack-arguments/21-RawFunctionDefinition
/// ```
inline auto RawStackArguments = defineRank<"raw-stack-arguments",
                                           detail::TDK,
                                           "StackArgumentsType">(Binary);

/// This is the rank for the return value of a *C ABI* function.
///
/// The key is the model key of the prototype.
///
/// Note that this is *not* a subrank of the prototype.
///
/// Example:
/// ```
/// /return-value/22-CABIFunctionDefinition
/// ```
inline auto ReturnValue = defineRank<"return-value", detail::TDK>(Binary);

/// This is the rank for representing an arbitrary byte within the binary.
///
/// The key is the address of the byte.
///
/// Example:
/// ```
/// /raw-byte/0x1000:Code_aarch64
/// ```
inline auto RawByte = defineRank<"raw-byte", MetaAddress>(Binary);

/// This is the rank for representing an arbitrary sequence of bytes within
/// the binary.
///
/// The key is the address of the *last* byte in the sequence.
/// The *first* byte is encoded in the parent `RawByte`'s key.
///
/// Example:
/// ```
/// /raw-byte-range/0x1000:Code_aarch64/0x1008:Code_aarch64
/// ```
inline auto RawByteRange = defineRank<"raw-byte-range", MetaAddress>(RawByte);

/// This is the rank for representing a binary segment.
///
/// The key is the model key (its start address + virtual size).
///
/// Example:
/// ```
/// /segment/0x1000:Code_aarch64-256
/// ```
inline auto Segment = defineRank<"segment", // formatting
                                 model::Segment::Key,
                                 "Segments">(Binary);

/// This is the rank for representing an imported *dynamic* function.
///
/// The key is the model key (its name).
///
/// Example:
/// ```
/// /dynamic-function/printf
/// ```
inline auto DynamicFunction = defineRank<"dynamic-function",
                                         model::DynamicFunction::Key,
                                         "ImportedDynamicFunctions">(Binary);

/// This is the rank for representing a primitive type.
///
/// The key is the *serialized* name of the primitive.
///
/// Examples:
/// ```
/// /primitive/void
/// /primitive/int64_t
/// /primitive/number64_t
/// /primitive/pointer_or_number64_t
/// ```
inline auto PrimitiveType = defineRank<"primitive", std::string>(Binary);

/// This is the rank for representing QEMU and LLVM helper functions.
///
/// The key is the name of the helper.
///
/// Examples:
/// ```
/// /helper-function/helper_clz64
/// /helper-function/helper_qsub16
/// /helper-function/aarch64_save_sp
/// /helper-function/qemu_strnlen
/// ```
inline auto HelperFunction = defineRank<"helper-function", std::string>(Binary);

/// This is the rank for representing *struct* return value types of
/// QEMU and LLVM helper functions.
///
/// The key is the name of the helper.
///
/// Examples:
/// ```
/// /helper-struct-type/helper_clz64
/// /helper-struct-type/helper_qsub16
/// /helper-struct-type/aarch64_save_sp
/// /helper-struct-type/qemu_strnlen
/// ```
inline auto HelperStructType = defineRank<"helper-struct-type", // formatting
                                          std::string>(Binary);

/// This is the rank for representing *fields* or struct return value types of
/// QEMU and LLVM helper functions.
///
/// The key is the name of the field.
/// TODO: consider using indexes instead.
///
/// Examples:
/// ```
/// /helper-struct-field/helper_clz64/field_0
/// /helper-struct-field/helper_qsub16/field_1
/// /helper-struct-field/aarch64_save_sp/field_2
/// /helper-struct-field/qemu_strnlen/field_3
/// ```
inline auto HelperStructField = defineRank<"helper-struct-field",
                                           std::string>(HelperStructType);

/// This is the rank for representing *arguments* of imported dynamic functions.
///
/// The key is the name of the argument.
///
/// Examples:
/// ```
/// /dynamic-function-argument/pow/base
/// /dynamic-function-argument/pow/exp
/// /dynamic-function-argument/printf/format
/// ```
inline auto DynamicFunctionArgument = defineRank<"dynamic-function-argument",
                                                 std::string>(DynamicFunction);

/// This is the rank for representing *local* variables.
///
/// The key is the index of the variable within its function
/// (based on the emission order)
///
/// Examples:
/// ```
/// /local-variable/0x1000:Code_aarch64/0
/// /local-variable/0x1024:Code_aarch64/1
/// /local-variable/0x1144:Code_aarch64/2
/// ```
inline auto LocalVariable = defineRank<"local-variable", uint64_t>(Function);

/// This is the rank for representing *local* variables with *reserved* names.
///
/// The key is the name of the variable.
///
/// Examples:
/// ```
/// /reserved-local-variable/0x1000:Code_aarch64/stack
/// /reserved-local-variable/0x1024:Code_aarch64/stack
/// ```
inline auto ReservedLocalVariable = defineRank<"reserved-local-variable",
                                               std::string>(Function);

/// This is the rank for representing goto labels.
///
/// The key is the index of the label within its function
/// (based on the emission order)
///
/// Examples:
/// ```
/// /goto-label/0x1000:Code_aarch64/0
/// /goto-label/0x1024:Code_aarch64/1
/// /goto-label/0x1144:Code_aarch64/2
/// ```
inline auto GotoLabel = defineRank<"goto-label", uint64_t>(Function);

/// This is the rank for representing comments in isolated function *bodies*.
///
/// The key is the index of the comment within its function
/// (based on the emission order)
///
/// Examples:
/// ```
/// /statement-comment/0x1000:Code_aarch64/0
/// /statement-comment/0x1024:Code_aarch64/1
/// /statement-comment/0x1144:Code_aarch64/2
/// ```
inline auto StatementComment = defineRank<"statement-comment", // formatting
                                          uint64_t>(Function);

/// This is the rank for representing *structs* returned by *raw* function
/// prototypes (they return a set of registers - so these structs are
/// artificially created to represent said registers as fields).
///
/// The key is the model key of the prototype.
///
/// Note that this is *not* a subrank of the prototype.
///
/// Example:
/// ```
/// /artificial-struct/23-RawFunctionDefinition
/// ```
inline auto ArtificialStruct = defineRank<"artificial-struct",
                                          model::TypeDefinition::Key>(Binary);

} // namespace revng::ranks
