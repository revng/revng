#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace model {
class QualifiedType;
class RawFunctionType;
class CABIFunctionType;
class Type;
class Binary;
} // namespace model

namespace ArtificialTypes {
constexpr const char *const RetStructPrefix = "artificial_struct_";
constexpr const char *const ArrayWrapperPrefix = "artificial_wrapper_";
constexpr const char *const FunctionTypedefPrefix = "function_type_";

constexpr const char *const RetFieldPrefix = "field_";
constexpr const char *const ArrayWrapperFieldName = "the_array";
} // namespace ArtificialTypes

using TypeString = llvm::SmallString<32>;

/// Return an escaped name for the type
/// \note If T is a function type, the appropriate function typename will be
/// returned
extern TypeString getTypeName(const model::Type &T);

/// Print a string containing the C Type name of \a QT and a
/// (possibly empty) \a InstanceName .
extern TypeString
getNamedCInstance(const model::QualifiedType &QT, llvm::StringRef InstanceName);

/// Return the name of the array wrapper that wraps \a QT (QT must be
/// an array).
extern TypeString getArrayWrapper(const model::QualifiedType &QT);

/// Return the name of the type returned by \a F
/// \note If F returns more than one value, the name of the wrapping struct
/// will be returned.
extern TypeString getReturnTypeName(const model::RawFunctionType &F);

/// Return the name of the array wrapper that wraps \a QT (QT must be
/// an array).
/// \note If F returns an array, the name of the wrapping struct will be
/// returned.
extern TypeString getReturnTypeName(const model::CABIFunctionType &F);

/// Return the name of the \a Index -th field of the struct returned
/// by \a F.
/// \note F must be returning more than one value, otherwise
/// there is no wrapping struct.
extern TypeString getReturnField(const model::RawFunctionType &F, size_t Index);

/// Print the function prototype (without any trailing ';') of \a FT
///        using \a FunctionName as the function's name. If the return value
///        or any of the arguments needs a wrapper, print it with the
///        corresponding wrapper type. The definition of such wrappers
///        should have already been printed before this function is called.
extern void printFunctionPrototype(const model::Type &FT,
                                   llvm::StringRef FunctionName,
                                   llvm::raw_ostream &Header,
                                   const model::Binary &Model);
