#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Model/ForwardDecls.h"

#include "revng-c/Support/PTMLC.h"
#include "revng-c/Support/TokenDefinitions.h"

// TODO: find a more suitable place for this
inline const char *StructPaddingPrefix = "_padding_at_";

namespace ArtificialTypes {

constexpr const char *const RetStructPrefix = "_artificial_struct_returned_by_";
constexpr const char *const ArrayWrapperPrefix = "_artificial_wrapper_";

constexpr const char *const RetFieldPrefix = "field_";
constexpr const char *const ArrayWrapperFieldName = "the_array";

} // namespace ArtificialTypes

/// Return a string containing the C Type name of \a QT and a
/// (possibly empty) \a InstanceName.
extern tokenDefinition::types::TypeString
getNamedCInstance(const model::QualifiedType &QT,
                  llvm::StringRef InstanceName,
                  const ptml::PTMLCBuilder &B,
                  llvm::ArrayRef<std::string> AllowedActions = {});

extern tokenDefinition::types::TypeString
getNamedCInstance(llvm::StringRef TypeName,
                  const std::vector<model::Qualifier> &Qualifiers,
                  llvm::StringRef InstanceName,
                  const ptml::PTMLCBuilder &B);

inline tokenDefinition::types::TypeString
getTypeName(const model::QualifiedType &QT, const ptml::PTMLCBuilder &B) {
  return getNamedCInstance(QT, "", B);
}

/// Return the name of the array wrapper that wraps \a QT (QT must be
/// an array).
extern tokenDefinition::types::TypeString
getArrayWrapper(const model::QualifiedType &QT, const ptml::PTMLCBuilder &B);

/// Return a string containing the C Type name of the return type of \a
/// FunctionType, and a (possibly empty) \a InstanceName.
/// \note If F returns more than one value, the name of the wrapping struct
/// will be returned.
extern tokenDefinition::types::TypeString
getNamedInstanceOfReturnType(const model::Type &FunctionType,
                             llvm::StringRef InstanceName,
                             const ptml::PTMLCBuilder &B,
                             bool IsDefinition);

inline tokenDefinition::types::TypeString
getReturnTypeName(const model::Type &FunctionType,
                  const ptml::PTMLCBuilder &B,
                  bool IsDefinition) {
  return getNamedInstanceOfReturnType(FunctionType, "", B, IsDefinition);
}

/// Print the function prototype (without any trailing ';') of \a FT
///        using \a FunctionName as the function's name. If the return value
///        or any of the arguments needs a wrapper, print it with the
///        corresponding wrapper type. The definition of such wrappers
///        should have already been printed before this function is called.
extern void printFunctionPrototype(const model::Type &FT,
                                   const model::Function &Function,
                                   llvm::raw_ostream &Header,
                                   ptml::PTMLCBuilder &B,
                                   const model::Binary &Model,
                                   bool SingleLine);
extern void printFunctionPrototype(const model::Type &FT,
                                   const model::DynamicFunction &Function,
                                   llvm::raw_ostream &Header,
                                   ptml::PTMLCBuilder &B,
                                   const model::Binary &Model,
                                   bool SingleLine);
extern void printFunctionTypeDeclaration(const model::Type &FT,
                                         llvm::raw_ostream &Header,
                                         ptml::PTMLCBuilder &B,
                                         const model::Binary &Model);

extern std::string getArgumentLocationReference(llvm::StringRef ArgumentName,
                                                const model::Function &F,
                                                ptml::PTMLCBuilder &B);
extern std::string getVariableLocationDefinition(llvm::StringRef VariableName,
                                                 const model::Function &F,
                                                 ptml::PTMLCBuilder &B);
extern std::string getVariableLocationReference(llvm::StringRef VariableName,
                                                const model::Function &F,
                                                ptml::PTMLCBuilder &B);
