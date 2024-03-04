/// \file CommonTypeMethods.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/VerifyHelper.h"

template<typename CRTP>
using Common = model::CommonTypeMethods<CRTP>;

//
// Size Helpers
//

template<typename CRTP>
std::optional<uint64_t> Common<CRTP>::size() const {
  model::VerifyHelper VH;
  return size(VH);
}

template<typename CRTP>
RecursiveCoroutine<std::optional<uint64_t>>
Common<CRTP>::size(model::VerifyHelper &VH) const {
  std::optional<uint64_t> MaybeSize = rc_recur get().trySize(VH);
  revng_check(MaybeSize.has_value());
  if (MaybeSize.value() == 0)
    rc_return std::nullopt;
  else
    rc_return MaybeSize;
}

//
// Other Helpers
//

template<typename CRTP>
model::TypeDefinition *Common<CRTP>::getPrototype() {
  model::Type *U = skipTypedefs();
  if (U && U->isConst())
    return nullptr;

  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition()) {
    if (llvm::isa<model::CABIFunctionDefinition>(Definition)
        || llvm::isa<model::RawFunctionDefinition>(Definition)) {
      return Definition;
    }
  }

  return nullptr;
}

template<typename CRTP>
const model::TypeDefinition *Common<CRTP>::getPrototype() const {
  const model::Type *U = skipTypedefs();
  if (U && U->isConst())
    return nullptr;

  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition()) {
    if (llvm::isa<model::CABIFunctionDefinition>(Definition)
        || llvm::isa<model::RawFunctionDefinition>(Definition)) {
      return Definition;
    }
  }

  return nullptr;
}

template<typename CRTP>
model::StructDefinition *Common<CRTP>::getStruct() {
  model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::StructDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
const model::StructDefinition *Common<CRTP>::getStruct() const {
  const model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::StructDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
model::UnionDefinition *Common<CRTP>::getUnion() {
  model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::UnionDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
const model::UnionDefinition *Common<CRTP>::getUnion() const {
  const model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::UnionDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
model::EnumDefinition *Common<CRTP>::getEnum() {
  model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::EnumDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
const model::EnumDefinition *Common<CRTP>::getEnum() const {
  const model::Type *U = skipConstAndTypedefs();
  if (auto *Definition = U ? U->tryGetAsDefinition() : tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::EnumDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
model::PrimitiveType *Common<CRTP>::getPrimitive() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Unwrapped))
    return Primitive;

  return nullptr;
}

template<typename CRTP>
const model::PrimitiveType *Common<CRTP>::getPrimitive() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Unwrapped))
    return Primitive;

  return nullptr;
}

template<typename CRTP>
model::ArrayType *Common<CRTP>::getArray() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Array = llvm::dyn_cast<model::ArrayType>(Unwrapped))
    return Array;

  return nullptr;
}

template<typename CRTP>
const model::ArrayType *Common<CRTP>::getArray() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Array = llvm::dyn_cast<model::ArrayType>(Unwrapped))
    return Array;

  return nullptr;
}

template<typename CRTP>
model::PointerType *Common<CRTP>::getPointer() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Unwrapped))
    return Pointer;

  return nullptr;
}

template<typename CRTP>
const model::PointerType *Common<CRTP>::getPointer() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Unwrapped))
    return Pointer;

  return nullptr;
}

template<typename CRTP>
bool Common<CRTP>::isPrimitive(model::PrimitiveKind::Values Kind) const {
  if (const model::PrimitiveType *Primitive = getPrimitive())
    return Primitive->PrimitiveKind() == Kind;

  return false;
}

template<typename CRTP>
bool Common<CRTP>::isScalar() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::same_as<CRTP, model::TypeDefinition>) {
    if (Unwrapped == nullptr) {
      revng_assert(get().isObject());
      return llvm::isa<model::EnumDefinition>(get());
    }
  }

  revng_assert(Unwrapped->isObject());
  if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Unwrapped)) {
    return true;
  } else if (auto *D = llvm::dyn_cast<model::DefinedType>(Unwrapped)) {
    return llvm::isa<model::EnumDefinition>(D->unwrap());
  } else {
    return llvm::isa<model::PointerType>(*Unwrapped);
  }
}

template<typename CRTP>
bool Common<CRTP>::isConst() const {
  if (auto *Unwrapped = skipTypedefs())
    return Unwrapped->IsConst();

  // `skipTypedefs` can only fail if it's called on a non-typedef
  // `TypeDefinition` and those can never be `const`.
  return false;
}

template<typename CRTP>
model::TypeDefinition *Common<CRTP>::tryGetAsDefinition() {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    return &get();
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      return Def->Definition().get();

    return nullptr;
  }
}

template<typename CRTP>
const model::TypeDefinition *Common<CRTP>::tryGetAsDefinition() const {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    return &get();
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      return Def->Definition().getConst();

    return nullptr;
  }
}

using TypedefD = model::TypedefDefinition;

template<typename T>
static RecursiveCoroutine<ConstPtrIfConst<T, model::Type>>
skipTypedefsImpl(T &&V) {
  if constexpr (std::is_same_v<std::decay_t<T>, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&V))
      rc_return rc_recur skipTypedefsImpl(*Typedef->UnderlyingType());

    rc_return nullptr;

  } else if constexpr (std::is_same_v<std::decay_t<T>, model::Type>) {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&V))
      if (!Def->IsConst())
        if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->unwrap()))
          rc_return rc_recur skipTypedefsImpl(*Typedef->UnderlyingType());

    rc_return &V;

  } else {
    static_assert(type_always_false_v<T>);
  }
}

template<typename CRTP>
model::Type *Common<CRTP>::skipTypedefs() {
  return skipTypedefsImpl(get());
}

template<typename CRTP>
const model::Type *Common<CRTP>::skipTypedefs() const {
  return skipTypedefsImpl(get());
}

template<typename T>
static RecursiveCoroutine<ConstPtrIfConst<T, model::Type>>
skipConstAndTypedefsImpl(T &&V) {
  if constexpr (std::is_same_v<std::decay_t<T>, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&V))
      rc_return rc_recur skipConstAndTypedefsImpl(*Typedef->UnderlyingType());

    rc_return nullptr;

  } else if constexpr (std::is_same_v<std::decay_t<T>, model::Type>) {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&V))
      if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->unwrap()))
        rc_return rc_recur skipConstAndTypedefsImpl(*Typedef->UnderlyingType());

    rc_return &V;

  } else {
    static_assert(type_always_false_v<T>);
  }
}

template<typename CRTP>
model::Type *Common<CRTP>::skipConstAndTypedefs() {
  return skipConstAndTypedefsImpl(get());
}

template<typename CRTP>
const model::Type *Common<CRTP>::skipConstAndTypedefs() const {
  return skipConstAndTypedefsImpl(get());
}

// Force instantiation for the cases we care about
template class model::CommonTypeMethods<model::Type>;
template class model::CommonTypeMethods<model::TypeDefinition>;
