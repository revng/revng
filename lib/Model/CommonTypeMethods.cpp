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
std::optional<uint64_t> Common<CRTP>::trySize() const {
  model::VerifyHelper VH;
  return trySize(VH);
}

template<typename CRTP>
RecursiveCoroutine<std::optional<uint64_t>>
Common<CRTP>::size(model::VerifyHelper &VH) const {
  std::optional<uint64_t> MaybeSize = rc_recur trySize(VH);
  revng_check(MaybeSize.has_value());
  if (MaybeSize.value() == 0)
    rc_return std::nullopt;
  else
    rc_return MaybeSize;
}

// NOTE: there's a really similar function for computing alignment in
//       `lib/ABI/Definition.cpp`. It's better if two are kept in sync, so
//       when modifying this function, please apply corresponding modifications
//       to its little brother as well.
template<typename CRTP>
RecursiveCoroutine<std::optional<uint64_t>>
Common<CRTP>::trySize(VerifyHelper &VH) const {
  // This code assumes that this type is well formed.
  // TODO: handle recursive types

  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    if (auto MaybeSize = VH.size(get()))
      rc_return MaybeSize;

    std::optional<uint64_t> Result = 0;
    if (llvm::isa<model::CABIFunctionDefinition>(&get())
        || llvm::isa<model::RawFunctionDefinition>(&get())) {
      rc_return 0;

    } else if (auto *E = llvm::dyn_cast<model::EnumDefinition>(&get())) {
      Result = rc_recur E->UnderlyingType()->trySize(VH);

    } else if (auto *T = llvm::dyn_cast<model::TypedefDefinition>(&get())) {
      Result = rc_recur T->UnderlyingType()->trySize(VH);

    } else if (auto *S = llvm::dyn_cast<model::StructDefinition>(&get())) {
      Result = S->Size();

    } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(&get())) {
      for (const auto &Field : U->Fields()) {
        if (auto Size = rc_recur Field.Type()->trySize(VH))
          Result = std::max(Result.value(), Size.value());
        else
          rc_return std::nullopt;
      }

    } else {
      revng_abort("Unsupported type definition kind.");
    }

    if (!Result.has_value())
      rc_return std::nullopt;

    rc_return VH.setSize(get(), Result.value());

  } else if constexpr (std::is_same_v<CRTP, model::Type>) {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(&get())) {
      if (auto ElementSize = rc_recur Array->ElementType()->trySize(VH))
        rc_return Array->ElementCount() * ElementSize.value();
      else
        rc_return std::nullopt;

    } else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(&get())) {
      rc_return Defined->asDefinition().trySize(VH);

    } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&get())) {
      rc_return Pointer->PointerSize();

    } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&get())) {
      if (Primitive->PrimitiveKind() != model::PrimitiveKind::Void)
        rc_return Primitive->Size();
      else
        rc_return 0;

    } else {
      revng_abort("Unsupported type kind.");
    }

  } else {
    static_assert(type_always_false_v<CRTP>, "Unsupported 'common' type.");
  }
}

//
// Other Helpers
//

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

template<typename CRTP>
model::TypeDefinition *Common<CRTP>::tryGetAsPrototype() {
  if (model::TypeDefinition *Definition = tryGetAsDefinition()) {
    if (llvm::isa<model::CABIFunctionDefinition>(Definition)
        || llvm::isa<model::RawFunctionDefinition>(Definition)) {
      return Definition;
    }
  }

  return nullptr;
}

template<typename CRTP>
const model::TypeDefinition *Common<CRTP>::tryGetAsPrototype() const {
  if (const model::TypeDefinition *Definition = tryGetAsDefinition()) {
    if (llvm::isa<model::CABIFunctionDefinition>(Definition)
        || llvm::isa<model::RawFunctionDefinition>(Definition)) {
      return Definition;
    }
  }

  return nullptr;
}

template<typename CRTP>
model::StructDefinition *Common<CRTP>::tryGetAsStruct() {
  if (model::TypeDefinition *Definition = tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::StructDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
const model::StructDefinition *Common<CRTP>::tryGetAsStruct() const {
  if (const model::TypeDefinition *Definition = tryGetAsDefinition())
    if (auto *Cast = llvm::dyn_cast<model::StructDefinition>(Definition))
      return Cast;

  return nullptr;
}

template<typename CRTP>
model::PrimitiveType *Common<CRTP>::tryGetAsPrimitive() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Unwrapped))
    return Primitive;

  return nullptr;
}

template<typename CRTP>
const model::PrimitiveType *Common<CRTP>::tryGetAsPrimitive() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Unwrapped))
    return Primitive;

  return nullptr;
}

template<typename CRTP>
model::ArrayType *Common<CRTP>::tryGetAsArray() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Array = llvm::dyn_cast<model::ArrayType>(Unwrapped))
    return Array;

  return nullptr;
}

template<typename CRTP>
const model::ArrayType *Common<CRTP>::tryGetAsArray() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Array = llvm::dyn_cast<model::ArrayType>(Unwrapped))
    return Array;

  return nullptr;
}

template<typename CRTP>
model::PointerType *Common<CRTP>::tryGetAsPointer() {
  model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return nullptr;

  if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Unwrapped))
    return Pointer;

  return nullptr;
}

template<typename CRTP>
const model::PointerType *Common<CRTP>::tryGetAsPointer() const {
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
  if (const model::PrimitiveType *Primitive = tryGetAsPrimitive())
    return Primitive->PrimitiveKind() == Kind;

  return false;
}

template<typename CRTP>
bool Common<CRTP>::isScalar() const {
  const model::Type *Unwrapped = skipConstAndTypedefs();
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>)
    if (Unwrapped == nullptr)
      return false;

  return llvm::isa<model::PrimitiveType>(*Unwrapped)
         || llvm::isa<model::PointerType>(*Unwrapped);
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
model::TypeDefinition *Common<CRTP>::skipToDefinition() {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    return &get();
  } else {
    model::Type *Current = &get();
    while (Current != nullptr) {
      if (auto *Array = llvm::dyn_cast<model::ArrayType>(Current))
        Current = Array->ElementType().get();
      else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Current))
        return &Defined->asDefinition();
      else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Current))
        Current = Pointer->PointeeType().get();
      else if (llvm::isa<model::PrimitiveType>(Current))
        return nullptr;
      else
        revng_abort("Unsupported type kind.");
    }
  }

  revng_abort("There's an empty type where there's not supposed to be one.");
}

template<typename CRTP>
const model::TypeDefinition *Common<CRTP>::skipToDefinition() const {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    return &get();
  } else {
    const model::Type *Current = &get();
    while (Current != nullptr) {
      if (auto *Array = llvm::dyn_cast<model::ArrayType>(Current))
        Current = Array->ElementType().get();
      else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Current))
        return &Defined->asDefinition();
      else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Current))
        Current = Pointer->PointeeType().get();
      else if (llvm::isa<model::PrimitiveType>(Current))
        return nullptr;
      else
        revng_abort("Unsupported type kind.");
    }
  }

  revng_abort("There's an empty type where there's not supposed to be one.");
}

template<typename CRTP>
model::Type &Common<CRTP>::stripPointer() {
  return *asPointer().PointeeType();
}

template<typename CRTP>
const model::Type &Common<CRTP>::stripPointer() const {
  return *asPointer().PointeeType();
}

// NOTE: I'm not making these into coroutines for now since I can't imagine
//       the recursion here to be that deep. We can always turn them if we run
//       into problems.

using TypedefD = model::TypedefDefinition;

template<typename CRTP>
model::Type *Common<CRTP>::skipTypedefs() {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&get()))
      return Typedef->UnderlyingType()->skipTypedefs();

    return nullptr;
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      if (!Def->IsConst())
        if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->asDefinition()))
          return Typedef->UnderlyingType()->skipTypedefs();

    return &get();
  }
}

template<typename CRTP>
const model::Type *Common<CRTP>::skipTypedefs() const {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&get()))
      return Typedef->UnderlyingType()->skipTypedefs();

    return nullptr;
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      if (!Def->IsConst())
        if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->asDefinition()))
          return Typedef->UnderlyingType()->skipTypedefs();

    return &get();
  }
}

template<typename CRTP>
model::Type *Common<CRTP>::skipConstAndTypedefs() {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&get()))
      return Typedef->UnderlyingType()->skipConstAndTypedefs();

    return nullptr;
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->asDefinition()))
        return Typedef->UnderlyingType()->skipConstAndTypedefs();

    return &get();
  }
}

template<typename CRTP>
const model::Type *Common<CRTP>::skipConstAndTypedefs() const {
  if constexpr (std::is_same_v<CRTP, model::TypeDefinition>) {
    if (auto *Typedef = llvm::dyn_cast<TypedefD>(&get()))
      return Typedef->UnderlyingType()->skipConstAndTypedefs();

    return nullptr;
  } else {
    if (auto *Def = llvm::dyn_cast<model::DefinedType>(&get()))
      if (auto *Typedef = llvm::dyn_cast<TypedefD>(&Def->asDefinition()))
        return Typedef->UnderlyingType()->skipConstAndTypedefs();

    return &get();
  }
}

// Force instantiation for the cases we care about.
template class model::CommonTypeMethods<model::TypeDefinition>;
template class model::CommonTypeMethods<model::Type>;
