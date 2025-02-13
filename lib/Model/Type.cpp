/// \file Type.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/VerifyHelper.h"

// NOTE: there's a really similar function for computing alignment in
//       `lib/ABI/Definition.cpp`. It's better if two are kept in sync, so
//       when modifying this function, please apply corresponding modifications
//       to its little brother as well.
RecursiveCoroutine<std::optional<uint64_t>>
model::Type::trySize(model::VerifyHelper &VH) const {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(this)) {
    if (auto ElementSize = rc_recur Array->ElementType()->trySize(VH))
      rc_return Array->ElementCount() * ElementSize.value();
    else
      rc_return std::nullopt;

  } else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(this)) {
    rc_return Defined->unwrap().trySize(VH);

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(this)) {
    rc_return Pointer->PointerSize();

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(this)) {
    if (Primitive->PrimitiveKind() != model::PrimitiveKind::Void)
      rc_return Primitive->Size();
    else
      rc_return 0;

  } else {
    revng_abort("Unsupported type kind.");
  }
}

std::optional<uint64_t> model::Type::trySize() const {
  model::VerifyHelper VH;
  return trySize(VH);
}

model::TypeDefinition *model::Type::skipToDefinition() {
  model::DefinedType *Defined = skipToDefinedType();
  return Defined ? &Defined->unwrap() : nullptr;
}

const model::TypeDefinition *model::Type::skipToDefinition() const {
  const model::DefinedType *Defined = skipToDefinedType();
  return Defined ? &Defined->unwrap() : nullptr;
}

model::DefinedType *model::Type::skipToDefinedType() {
  model::Type *Current = this;
  while (Current != nullptr) {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(Current))
      Current = Array->ElementType().get();
    else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Current))
      return Defined;
    else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Current))
      Current = Pointer->PointeeType().get();
    else if (llvm::isa<model::PrimitiveType>(Current))
      return nullptr;
    else
      revng_abort("Unsupported type kind.");
  }

  revng_abort("Found an empty type where there's not supposed to be one.");
}

const model::DefinedType *model::Type::skipToDefinedType() const {
  const model::Type *Current = this;
  while (Current != nullptr) {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(Current))
      Current = Array->ElementType().get();
    else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Current))
      return Defined;
    else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Current))
      Current = Pointer->PointeeType().get();
    else if (llvm::isa<model::PrimitiveType>(Current))
      return nullptr;
    else
      revng_abort("Unsupported type kind.");
  }

  revng_abort("Found an empty type where there's not supposed to be one.");
}

std::strong_ordering
model::Type::operator<=>(const model::Type &Another) const {
  if (Kind() != Another.Kind())
    return Kind() <=> Another.Kind();

  // This is pretty nasty, but I don't feel like it's worth to try and hack
  // TTG even further to make this work automagically.
  // TODO: define a less "manual" way of providing the comparisons.

  if (auto *LHS = llvm::dyn_cast<model::ArrayType>(this)) {
    auto &RHS = llvm::cast<model::ArrayType>(Another);

    if (LHS->ElementCount() != RHS.ElementCount())
      return LHS->ElementCount() <=> RHS.ElementCount();
    else if (LHS->ElementType() != RHS.ElementType())
      return LHS->ElementType() <=> RHS.ElementType();

  } else if (auto *LHS = llvm::dyn_cast<model::DefinedType>(this)) {
    auto &RHS = llvm::cast<model::DefinedType>(Another);

    if (LHS->unwrap().key() != RHS.unwrap().key())
      return LHS->unwrap().key() <=> RHS.unwrap().key();

  } else if (auto *LHS = llvm::dyn_cast<model::PointerType>(this)) {
    auto &RHS = llvm::cast<model::PointerType>(Another);

    if (LHS->PointerSize() != RHS.PointerSize())
      return LHS->PointerSize() <=> RHS.PointerSize();
    else if (LHS->PointeeType() != RHS.PointeeType())
      return LHS->PointeeType() <=> RHS.PointeeType();

  } else if (auto *LHS = llvm::dyn_cast<model::PrimitiveType>(this)) {
    auto &RHS = llvm::cast<model::PrimitiveType>(Another);

    if (LHS->PrimitiveKind() != RHS.PrimitiveKind())
      return LHS->PrimitiveKind() <=> RHS.PrimitiveKind();
    else if (LHS->Size() != RHS.Size())
      return LHS->Size() <=> RHS.Size();

  } else {
    revng_abort("Unsupported type kind.");
  }

  return IsConst() <=> Another.IsConst();
}

static RecursiveCoroutine<model::UpcastableType>
getNonConstImpl(const model::Type &T) {
  if (const auto *A = llvm::dyn_cast<model::ArrayType>(&T)) {
    revng_assert(!A->IsConst());
    model::UpcastableType Copy = T;
    Copy->toArray().ElementType() = rc_recur getNonConstImpl(*A->ElementType());
    rc_return Copy;

  } else if (const auto *Def = llvm::dyn_cast<model::DefinedType>(&T)) {
    if (auto *TD = llvm::dyn_cast<model::TypedefDefinition>(&Def->unwrap())) {
      if (TD->UnderlyingType()->isConst())
        rc_return rc_recur getNonConstImpl(*TD->UnderlyingType());
      else
        rc_return T;
    } else {
      // Non-typedef definitions are never const, so no need to go any deeper.
    }

  } else if (const auto *Pointer = llvm::dyn_cast<model::PointerType>(&T)) {
    // Do not propagate further if there's a pointer in the way.

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&T)) {
    // There's no-where to go after we met a primitive.
  }

  // Ensure the very last leaf we looked at is not const.
  if (T.IsConst()) {
    model::UpcastableType Copy = T;
    Copy->IsConst() = false;
    rc_return Copy;
  } else {
    rc_return T;
  }
}

model::UpcastableType model::getNonConst(const model::Type &Type) {
  return getNonConstImpl(Type);
}
