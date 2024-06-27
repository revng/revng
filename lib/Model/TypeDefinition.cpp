/// \file TypeDefinition.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

// NOTE: there's a really similar function for computing alignment in
//       `lib/ABI/Definition.cpp`. It's better if two are kept in sync, so
//       when modifying this function, please apply corresponding modifications
//       to its little brother as well.
RecursiveCoroutine<std::optional<uint64_t>>
model::TypeDefinition::trySize(VerifyHelper &VH) const {
  if (auto MaybeSize = VH.size(*this))
    rc_return MaybeSize;

  std::optional<uint64_t> Result = 0;
  if (llvm::isa<model::CABIFunctionDefinition>(this)
      || llvm::isa<model::RawFunctionDefinition>(this)) {
    rc_return 0;

  } else if (auto *E = llvm::dyn_cast<model::EnumDefinition>(this)) {
    Result = rc_recur E->UnderlyingType()->trySize(VH);

  } else if (auto *T = llvm::dyn_cast<model::TypedefDefinition>(this)) {
    Result = rc_recur T->UnderlyingType()->trySize(VH);

  } else if (auto *S = llvm::dyn_cast<model::StructDefinition>(this)) {
    Result = S->Size();

  } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(this)) {
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

  rc_return VH.setSize(*this, Result.value());
}

std::optional<uint64_t> model::TypeDefinition::trySize() const {
  model::VerifyHelper VH;
  return trySize(VH);
}

using DefR = model::DefinitionReference;
template DefR DefR::fromString<model::Binary>(model::Binary *Root,
                                              llvm::StringRef Path);
template DefR DefR::fromString<const model::Binary>(const model::Binary *Root,
                                                    llvm::StringRef Path);

llvm::SmallVector<const model::Type *, 4> model::TypeDefinition::edges() const {
  const auto *PtrCopy = this;
  auto GetEdges = [](const auto &Upcasted) { return Upcasted.edges(); };
  return upcast(PtrCopy, GetEdges, llvm::SmallVector<const model::Type *, 4>());
}
llvm::SmallVector<model::Type *, 4> model::TypeDefinition::edges() {
  auto *PtrCopy = this;
  auto GetEdges = [](auto &Upcasted) { return Upcasted.edges(); };
  return upcast(PtrCopy, GetEdges, llvm::SmallVector<model::Type *, 4>());
}

model::Identifier model::TypeDefinition::name() const {
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    auto Prefix = model::TypeDefinitionKind::automaticNamePrefix(Kind());
    return Identifier((llvm::Twine("_") + Prefix + llvm::Twine(ID())).str());
  }
}

model::Identifier
model::EnumDefinition::entryName(const model::EnumEntry &Entry) const {
  revng_assert(Entries().count(Entry.Value()) != 0);

  if (Entry.CustomName().size() > 0) {
    return Entry.CustomName();
  } else {
    return Identifier((llvm::Twine("_enum_entry_") + name().str() + "_"
                       + llvm::Twine(Entry.Value()))
                        .str());
  }
}

model::Identifier model::UnionField::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (llvm::Twine("_member") + llvm::Twine(Index())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

model::Identifier model::StructField::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (llvm::Twine("_offset_") + llvm::Twine(Offset())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

model::Identifier model::Argument::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (llvm::Twine("_argument") + llvm::Twine(Index())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

model::Identifier model::NamedTypedRegister::name() const {
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    return Identifier((llvm::Twine("_register_") + getRegisterName(Location()))
                        .str());
  }
}
