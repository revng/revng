//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

//
// NEW
//

using DefR = model::DefinitionReference;
template DefR DefR::fromString<model::Binary>(model::Binary *Root,
                                              llvm::StringRef Path);
template DefR DefR::fromString<const model::Binary>(const model::Binary *Root,
                                                    llvm::StringRef Path);

const llvm::SmallVector<const model::Type *, 4>
model::TypeDefinition::edges() const {
  const auto *PtrCopy = this;
  auto GetEdges = [](const auto &Upcasted) { return Upcasted.edges(); };
  return upcast(PtrCopy, GetEdges, llvm::SmallVector<const model::Type *, 4>());
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
