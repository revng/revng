#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include "revng/Model/Binary.h"
#include "revng/Support/MetaAddress.h"

class Model {
private:
  TupleTree<model::Binary> TheModel;

public:
  static constexpr llvm::StringRef Name = "Model";
  Model() : TheModel() {}

  std::set<detail::ModelPath> diff(const Model &Other) const {
    auto Diff = ::diff(*TheModel.get(), *Other.TheModel.get());
    std::set<detail::ModelPath> Result;
    for (auto &Entry : Diff.Changes) {
      std::optional<std::string>
        MaybePath = pathAsString<model::Binary>(Entry.Path);
      revng_assert(MaybePath.has_value());
      Result.insert(*MaybePath);
    }
    return Result;
  }

  Model clone() const {
    Model Clone = *this;
    return Clone;
  }

  std::set<ObjectID> children(const ObjectID &Obj, ObjectID::Kind Kind) const {
    std::set<ObjectID> Result;
    return Result;
  }

  detail::Buffer serialize() const {
    detail::Buffer Out;
    llvm::raw_svector_ostream OS(Out.get());
    TheModel.serialize(OS);
    return Out;
  }

  bool deserialize(llvm::StringRef Input) {
    auto MaybeModel = TupleTree<model::Binary>::fromString(Input);
    if (not MaybeModel) {
      llvm::consumeError(MaybeModel.takeError());
      return false;
    }
    TheModel = std::move(*MaybeModel);
    return true;
  }

public:
  TupleTree<model::Binary> &get() { return TheModel; }
};
