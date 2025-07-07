#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/Support/MetaAddress.h"

class Model {
private:
  TupleTree<model::Binary> TheModel;

public:
  std::set<revng::pypeline::ModelPath> diff(const Model &Other) const {
    auto Diff = ::diff(*TheModel.get(), *Other.TheModel.get());
    std::set<revng::pypeline::ModelPath> Result;
    for (auto &Entry : Diff.Changes) {
      auto MaybePath = pathAsString<model::Binary>(Entry.Path);
      Result.insert(MaybePath.value());
    }
    return Result;
  }

  Model clone() const { return *this; }

  std::set<ObjectID> children(const ObjectID &Obj, Kind Kind) const {
    if (Obj.kind() == Kinds::Binary and Kind == Kinds::Function) {
      std::set<ObjectID> Result;
      for (const model::Function &F : TheModel->Functions())
        Result.insert(ObjectID(F.Entry()));
      return Result;
    }

    if (Obj.kind() == Kinds::Binary and Kind == Kinds::TypeDefinition) {
      std::set<ObjectID> Result;
      for (const UpcastablePointer<model::TypeDefinition> &TD :
           TheModel->TypeDefinitions())
        Result.insert(ObjectID(TD->key()));
      return Result;
    }
    revng_abort();
  }

  revng::pypeline::Buffer serialize() const {
    revng::pypeline::Buffer Out;
    llvm::raw_svector_ostream OS(Out.data());
    TheModel.serialize(OS);
    return Out;
  }

  llvm::Error deserialize(llvm::StringRef Input) {
    auto MaybeModel = TupleTree<model::Binary>::fromString(Input);
    if (not MaybeModel)
      return MaybeModel.takeError();

    TheModel = std::move(*MaybeModel);
    return llvm::Error::success();
  }

public:
  TupleTree<model::Binary> &get() { return TheModel; }
  const TupleTree<model::Binary> &get() const { return TheModel; }
};
