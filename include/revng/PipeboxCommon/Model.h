#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/Support/MetaAddress.h"

class ModelDiff {
private:
  TupleTreeDiff<model::Binary> Diff;

public:
  ModelDiff() = default;
  ModelDiff(TupleTreeDiff<model::Binary> Diff) : Diff(Diff) {}

  std::set<revng::pypeline::ModelPath> paths() const {
    std::set<revng::pypeline::ModelPath> Result;
    for (auto &Entry : Diff.Changes) {
      auto MaybePath = pathAsString<model::Binary>(Entry.Path);
      Result.insert(MaybePath.value());
    }
    return Result;
  }

  size_t size() const { return Diff.Changes.size(); }

  llvm::SmallVector<char, 0> serialize() const {
    llvm::SmallVector<char, 0> Out;
    llvm::raw_svector_ostream OS(Out);
    ::serialize(OS, Diff);
    return Out;
  }

public:
  TupleTreeDiff<model::Binary> &get() { return Diff; }
  const TupleTreeDiff<model::Binary> &get() const { return Diff; }
};

class Model {
private:
  TupleTree<model::Binary> TheModel;

public:
  ModelDiff diff(const Model &Other) const {
    return ModelDiff(::diff(*TheModel.get(), *Other.TheModel.get()));
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

  llvm::SmallVector<char, 0> serialize() const {
    llvm::SmallVector<char, 0> Out;
    llvm::raw_svector_ostream OS(Out);
    TheModel.serialize(OS);
    return Out;
  }

  static llvm::Expected<Model> deserialize(llvm::ArrayRef<uint8_t> Input) {
    llvm::StringRef String{ reinterpret_cast<const char *>(Input.data()),
                            Input.size() };
    auto MaybeModel = TupleTree<model::Binary>::fromString(String);
    if (not MaybeModel)
      return MaybeModel.takeError();

    Model Result;
    Result.TheModel = std::move(*MaybeModel);
    return Result;
  }

  bool operator==(const Model &Other) const {
    if (this == &Other)
      return true;
    return *this->get().get() == *Other.get().get();
  }

  void enableCaching() { TheModel.enableReferenceCaching(); }
  void disableCaching() { TheModel.disableReferenceCaching(); }

public:
  TupleTree<model::Binary> &get() { return TheModel; }
  const TupleTree<model::Binary> &get() const { return TheModel; }
};
