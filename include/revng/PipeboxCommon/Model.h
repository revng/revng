#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/Support/Error.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

class ModelDiff {
private:
  TupleTreeDiff<model::Binary> Diff;

public:
  ModelDiff() = default;
  ModelDiff(TupleTreeDiff<model::Binary> Diff) : Diff(Diff) {}

  std::set<revng::pypeline::ModelPath> paths() const { return Diff.paths(); }

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
  std::optional<std::string> Path;

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

  std::vector<std::string> aliases(const ObjectID &Object) const {
    model::CNameBuilder NameBuilder(*TheModel.get());
    Kind TheKind = Object.kind();

    std::vector<std::string> Result;
    if (TheKind == Kinds::Function) {

      const MetaAddress &Entry = std::get<MetaAddress>(Object.key());
      auto It = TheModel->Functions().find(Entry);
      if (It != TheModel->Functions().end()) {
        if (not It->Name().empty())
          Result.push_back(It->Name());
        Result.push_back(NameBuilder.automaticName(*It));
      }

    } else if (TheKind == Kinds::TypeDefinition) {

      const auto &Key = std::get<model::TypeDefinition::Key>(Object.key());
      auto It = TheModel->TypeDefinitions().find(Key);
      if (It != TheModel->TypeDefinitions().end()) {
        if (not(*It)->Name().empty())
          Result.push_back((*It)->Name());
        Result.push_back(NameBuilder.automaticName(**It));
      }
    }

    return Result;
  }

  std::optional<ObjectID> resolveAlias(Kind TheKind, std::string Alias) const {
    // A full object location, e.g. "/function/0x40:Code_x86_64".
    std::optional<ObjectID>
      Result = llvm::expectedToOptional(ObjectID::deserialize(Alias));

    // Otherwise, interpret Alias as a bare object key within the requested
    // kind, e.g. "0x40:Code_x86_64" or "42-StructDefinition".
    if (not Result.has_value()) {
      if (TheKind == Kinds::Binary) {
        if (Alias.empty())
          Result = ObjectID();
      } else if (TheKind == Kinds::Function) {
        MetaAddress Entry = MetaAddress::fromString(Alias);
        if (Entry.isValid())
          Result = ObjectID(Entry);
      } else if (TheKind == Kinds::TypeDefinition) {
        Result = ObjectID(getValueFromYAMLScalar<
                          model::TypeDefinition::Key>(Alias));
      } else {
        revng_abort();
      }
    }

    // Keep the candidate only if it is of the requested kind and names an
    // object that exists in the model. Invalid keys (e.g. from deserializing a
    // name-like location) are rejected before the keyed lookup, which would
    // otherwise assert.
    if (Result.has_value()) {
      if (Result->kind() != TheKind) {
        Result = std::nullopt;
      } else if (TheKind == Kinds::Function) {
        const MetaAddress &Entry = std::get<MetaAddress>(Result->key());
        if (not Entry.isValid() or not TheModel->Functions().contains(Entry))
          Result = std::nullopt;
      } else if (TheKind == Kinds::TypeDefinition) {
        const auto &Key = std::get<model::TypeDefinition::Key>(Result->key());
        if (std::get<1>(Key) == model::TypeDefinitionKind::Values::Invalid
            or not TheModel->TypeDefinitions().contains(Key))
          Result = std::nullopt;
      }
      // Binary: the root is the only object of its kind and always exists.
    }

    if (Result.has_value())
      return Result;

    // Fall back to matching by user name or automatic name. An empty string is
    // never a valid name.
    // TODO: build a name cache if this becomes a bottleneck.
    if (Alias.empty())
      return std::nullopt;

    model::CNameBuilder NameBuilder(*TheModel.get());
    if (TheKind == Kinds::Function) {
      for (const model::Function &Function : TheModel->Functions())
        if (Function.Name() == Alias
            or NameBuilder.automaticName(Function) == Alias)
          return ObjectID(Function.Entry());
    } else if (TheKind == Kinds::TypeDefinition) {
      for (const UpcastablePointer<model::TypeDefinition> &Definition :
           TheModel->TypeDefinitions())
        if (Definition->Name() == Alias
            or NameBuilder.automaticName(*Definition) == Alias)
          return ObjectID(Definition->key());
    }
    return std::nullopt;
  }

  llvm::SmallVector<char, 0> serialize() const {
    llvm::SmallVector<char, 0> Out;
    llvm::raw_svector_ostream OS(Out);
    TheModel.serialize(OS);
    return Out;
  }

  static llvm::Expected<Model> deserialize(llvm::ArrayRef<uint8_t> Input,
                                           std::optional<std::string> Path) {
    llvm::StringRef String{ reinterpret_cast<const char *>(Input.data()),
                            Input.size() };
    auto MaybeModel = TupleTree<model::Binary>::fromString(String);
    if (not MaybeModel)
      return MaybeModel.takeError();

    if (not MaybeModel->verify())
      return revng::createError("model failed to verify");

    Model Result;
    Result.TheModel = std::move(*MaybeModel);
    Result.Path = Path;
    return Result;
  }

  bool operator==(const Model &Other) const {
    if (this == &Other)
      return true;
    return *this->get().get() == *Other.get().get();
  }

  void enableCaching() { TheModel.enableReferenceCaching(); }
  void disableCaching() { TheModel.disableReferenceCaching(); }

  std::optional<std::string> path() { return Path; }

public:
  TupleTree<model::Binary> &get() { return TheModel; }
  const TupleTree<model::Binary> &get() const { return TheModel; }
};
