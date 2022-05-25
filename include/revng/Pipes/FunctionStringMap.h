#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/MetaAddress.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pipes {

class FunctionStringMap : public pipeline::Container<FunctionStringMap> {
public:
  /// Wrapper for std::string that allows YAML-serialization as multiline string
  struct String {
    std::string TheString;
    operator std::string() const { return TheString; }
  };

public:
  using MapType = std::map<MetaAddress, String>;
  using ValueType = MapType::value_type;
  using Iterator = MapType::iterator;
  using ConstIterator = MapType::const_iterator;

private:
  MapType Map;
  const pipeline::Kind *TheKind;
  const TupleTree<model::Binary> *Model;

public:
  static char ID;

public:
  FunctionStringMap(llvm::StringRef Name,
                    llvm::StringRef MIMEType,
                    const pipeline::Kind &K,
                    const TupleTree<model::Binary> &Model) :
    pipeline::Container<FunctionStringMap>(Name, MIMEType),
    Map(),
    TheKind(&K),
    Model(&Model) {
    revng_assert(&K.rank() == &FunctionsRank);
  }

  FunctionStringMap(const FunctionStringMap &) = default;
  FunctionStringMap &operator=(const FunctionStringMap &) = default;

  FunctionStringMap(FunctionStringMap &&) = default;
  FunctionStringMap &operator=(FunctionStringMap &&) = default;

  ~FunctionStringMap() override = default;

public:
  void clear() override { Map.clear(); }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override;

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override;

  pipeline::TargetsList enumerate() const override;

  bool remove(const pipeline::TargetsList &Targets) override;

  llvm::Error serialize(llvm::raw_ostream &OS) const override;

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override;

protected:
  void mergeBackImpl(FunctionStringMap &&Container) override;

public:
  /// std::map-like methods

  std::string &operator[](MetaAddress M) { return Map[M].TheString; };

  std::string &at(MetaAddress M) { return Map.at(M).TheString; };
  const std::string &at(MetaAddress M) const { return Map.at(M).TheString; };

  std::pair<Iterator, bool> insert(const ValueType &V) {
    return Map.insert(V);
  };
  std::pair<Iterator, bool> insert(ValueType &&V) {
    return Map.insert(std::move(V));
  };

  std::pair<Iterator, bool>
  insert_or_assign(MetaAddress Key, const std::string &Value) {
    return Map.insert_or_assign(Key, String{ Value });
  };
  std::pair<Iterator, bool>
  insert_or_assign(MetaAddress Key, std::string &&Value) {
    return Map.insert_or_assign(Key, String{ std::move(Value) });
  };

  bool contains(MetaAddress Key) const { return Map.contains(Key); }

  Iterator find(MetaAddress Key) { return Map.find(Key); }
  ConstIterator find(MetaAddress Key) const { return Map.find(Key); }

  Iterator begin() { return Map.begin(); }
  Iterator end() { return Map.end(); }

  ConstIterator begin() const { return Map.begin(); }
  ConstIterator end() const { return Map.end(); }

}; // end class FunctionStringMap

class RegisterFunctionStringMap : public pipeline::Registry {
private:
  llvm::StringRef Name;
  llvm::StringRef MIMEType;
  const pipeline::Kind &K;

public:
  RegisterFunctionStringMap(llvm::StringRef Name,
                            llvm::StringRef MIMEType,
                            const pipeline::Kind &K) :
    Name(Name), MIMEType(MIMEType), K(K) {}

public:
  virtual ~RegisterFunctionStringMap() override = default;

public:
  void registerContainersAndPipes(pipeline::Loader &Loader) override {
    const auto &Model = getModelFromContext(Loader.getContext());
    auto Factory = [&Model, this](llvm::StringRef ContainerName) {
      return std::make_unique<FunctionStringMap>(ContainerName,
                                                 MIMEType,
                                                 K,
                                                 Model);
    };
    Loader.addContainerFactory(Name, Factory);
  }
  void registerKinds(pipeline::KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // end namespace revng::pipes
