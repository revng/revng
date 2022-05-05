#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Support/MetaAddress.h"

namespace revng::pipes {

class StringMapContainer : public pipeline::Container<StringMapContainer> {
public:
  using MapType = std::map<MetaAddress, std::string>;
  using ValueType = MapType::value_type;
  using Iterator = MapType::iterator;
  using ConstIterator = MapType::const_iterator;

private:
  MapType Map;
  const pipeline::Kind *TheKind;

public:
  static char ID;

public:
  StringMapContainer(llvm::StringRef Name,
                     llvm::StringRef MIMEType,
                     const pipeline::Kind &K) :
    pipeline::Container<StringMapContainer>(Name, MIMEType),
    Map(),
    TheKind(&K) {}

  StringMapContainer(const StringMapContainer &) = default;
  StringMapContainer &operator=(const StringMapContainer &) = default;

  StringMapContainer(StringMapContainer &&) = default;
  StringMapContainer &operator=(StringMapContainer &&) = default;

  ~StringMapContainer() override = default;

public:
  void clear() override { Map.clear(); }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override;

  pipeline::TargetsList enumerate() const override;

  bool remove(const pipeline::TargetsList &Targets) override;

  llvm::Error serialize(llvm::raw_ostream &OS) const override;

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override;

protected:
  void mergeBackImpl(StringMapContainer &&Container) override;

public:
  /// std::map-like methods

  std::string &operator[](MetaAddress M) { return Map[M]; };

  std::string &at(MetaAddress M) { return Map.at(M); };
  const std::string &at(MetaAddress M) const { return Map.at(M); };

  std::pair<Iterator, bool> insert(const ValueType &V) {
    return Map.insert(V);
  };
  std::pair<Iterator, bool> insert(ValueType &&V) {
    return Map.insert(std::move(V));
  };

  std::pair<Iterator, bool>
  insert_or_assign(MetaAddress Key, const std::string &Value) {
    return Map.insert_or_assign(Key, Value);
  };
  std::pair<Iterator, bool>
  insert_or_assign(MetaAddress Key, std::string &&Value) {
    return Map.insert_or_assign(Key, std::move(Value));
  };

  bool contains(MetaAddress Key) const { return Map.contains(Key); }

  Iterator find(MetaAddress Key) { return Map.find(Key); }
  ConstIterator find(MetaAddress Key) const { return Map.find(Key); }

  Iterator begin() { return Map.begin(); }
  Iterator end() { return Map.end(); }

  ConstIterator begin() const { return Map.begin(); }
  ConstIterator end() const { return Map.end(); }

}; // end class StringMapContainer

inline pipeline::ContainerFactory
makeStringMapContainerFactory(pipeline::Kind &K, llvm::StringRef MIMEType) {
  return [&K, MIMEType](llvm::StringRef Name) {
    return std::make_unique<StringMapContainer>(Name, MIMEType, K);
  };
}

} // end namespace revng::pipes
