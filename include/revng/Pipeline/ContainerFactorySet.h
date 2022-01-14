#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/ContainerSet.h"

namespace Pipeline {

/// A container factory set is a collection of factories so that a set of empty
/// containers can be spawned on demand.
///
/// We keep one of this for each Runner, so that the containers used by that
/// runners can be created.
class ContainerFactorySet {
private:
  llvm::StringMap<ContainerFactory> Entries;

public:
  using const_iterator = llvm::StringMap<ContainerFactory>::const_iterator;
  using value_type = llvm::StringMap<ContainerFactory>::value_type;

  size_t size() const { return Entries.size(); }
  void registerContainerFactory(llvm::StringRef Name, ContainerFactory Entry) {
    Entries.insert_or_assign(Name, std::move(Entry));
  }

  template<typename Cont>
  void registerDefaultConstructibleFactory(llvm::StringRef Name) {
    registerContainerFactory(Name, [](llvm::StringRef Name) {
      return std::make_unique<Cont>(Name);
    });
  }

  ContainerSet createEmpty() const {
    ContainerSet ToReturn;
    for (const auto &Entry : Entries)
      ToReturn.add(Entry.first(), Entry.second);

    return ToReturn;
  }

  const_iterator begin() const { return Entries.begin(); }

  const_iterator end() const { return Entries.end(); }
};
} // namespace Pipeline
