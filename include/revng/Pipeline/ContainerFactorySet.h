#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/ContainerFactory.h"
#include "revng/Pipeline/ContainerSet.h"

namespace pipeline {

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

public:
  const_iterator begin() const { return Entries.begin(); }
  const_iterator end() const { return Entries.end(); }
  size_t size() const { return Entries.size(); }

  value_type &at(llvm::StringRef Name) {
    auto Search = Entries.find(Name);
    revng_assert(Search != Entries.end());
    return *Search;
  }

  const value_type &at(llvm::StringRef Name) const {
    auto Search = Entries.find(Name);
    revng_assert(Search != Entries.end());
    return *Search;
  }

public:
  void registerContainerFactory(llvm::StringRef Name, ContainerFactory Entry) {
    Entries.insert_or_assign(Name, std::move(Entry));
  }

  template<typename Cont>
  void registerDefaultConstructibleFactory(llvm::StringRef Name) {
    registerContainerFactory(Name, ContainerFactory::create<Cont>());
  }

  ContainerSet createEmpty() const {
    ContainerSet ToReturn;
    for (const auto &Entry : Entries)
      ToReturn.add(Entry.first(), Entry.second);

    return ToReturn;
  }
};

} // namespace pipeline
