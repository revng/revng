#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "revng/Pipeline/Container.h"
#include "revng/Support/MetaAddress.h"

namespace revng::pipes {

class StringMapContainer : public pipeline::Container<StringMapContainer> {
  std::map<MetaAddress, std::string> Map;
  const pipeline::Kind *TheKind;

public:
  static char ID;

public:
  StringMapContainer(llvm::StringRef Name, const pipeline::Kind &K) :
    pipeline::Container<StringMapContainer>(Name), Map(), TheKind(&K) {}

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

}; // end class StringMapContainer

} // end namespace revng::pipes
