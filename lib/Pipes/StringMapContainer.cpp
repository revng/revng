//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/IR/ModuleSummaryIndexYAML.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Target.h"
#include "revng/Pipes/StringMapContainer.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

using namespace pipeline;

namespace revng::pipes {

char StringMapContainer::ID = 0;

std::unique_ptr<ContainerBase>
StringMapContainer::cloneFiltered(const TargetsList &Targets) const {
  auto Clone = std::make_unique<StringMapContainer>(*this);

  // Returns true if Targets contains a Target that matches the Entry in the Map
  const auto EntryIsInTargets = [&](const auto &Entry) {
    const auto &KeyMetaAddress = Entry.first;
    pipeline::Target EntryTarget{KeyMetaAddress.toString(), *TheKind };
    return Targets.contains(EntryTarget);
  };

  // Drop all the entries in Map that are not in Targets
  std::erase_if(Clone->Map, std::not_fn(EntryIsInTargets));

  return Clone;
}

void StringMapContainer::mergeBackImpl(StringMapContainer &&Other) {
  // Stuff in Other should overwrite what's in this container.
  // We first merge this->Map into Other.Map (which keeps Other's version if
  // present), and then we replace this->Map with the newly merged version of
  // Other.Map.
  Other.Map.merge(std::move(this->Map));
  this->Map = std::move(Other.Map);
}

TargetsList StringMapContainer::enumerate() const {
  TargetsList Result;
  for (const auto &[MetaAddress, Mapped] : Map)
    Result.push_back({ MetaAddress.toString(), *TheKind });
  return Result;
}

bool StringMapContainer::remove(const TargetsList &Targets) {
  bool Changed = false;

  auto End = Map.end();
  for (const Target &T : Targets) {
    revng_assert(T.getPathComponents().size() == 1);
    std::string MetaAddrStr = T.getPathComponents().back().getName();
    auto It = Map.find(MetaAddress::fromString(MetaAddrStr));
    if (It != End) {
      Map.erase(It);
      Changed = true;
    }
  }

  return Changed;
}

} // end namespace revng::pipes

namespace llvm {
namespace yaml {

template<>
struct CustomMappingTraits<std::map<MetaAddress, std::string>> {

  static void
  inputOne(IO &IO, StringRef Key, std::map<MetaAddress, std::string> &M) {
    IO.mapRequired(Key.str().c_str(), M[MetaAddress::fromString(Key)]);
  }

  static void output(IO &IO, std::map<MetaAddress, std::string> &M) {
    for (auto &[MetaAddr, String] : M)
      IO.mapRequired(MetaAddr.toString().c_str(), String);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace revng::pipes {

llvm::Error StringMapContainer::serialize(llvm::raw_ostream &OS) const {
  llvm::yaml::Output YAMLOutput(OS);
  YAMLOutput << const_cast<std::map<MetaAddress, std::string> &>(Map);
  return llvm::Error::success();
}

llvm::Error StringMapContainer::deserialize(const llvm::MemoryBuffer &Buffer) {

  llvm::yaml::Input YAMLInput(Buffer);
  YAMLInput >> Map;

  if (YAMLInput.error()) {
    this->Map.clear();
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   YAMLInput.error().message());
  }

  return llvm::Error::success();
}

} // end namespace revng::pipes
