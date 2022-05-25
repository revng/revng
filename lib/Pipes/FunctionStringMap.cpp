/// \file FunctionStringMap.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/IR/ModuleSummaryIndexYAML.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

using namespace pipeline;

namespace revng::pipes {

char FunctionStringMap::ID = 0;

std::unique_ptr<ContainerBase>
FunctionStringMap::cloneFiltered(const TargetsList &Targets) const {
  auto Clone = std::make_unique<FunctionStringMap>(*this);

  // Returns true if Targets contains a Target that matches the Entry in the Map
  const auto EntryIsInTargets = [&](const auto &Entry) {
    const auto &KeyMetaAddress = Entry.first;
    pipeline::Target EntryTarget{ KeyMetaAddress.toString(), *TheKind };
    return Targets.contains(EntryTarget);
  };

  // Drop all the entries in Map that are not in Targets
  std::erase_if(Clone->Map, std::not_fn(EntryIsInTargets));

  return Clone;
}

llvm::Error
FunctionStringMap::extractOne(llvm::raw_ostream &OS,
                              const pipeline::Target &Target) const {
  revng_check(Target.getPathComponents().back().isSingle());
  revng_check(&Target.getKind() == TheKind);

  std::string MetaAddrStr = Target.getPathComponents().back().getName();

  auto It = Map.find(MetaAddress::fromString(MetaAddrStr));
  revng_check(It != Map.end());

  OS << It->second;

  return llvm::Error::success();
}

void FunctionStringMap::mergeBackImpl(FunctionStringMap &&Other) {
  // Stuff in Other should overwrite what's in this container.
  // We first merge this->Map into Other.Map (which keeps Other's version if
  // present), and then we replace this->Map with the newly merged version of
  // Other.Map.
  Other.Map.merge(std::move(this->Map));
  this->Map = std::move(Other.Map);
}

TargetsList FunctionStringMap::enumerate() const {
  TargetsList::List Result;
  for (const auto &[MetaAddress, Mapped] : Map)
    Result.push_back({ MetaAddress.toString(), *TheKind });

  return compactFunctionTargets(*Model, Result, *TheKind);
}

bool FunctionStringMap::remove(const TargetsList &Targets) {
  bool Changed = false;

  auto End = Map.end();
  for (const Target &T : Targets) {
    revng_assert(&T.getKind() == TheKind);

    // if a target to remove is *, drop everything
    if (T.getPathComponents().back().isAll()) {
      clear();
      return true;
    }

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

using StringType = revng::pipes::FunctionStringMap::String;

template<>
struct BlockScalarTraits<StringType> {
  static void
  output(const StringType &Value, void *Ctxt, llvm::raw_ostream &OS) {
    OS << Value.TheString;
  }

  static StringRef input(StringRef Scalar, void *Ctxt, StringType &Value) {
    Value.TheString = Scalar.str();
    return StringRef();
  }
};

template<>
struct CustomMappingTraits<std::map<MetaAddress, StringType>> {

  static void
  inputOne(IO &IO, StringRef Key, std::map<MetaAddress, StringType> &M) {
    IO.mapRequired(Key.str().c_str(), M[MetaAddress::fromString(Key)]);
  }

  static void output(IO &IO, std::map<MetaAddress, StringType> &M) {
    for (auto &[MetaAddr, String] : M)
      IO.mapRequired(MetaAddr.toString().c_str(), String);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace revng::pipes {

llvm::Error FunctionStringMap::serialize(llvm::raw_ostream &OS) const {
  ::serialize(OS, Map);
  return llvm::Error::success();
}

llvm::Error FunctionStringMap::deserialize(const llvm::MemoryBuffer &Buffer) {

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
