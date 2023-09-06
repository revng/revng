#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/GzipTarFile.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace detail {

struct DataOffset {
  size_t Start;
  size_t End;
};

using OffsetMap = std::map<MetaAddress, DataOffset>;

} // namespace detail

namespace llvm::yaml {

template<>
struct MappingTraits<::detail::DataOffset> {
  static void mapping(IO &IO, ::detail::DataOffset &Value) {
    IO.mapRequired("Start", Value.Start);
    IO.mapRequired("End", Value.End);
  }
};

template<>
struct CustomMappingTraits<::detail::OffsetMap> {
  static void
  inputOne(IO &IO, llvm::StringRef Key, ::detail::OffsetMap &Value) {
    MetaAddress Address = MetaAddress::fromString(Key);
    IO.mapRequired(Key.str().c_str(), Value[Address]);
  }

  static void output(IO &IO, ::detail::OffsetMap &Value) {
    for (auto &[MetaAddr, Offset] : Value) {
      IO.mapRequired(MetaAddr.toString().c_str(), Offset);
    }
  }
};

} // namespace llvm::yaml

namespace revng::pipes {

template<kinds::FunctionKind *K,
         const char *TypeName,
         const char *MIMETypeParam,
         const char *ArchiveSuffix>
class FunctionStringMap
  : public pipeline::Container<
      FunctionStringMap<K, TypeName, MIMETypeParam, ArchiveSuffix>> {
public:
  using MapType = typename std::map<MetaAddress, std::string>;
  using ValueType = typename MapType::value_type;
  using Iterator = typename MapType::iterator;
  using ConstIterator = typename MapType::const_iterator;

  inline static const llvm::StringRef MIMEType = MIMETypeParam;
  inline static const char *Name = TypeName;

private:
  using OffsetMap = ::detail::OffsetMap;
  MapType Map;
  const TupleTree<model::Binary> *Model;

public:
  inline static char ID = '0';

public:
  FunctionStringMap(llvm::StringRef Name,
                    const TupleTree<model::Binary> *Model) :
    pipeline::Container<FunctionStringMap>(Name), Map(), Model(Model) {
    revng_assert(&K->rank() == &ranks::Function);
  }

  FunctionStringMap(const FunctionStringMap &) = default;
  FunctionStringMap &operator=(const FunctionStringMap &) = default;

  FunctionStringMap(FunctionStringMap &&) = default;
  FunctionStringMap &operator=(FunctionStringMap &&) = default;

  ~FunctionStringMap() override = default;

public:
  void clear() override { Map.clear(); }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override {
    auto Clone = std::make_unique<FunctionStringMap>(*this);

    // Returns true if Targets contains a Target that matches the Entry in the
    // Map
    const auto EntryIsInTargets = [&](const auto &Entry) {
      const auto &KeyMetaAddress = Entry.first;
      pipeline::Target EntryTarget{ KeyMetaAddress.toString(), *K };
      return Targets.contains(EntryTarget);
    };

    // Drop all the entries in Map that are not in Targets
    std::erase_if(Clone->Map, std::not_fn(EntryIsInTargets));

    return Clone;
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_check(&Target.getKind() == K);

    std::string MetaAddrStr = Target.getPathComponents().back();

    auto It = find(MetaAddress::fromString(MetaAddrStr));
    revng_check(It != end());

    OS << It->second;

    return llvm::Error::success();
  }

  pipeline::TargetsList enumerate() const override {
    pipeline::TargetsList::List Result;
    for (const auto &[MetaAddress, Mapped] : Map)
      Result.push_back({ MetaAddress.toString(), *K });

    return Result;
  }

  bool remove(const pipeline::TargetsList &Targets) override {
    bool Changed = false;

    auto End = Map.end();
    for (const pipeline::Target &T : Targets) {
      revng_assert(&T.getKind() == K);

      std::string MetaAddrStr = T.getPathComponents().back();
      auto It = Map.find(MetaAddress::fromString(MetaAddrStr));
      if (It != End) {
        Map.erase(It);
        Changed = true;
      }
    }

    return Changed;
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const override {
    serializeWithOffsets(OS);
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) override {
    GzipTarReader Reader(Buffer);
    deserializeImpl(Reader);
    return llvm::Error::success();
  }

  llvm::Error store(const revng::FilePath &Path) const override {
    auto MaybeWritableFile = Path.getWritableFile(ContentEncoding::Gzip);
    if (not MaybeWritableFile)
      return MaybeWritableFile.takeError();

    OffsetMap Offsets = serializeWithOffsets(MaybeWritableFile.get()->os());

    if (auto Error = MaybeWritableFile.get()->commit(); Error)
      return Error;

    revng::FilePath IndexPath = Path.addExtension("idx");
    auto MaybeWritableIndexFile = IndexPath.getWritableFile();
    if (!!MaybeWritableIndexFile) {
      llvm::yaml::Output IndexOutput(MaybeWritableIndexFile.get()->os());
      IndexOutput << Offsets;

      if (auto Error = MaybeWritableIndexFile.get()->commit(); Error)
        return Error;
    } else {
      llvm::consumeError(MaybeWritableIndexFile.takeError());
    }

    return llvm::Error::success();
  }

  llvm::Error load(const revng::FilePath &Path) override {
    auto MaybeExists = Path.exists();
    if (not MaybeExists)
      return MaybeExists.takeError();

    if (not MaybeExists.get()) {
      clear();
      return llvm::Error::success();
    }

    auto MaybeBuffer = Path.getReadableFile();
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();

    GzipTarReader Reader(MaybeBuffer.get()->buffer());
    deserializeImpl(Reader);
    return llvm::Error::success();
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { K }; }

protected:
  void mergeBackImpl(FunctionStringMap &&Other) override {
    // Stuff in Other should overwrite what's in this container.
    // We first merge this->Map into Other.Map (which keeps Other's version if
    // present), and then we replace this->Map with the newly merged version of
    // Other.Map.
    Other.Map.merge(std::move(this->Map));
    this->Map = std::move(Other.Map);
  }

public:
  /// std::map-like methods

  std::string &operator[](MetaAddress M) { return Map[M]; };

  std::string &at(MetaAddress M) { return Map.at(M); };
  const std::string &at(MetaAddress M) const { return Map.at(M); };

private:
  using IteratedValue = std::pair<const MetaAddress &, std::string &>;
  inline constexpr static auto mapIt = [](auto &Iterated) -> IteratedValue {
    return { Iterated.first, Iterated.second };
  };

  using IteratedCValue = std::pair<const MetaAddress &, const std::string &>;
  inline constexpr static auto mapCIt = [](auto &Iterated) -> IteratedCValue {
    return { Iterated.first, Iterated.second };
  };

public:
  auto insert(const ValueType &V) {
    auto [Iterator, Success] = Map.insert(V);
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };
  auto insert(ValueType &&V) {
    auto [Iterator, Success] = Map.insert(std::move(V));
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };

  auto insert_or_assign(MetaAddress Key, const std::string &Value) {
    auto [Iterator, Success] = Map.insert_or_assign(Key, Value);
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };
  auto insert_or_assign(MetaAddress Key, std::string &&Value) {
    auto [Iterator, Success] = Map.insert_or_assign(Key, std::move(Value));
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };

  bool contains(MetaAddress Key) const { return Map.contains(Key); }

  auto find(MetaAddress Key) {
    return revng::map_iterator(Map.find(Key), this->mapIt);
  }
  auto find(MetaAddress Key) const {
    return revng::map_iterator(Map.find(Key), this->mapCIt);
  }

  auto begin() { return revng::map_iterator(Map.begin(), this->mapIt); }
  auto end() { return revng::map_iterator(Map.end(), this->mapIt); }

  auto begin() const { return revng::map_iterator(Map.begin(), this->mapCIt); }
  auto end() const { return revng::map_iterator(Map.end(), this->mapCIt); }

private:
  void deserializeImpl(GzipTarReader &Reader) {
    for (ArchiveEntry &Entry : Reader.entries()) {
      llvm::StringRef Name = Entry.Filename;
      revng_assert(Name.consume_back(ArchiveSuffix));
      MetaAddress Address = MetaAddress::fromString(Name);
      std::string Data = std::string(Entry.Data.data(), Entry.Data.size());
      Map[Address] = Data;
    }
  }

  OffsetMap serializeWithOffsets(llvm::raw_ostream &OS) const {
    OffsetMap Result;
    revng::GzipTarWriter Writer(OS);
    for (auto &[MetaAddr, Data] : Map) {
      std::string Name = MetaAddr.toString() + ArchiveSuffix;
      OffsetDescriptor Offsets = Writer.append(Name,
                                               { Data.data(), Data.size() });
      Result[MetaAddr] = { .Start = Offsets.DataStart,
                           .End = Offsets.PaddingStart - 1 };
    }
    Writer.close();

    return Result;
  }
}; // end class FunctionStringMap

template<typename ToRegister>
class RegisterFunctionStringMap : public pipeline::Registry {

public:
  virtual ~RegisterFunctionStringMap() override = default;

public:
  void registerContainersAndPipes(pipeline::Loader &Loader) override {
    const auto &Model = getModelFromContext(Loader.getContext());
    auto Factory = pipeline::ContainerFactory::fromGlobal<ToRegister>(&Model);
    Loader.addContainerFactory(ToRegister::Name, std::move(Factory));
  }
  void registerKinds(pipeline::KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};
} // namespace revng::pipes
