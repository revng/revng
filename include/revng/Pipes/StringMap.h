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
#include "revng/Pipes/TypeKind.h"
#include "revng/Support/GzipTarFile.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace detail {

struct DataOffset {
  size_t UncompressedSize;
  size_t Start;
  size_t End;
};

template<typename T>
using OffsetMap = std::map<T, DataOffset>;

} // namespace detail

namespace llvm::yaml {

template<>
struct MappingTraits<::detail::DataOffset> {
  static void mapping(IO &IO, ::detail::DataOffset &Value) {
    IO.mapRequired("UncompressedSize", Value.UncompressedSize);
    IO.mapRequired("Start", Value.Start);
    IO.mapRequired("End", Value.End);
  }
};

template<typename T>
  requires HasScalarTraits<T>
struct CustomMappingTraits<::detail::OffsetMap<T>> {
  static void
  inputOne(IO &IO, llvm::StringRef StringKey, ::detail::OffsetMap<T> &Value) {
    T Key = getValueFromYAMLScalar<T>(StringKey);
    IO.mapRequired(StringKey.str().c_str(), Value[Key]);
  }

  static void output(IO &IO, ::detail::OffsetMap<T> &Value) {
    for (auto &[Key, Offset] : Value) {
      std::string StringKey = ::toString(Key);
      IO.mapRequired(StringKey.c_str(), Offset);
    }
  }
};

} // namespace llvm::yaml

namespace revng::pipes {

namespace detail {

template<auto *Rank,
         auto *K,
         const char *TypeName,
         const char *MIMETypeParam,
         const char *ArchiveSuffix>
class GenericStringMap
  : public pipeline::Container<
      GenericStringMap<Rank, K, TypeName, MIMETypeParam, ArchiveSuffix>> {
private:
  using RankType = std::remove_pointer_t<decltype(Rank)>;
  static_assert(pipeline::RankSpecialization<RankType>);
  static_assert(RankType::Depth == 1);
  static_assert(HasScalarTraits<typename RankType::Type>);
  static_assert(std::is_convertible_v<decltype(K), pipeline::Kind *>);

public:
  using KeyType = RankType::Type;
  using ValueType = std::string;
  using MapType = typename std::map<KeyType, ValueType>;
  using Iterator = typename MapType::iterator;
  using ConstIterator = typename MapType::const_iterator;

  inline static const llvm::StringRef MIMEType = MIMETypeParam;
  inline static const char *Name = TypeName;

private:
  using OffsetMap = ::detail::OffsetMap<KeyType>;
  MapType Map;

public:
  inline static char ID = '0';

public:
  GenericStringMap(llvm::StringRef Name) :
    pipeline::Container<GenericStringMap>(Name), Map() {
    revng_assert(&K->rank() == Rank);
  }

  GenericStringMap(const GenericStringMap &) = default;
  GenericStringMap &operator=(const GenericStringMap &) = default;

  GenericStringMap(GenericStringMap &&) = default;
  GenericStringMap &operator=(GenericStringMap &&) = default;

  ~GenericStringMap() override = default;

public:
  void clear() override { Map.clear(); }

  std::unique_ptr<pipeline::ContainerBase>
  cloneFiltered(const pipeline::TargetsList &Targets) const override {
    auto Clone = std::make_unique<GenericStringMap>(*this);

    // Returns true if Targets contains a Target that matches the Entry in the
    // Map
    const auto EntryIsInTargets = [&](const auto &Entry) {
      const auto &Key = Entry.first;
      pipeline::Target EntryTarget{ keyToString(Key), *K };
      return Targets.contains(EntryTarget);
    };

    // Drop all the entries in Map that are not in Targets
    std::erase_if(Clone->Map, std::not_fn(EntryIsInTargets));

    return Clone;
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_check(&Target.getKind() == K);

    std::string KeyString = Target.getPathComponents().back();

    auto It = find(keyFromString(KeyString));
    revng_check(It != end());

    OS << It->second;

    return llvm::Error::success();
  }

  pipeline::TargetsList enumerate() const override {
    pipeline::TargetsList::List Result;
    for (const auto &[Key, Value] : Map)
      Result.push_back({ keyToString(Key), *K });

    return Result;
  }

  bool remove(const pipeline::TargetsList &Targets) override {
    bool Changed = false;

    auto End = Map.end();
    for (const pipeline::Target &T : Targets) {
      revng_assert(&T.getKind() == K);

      std::string KeyString = T.getPathComponents().back();
      auto It = Map.find(keyFromString(KeyString));
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

    if (auto Error = MaybeWritableFile.get()->commit())
      return Error;

    revng::FilePath IndexPath = Path.addExtension("idx");
    auto MaybeWritableIndexFile = IndexPath.getWritableFile();
    if (MaybeWritableIndexFile) {
      llvm::yaml::Output IndexOutput(MaybeWritableIndexFile.get()->os());
      IndexOutput << Offsets;

      if (auto Error = MaybeWritableIndexFile.get()->commit())
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

  static std::vector<revng::FilePath>
  getWrittenFiles(const revng::FilePath &Path) {
    return { Path, Path.addExtension("idx") };
  }

  static std::vector<pipeline::Kind *> possibleKinds() { return { K }; }

protected:
  void mergeBackImpl(GenericStringMap &&Other) override {
    // Stuff in Other should overwrite what's in this container.
    // We first merge this->Map into Other.Map (which keeps Other's version if
    // present), and then we replace this->Map with the newly merged version of
    // Other.Map.
    Other.Map.merge(std::move(this->Map));
    this->Map = std::move(Other.Map);
  }

public:
  /// std::map-like methods

  std::string &operator[](KeyType M) { return Map[M]; };

  std::string &at(KeyType M) { return Map.at(M); };
  const std::string &at(KeyType M) const { return Map.at(M); };

private:
  using IteratedValue = std::pair<const KeyType &, std::string &>;
  inline constexpr static auto mapIt = [](auto &Iterated) -> IteratedValue {
    return { Iterated.first, Iterated.second };
  };

  using IteratedCValue = std::pair<const KeyType &, const std::string &>;
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

  auto insert_or_assign(KeyType Key, const std::string &Value) {
    auto [Iterator, Success] = Map.insert_or_assign(Key, Value);
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };
  auto insert_or_assign(KeyType Key, std::string &&Value) {
    auto [Iterator, Success] = Map.insert_or_assign(Key, std::move(Value));
    return std::pair{ revng::map_iterator(Iterator, mapIt), Success };
  };

  bool contains(KeyType Key) const { return Map.contains(Key); }

  auto find(KeyType Key) {
    return revng::map_iterator(Map.find(Key), this->mapIt);
  }

  auto find(KeyType Key) const {
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
      KeyType Key = keyFromString(Name);
      std::string Data = std::string(Entry.Data.data(), Entry.Data.size());
      Map[Key] = Data;
    }
  }

  OffsetMap serializeWithOffsets(llvm::raw_ostream &OS) const {
    OffsetMap Result;
    revng::GzipTarWriter Writer(OS);
    for (auto &[Key, Data] : Map) {
      std::string Name = keyToString(Key) + ArchiveSuffix;
      OffsetDescriptor Offsets = Writer.append(Name,
                                               { Data.data(), Data.size() });
      Result[Key] = { .UncompressedSize = Data.size(),
                      .Start = Offsets.DataStart,
                      .End = Offsets.PaddingStart - 1 };
    }
    Writer.close();

    return Result;
  }

public:
  static std::string keyToString(const KeyType &Key) { return toString(Key); }

  static KeyType keyFromString(llvm::StringRef StringKey) {
    return getValueFromYAMLScalar<KeyType>(StringKey);
  }
}; // end class GenericStringMap

} // namespace detail

template<kinds::FunctionKind *TheKind,
         const char *TypeName,
         const char *MIMETypeParam,
         const char *ArchiveSuffix>
using FunctionStringMap = detail::GenericStringMap<&ranks::Function,
                                                   TheKind,
                                                   TypeName,
                                                   MIMETypeParam,
                                                   ArchiveSuffix>;

template<kinds::TypeKind *TheKind,
         const char *TypeName,
         const char *MIMETypeParam,
         const char *ArchiveSuffix>
using TypeStringMap = detail::GenericStringMap<&ranks::TypeDefinition,
                                               TheKind,
                                               TypeName,
                                               MIMETypeParam,
                                               ArchiveSuffix>;

} // namespace revng::pipes
