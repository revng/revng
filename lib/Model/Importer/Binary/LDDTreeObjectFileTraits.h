#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <queue>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "revng/Support/Configuration.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FileSystem.h"
#include "revng/Support/LDDTree.h"

template<IsObjectFile T>
class LDDTreeObjectFileTraits;

template<typename D>
class Queue {
public:
  struct Entry {
    /// Identifier of the library to lookup
    std::string ToImport;

    /// Identifier of the library importing this library
    std::string ImporterIdentifier;

    /// The canonical path of the importer
    std::string ImporterCanonicalPath;

    /// Image format-specific information for resolve
    D Dependency;
  };

private:
  std::queue<Entry> Queue;

public:
  bool empty() const { return Queue.empty(); }

public:
  void enqueue(std::vector<Entry> &&Entries) {
    for (Entry &NewEntry : Entries)
      Queue.push(std::move(NewEntry));
  }

  Entry pop() {
    Entry Result = Queue.front();
    Queue.pop();
    return Result;
  }
};

template<IsObjectFile T>
using ResolutionInfo = typename LDDTreeObjectFileTraits<T>::ResolutionInfo;

template<IsObjectFile T>
using QueueEntry = typename Queue<ResolutionInfo<T>>::Entry;
template<typename T,
         typename ObjectFile = typename T::ObjectFile,
         typename ResolutionInfo = typename T::ResolutionInfo,
         typename QueueEntry = typename Queue<ResolutionInfo>::Entry>
concept LDDTreeObjectFileTraitsConcept = requires(T Trait,
                                                  ObjectFile Object,
                                                  llvm::StringRef String,
                                                  QueueEntry Entry,
                                                  const revng::RootEntry
                                                    &Root) {
  typename T::ResolutionInfo;

  { T::getExportedSymbols(Object) } -> std::same_as<LDDTree::SymbolMap>;

  {
    T::getDependencies(Root, String, String, Object)
  } -> std::same_as<std::vector<QueueEntry>>;

  {
    T::computeSearchPaths(Root, Entry)
  } -> std::same_as<llvm::SmallVector<std::string, 16>>;

  { T::isValidObjectFor(Entry, Object) } -> std::same_as<bool>;
};

template<IsObjectFile T>
std::optional<LDDTree::DependencyFile>
resolve(const revng::RootEntry &Root, QueueEntry<T> &Entry) {
  using namespace llvm;
  using Traits = LDDTreeObjectFileTraits<T>;

  revng_log(LDDTreeLog, "Looking for " << Entry.ToImport);
  LoggerIndent Indent(LDDTreeLog);

  auto &Dependency = Entry.Dependency;

  // TODO: computeSearchPaths could be a generator
  SmallVector<std::string, 16> SearchPaths = Traits::computeSearchPaths(Root,
                                                                        Entry);

  revng_assert(llvm::none_of(SearchPaths, [](const std::string &Path) {
    return Path.empty();
  }));

  for (const auto &Path : SearchPaths) {
    llvm::SmallString<128> Candidate;
    llvm::sys::path::append(Candidate, Path, Entry.ToImport);
    auto MaybeFullPath = Root.getExistingPath(Candidate);

    if (not MaybeFullPath.has_value())
      continue;

    // Parse the binary
    auto MaybeBinary = object::createBinary(*MaybeFullPath);
    if (auto Error = MaybeBinary.takeError()) {
      revng_log(LDDTreeLog, "Can't create binary: " << Error);
      llvm::consumeError(std::move(Error));
      continue;
    }

    // Ensure it's an object of the right type
    using namespace object;
    auto *Binary = dyn_cast<T>(MaybeBinary->getBinary());
    if (Binary == nullptr) {
      revng_log(LDDTreeLog,
                "Found " << Candidate.str()
                         << " but it's not the right type of object file.");
      continue;
    }

    if (not Traits::isValidObjectFor(Entry, *Binary)) {
      continue;
    }

    revng_log(LDDTreeLog, "Found: " << Candidate.str());

    return LDDTree::DependencyFile(std::move(*MaybeBinary),
                                   std::move(*MaybeFullPath),
                                   Candidate.str().str());
  }

  revng_log(LDDTreeLog, Entry.ToImport << " not found");

  if (LDDTreeLog.isEnabled()) {
    LDDTreeLog << "List of searched paths:\n";
    for (const auto &Path : SearchPaths)
      LDDTreeLog << "  " << Path << "\n";
    LDDTreeLog << DoLog;
  }

  return std::nullopt;
}

/// \p MissingSymbols if specified, the function will try to solve all the
//     symbols.
///    If not specified, the function will try to identify all the libraries,
///    recursively.
template<IsObjectFile T>
const LDDTree LDDTree::fromPath(const revng::RootEntry &Root,
                                llvm::StringRef CanonicalPath,
                                const T &Binary,
                                std::optional<LDDTree::SymbolSet> Symbols) {
  revng_log(LDDTreeLog, "Computing dependencies of " << CanonicalPath);
  LoggerIndent Indent(LDDTreeLog);

  if (Symbols.has_value())
    revng_log(LDDTreeLog,
              "We need to resolve " << Symbols->size() << " symbols");

  using ObjectFileTraits = LDDTreeObjectFileTraits<T>;
  static_assert(LDDTreeObjectFileTraitsConcept<ObjectFileTraits>);

  LDDTree Result(&Root);

  bool SymbolsRequested = Symbols.has_value();
  if (SymbolsRequested)
    Result.MissingSymbols = std::move(*Symbols);

  Queue<typename ObjectFileTraits::ResolutionInfo> Queue;
  std::set<std::string> Visited;

  Queue.enqueue(ObjectFileTraits::getDependencies(Root,
                                                  "",
                                                  CanonicalPath,
                                                  Binary));

  while (not Queue.empty()) {
    auto Entry = Queue.pop();

    revng_log(LDDTreeLog, "Considering dependency " << Entry.ToImport);
    LoggerIndent Indent(LDDTreeLog);

    if (Visited.contains(Entry.ToImport)
        or Result.Dependencies.contains(Entry.ToImport)) {
      revng_log(LDDTreeLog, "Skipping, we already have this library.");
      continue;
    }
    Visited.insert(Entry.ToImport);

    // Resolve the path of the entry in the queue
    auto MaybeDependencyFile = resolve<T>(Root, Entry);
    if (not MaybeDependencyFile.has_value()) {
      revng_log(LDDTreeLog, "Couldn't resolve the dependency.");
      continue;
    }

    revng_log(LDDTreeLog,
              "Found at " << MaybeDependencyFile->fullPathForExternalTools());

    auto *Binary = llvm::dyn_cast<T>(&MaybeDependencyFile->objectFile());
    if (Binary == nullptr) {
      revng_log(LDDTreeLog,
                "We found a binary, but it is not of the expected type");
      continue;
    }

    LDDTree::SymbolMap ProvidedSymbols;

    // Purge from MissingSymbols those exported
    if (SymbolsRequested) {
      for (auto &[SymbolName, SymbolData] :
           ObjectFileTraits::getExportedSymbols(*Binary)) {
        if (Result.MissingSymbols->contains(SymbolName)) {
          Result.MissingSymbols->erase(SymbolName);
          ProvidedSymbols[SymbolName] = SymbolData;
        }
      }
    }

    if (not SymbolsRequested or ProvidedSymbols.size() > 0) {
      revng_log(LDDTreeLog, "Registering the library as a dependency");

      if (SymbolsRequested) {
        if (LDDTreeLog.isEnabled()) {
          LDDTreeLog << "The library provides " << ProvidedSymbols.size()
                     << " symbols:\n";
          for (auto &Symbol : ProvidedSymbols) {
            LDDTreeLog << "  " << Symbol.first << "\n";
          }
          LDDTreeLog << DoLog;
        }
      }

      // We fulfilled at least a symbol, record it as a dependency
      LDDTree::Dependency NewDependency(std::move(*MaybeDependencyFile),
                                        std::move(ProvidedSymbols),
                                        Entry.ImporterIdentifier);
      Result.Dependencies.emplace(Entry.ToImport, std::move(NewDependency));
    } else {
      revng_log(LDDTreeLog, "Ignoring this library");
    }

    // Check if completed the quest. If so, no need to proceed any further!
    if (SymbolsRequested and Result.MissingSymbols->size() == 0) {
      revng_log(LDDTreeLog, "We found all the symbols!");
      return Result;
    }

    // Enqueue new entries, once for each dependency
    revng_log(LDDTreeLog, "Collecting dependencies");
    LoggerIndent Indent2(LDDTreeLog);

    auto DependencyCanonicalPath = MaybeDependencyFile->canonicalPath();
    Queue.enqueue(ObjectFileTraits::getDependencies(Root,
                                                    Entry.ToImport,
                                                    DependencyCanonicalPath,
                                                    *Binary));
  }

  return Result;
}
