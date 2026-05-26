#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/OperatingSystem.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

namespace revng {
class RootEntry;
}

inline Logger LDDTreeLog("lddtree");

template<typename T>
concept IsObjectFile = std::derived_from<T, llvm::object::ObjectFile>;

class LDDTree {
public:
  using SymbolSet = std::set<std::string>;

  struct Symbol {
    MetaAddress Address = MetaAddress::invalid();
    bool IsIfunc = false;
  };

  using SymbolMap = std::map<std::string, Symbol>;

  class DependencyFile;
  class Dependency;

public:
  const revng::RootEntry *Root = nullptr;
  std::map<std::string, Dependency> Dependencies;
  std::optional<LDDTree::SymbolSet> MissingSymbols;

public:
  template<typename T>
  void findMissingTypes(const ImporterOptions &Options);

public:
  /// \p MissingSymbols if specified, the function will try to identify the root
  ///    that resolves the most.
  ///    If not specified, the function will try to identify all the libraries,
  ///    recursively.
  template<IsObjectFile T>
  static const std::optional<LDDTree>
  fromPath(llvm::StringRef CanonicalPath,
           const T &ObjectFile,
           std::optional<LDDTree::SymbolSet> MissingSymbols) {
    revng_log(LDDTreeLog, "Getting dependencies of " << CanonicalPath);
    LoggerIndent Indent(LDDTreeLog);

    bool SymbolsRequested = MissingSymbols.has_value();

    LDDTree Result;
    for (const auto &[Name, Root] : revng::configuration().Root) {
      revng_log(LDDTreeLog, "Inspecting root " << Root.Path);
      LoggerIndent Indent(LDDTreeLog);

      auto IsCompatible = [](model::OperatingSystem::Values OperatingSystem) {
        using namespace llvm::object;
        using std::derived_from;
        if constexpr (derived_from<T, ELFObjectFileBase>) {
          return OperatingSystem == model::OperatingSystem::Linux;
        } else if constexpr (derived_from<T, MachOObjectFile>) {
          return OperatingSystem == model::OperatingSystem::MacOS;
        } else if constexpr (derived_from<T, COFFObjectFile>) {
          return OperatingSystem == model::OperatingSystem::Windows;
        } else {
          revng_abort();
        }
      };

      if (not IsCompatible(Root.OperatingSystem)) {
        revng_log(LDDTreeLog,
                  "Skipping incompatible rootfs: incompatible binary format");
        continue;
      }

      auto
        Architecture = model::Architecture::fromLLVMArchitecture(ObjectFile
                                                                   .getArch());
      if (Root.Architecture != Architecture) {
        revng_log(LDDTreeLog,
                  "Skipping incompatible rootfs: incompatible architecture");
        continue;
      }

      auto NewResult = LDDTree::fromPath(Root,
                                         CanonicalPath,
                                         ObjectFile,
                                         MissingSymbols);

      if (SymbolsRequested and NewResult.MissingSymbols.has_value()) {
        auto ResolvedSymbolsCount = (MissingSymbols->size()
                                     - NewResult.MissingSymbols->size());
        revng_log(LDDTreeLog,
                  "Resolved " << ResolvedSymbolsCount << " out of "
                              << MissingSymbols->size() << " symbols");

        if (NewResult.MissingSymbols->size() > 0) {
          if (LDDTreeLog.isEnabled()) {
            LDDTreeLog << "The following symbols have not been resolved:\n";
            for (const std::string &MissingSymbol : *NewResult.MissingSymbols) {
              LDDTreeLog << "  " << MissingSymbol << "\n";
            }
            LDDTreeLog << DoLog;
          }
        }

        auto UnresolvedSymbols = NewResult.MissingSymbols->size();
        if (UnresolvedSymbols == 0) {
          Result = std::move(NewResult);
          // Stop the search, we found all the symbols!
          break;
        } else if (UnresolvedSymbols < Result.MissingSymbols->size()) {
          revng_assert(NewResult.MissingSymbols.has_value());
          Result = std::move(NewResult);
        }
      } else {
        if (Result.Dependencies.size() < NewResult.Dependencies.size()) {
          Result = std::move(NewResult);
        }
      }

      // It was the first attempt
      if (Result.Root == nullptr)
        Result = std::move(NewResult);
    }

    if (Result.Root == nullptr) {
      revng_log(LDDTreeLog, "No matching rootfs found");
      return std::nullopt;
    } else {
      return Result;
    }
  }

private:
  template<IsObjectFile T>
  static const LDDTree
  fromPath(const revng::RootEntry &Root,
           llvm::StringRef CanonicalPath,
           const T &ObjectFile,
           std::optional<LDDTree::SymbolSet> MissingSymbols);

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Stream) const;
};

class LDDTree::DependencyFile {
private:
  /// The loaded binary.
  llvm::object::OwningBinary<llvm::object::Binary> Binary;

  /// The path where the file is. Use this only to pass it to external tools.
  /// Use ObjectFile otherwise.
  std::string FullPathForExternalTools;

  /// The path where the file is supposed to be. Don't open this.
  std::string CanonicalPath;

public:
  DependencyFile(llvm::object::OwningBinary<llvm::object::Binary> &&Binary,
                 std::string FullPathForExternalTools,
                 std::string CanonicalPath) :
    Binary(std::move(Binary)),
    FullPathForExternalTools(std::move(FullPathForExternalTools)),
    CanonicalPath(std::move(CanonicalPath)) {}

  DependencyFile(const DependencyFile &) = delete;
  DependencyFile(DependencyFile &&) = default;
  DependencyFile &operator=(const DependencyFile &) = delete;
  DependencyFile &operator=(DependencyFile &&) = default;

public:
  llvm::object::ObjectFile &objectFile() {
    return *llvm::cast<llvm::object::ObjectFile>(Binary.getBinary());
  }

  const auto &fullPathForExternalTools() const {
    return FullPathForExternalTools;
  }

  const auto &canonicalPath() const { return CanonicalPath; }

public:
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Stream, const char *Prefix) const {
    Stream << Prefix << "ObjectFile: ";
    if (auto *Binary = this->Binary.getBinary()) {
      Stream << Binary->getFileName().str();
    } else {
      Stream << "unavailable";
    }
    Stream << "\n";

    Stream << Prefix << "CanonicalPath: " << CanonicalPath << "\n";
  }
};

class LDDTree::Dependency : public DependencyFile {
private:
  /// Subset of the requested symbols that are provided by this library.
  SymbolMap ProvidedSymbols;

  /// Identifier the library requesting this.
  std::string RequestedBy;

public:
  Dependency(DependencyFile &&TheDependencyFile,
             SymbolMap ProvidedSymbols,
             std::string RequestedBy) :
    DependencyFile(std::move(TheDependencyFile)),
    ProvidedSymbols(std::move(ProvidedSymbols)),
    RequestedBy(std::move(RequestedBy)) {}
  Dependency(const Dependency &) = delete;
  Dependency(Dependency &&) = default;
  Dependency &operator=(const Dependency &) = delete;
  Dependency &operator=(Dependency &&) = default;

public:
  const auto &providedSymbols() const { return ProvidedSymbols; }

  const auto &requestedBy() const { return RequestedBy; }

public:
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Stream, const char *Prefix) const {
    DependencyFile::dump(Stream, Prefix);
    Stream << Prefix << "RequestedBy: " << RequestedBy << "\n";

    Stream << Prefix << "Symbols provided:";
    if (ProvidedSymbols.size() == 0) {
      Stream << " none\n";
    } else {
      Stream << "\n";
      for (auto &[Name, Symbol] : ProvidedSymbols)
        Stream << Prefix << "  " << Name << " at " << Symbol.Address.toString()
               << (Symbol.IsIfunc ? " (IFUNC)" : "") << "\n";
    }
  }
};

template<typename T>
inline void
dumpDependenciesOf(T &Stream,
                   unsigned Depth,
                   const std::map<std::string, LDDTree::Dependency> &Map,
                   llvm::StringRef Name) {
  std::string Prefix;
  for (unsigned I = 0; I < Depth; ++I)
    Prefix += "  ";
  for (auto &[DependencyName, Dependency] : Map) {
    if (Dependency.requestedBy() != Name)
      continue;

    Stream << Prefix << DependencyName << ":\n";
    Dependency.dump(Stream, (Prefix + "  ").c_str());

    Stream << Prefix << "  Dependencies:\n";
    dumpDependenciesOf(Stream, Depth + 2, Map, DependencyName);
  }
}

template<typename T>
inline void LDDTree::dump(T &Stream) const {
  Stream << "Dependencies:\n";
  dumpDependenciesOf(Stream, 1, Dependencies, "");

  if (MissingSymbols.has_value()) {
    Stream << "Missing symbols:\n";
    for (auto Name : *MissingSymbols)
      Stream << "  " << Name << "\n";
  }
}
