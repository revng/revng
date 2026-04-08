//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Triple.h"

#include "revng/Model/Architecture.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/LDDTree.h"

#include "LDDTreeObjectFileTraits.h"

using namespace llvm;
using namespace llvm::object;

template<>
class LDDTreeObjectFileTraits<COFFObjectFile> {
public:
  using ObjectFile = COFFObjectFile;

  struct ResolutionInfo {
    llvm::Triple::ArchType Architecture{};

    bool is64() const {
      return Architecture == llvm::Triple::x86_64
             or Architecture == llvm::Triple::aarch64;
    }
  };

  using QueueEntry = typename Queue<ResolutionInfo>::Entry;

public:
  static std::vector<std::pair<llvm::StringRef, uint64_t>>
  getExportedSymbols(const COFFObjectFile &COFFObjectFile) {
    std::vector<std::pair<llvm::StringRef, uint64_t>> Result;

    for (const ExportDirectoryEntryRef &Entry :
         COFFObjectFile.export_directories()) {

      StringRef SymbolName;
      if (Error E = Entry.getSymbolName(SymbolName)) {
        std::string Message = llvm::toString(std::move(E));
        revng_log(LDDTreeLog,
                  "Couldn't get symbol name for export entry: " << Message);
        continue;
      }

      uint32_t RVA = 0;
      if (Error E = Entry.getExportRVA(RVA)) {
        std::string Message = llvm::toString(std::move(E));
        revng_log(LDDTreeLog, "Couldn't get symbol RVA: " << Message);
        continue;
      }

      Result.push_back({ SymbolName, RVA });
    }

    return Result;
  }

  static std::vector<QueueEntry>
  getDependencies(const revng::RootEntry &Root,
                  StringRef ImporterIdentifier,
                  StringRef ImporterCanonicalPath,
                  const COFFObjectFile &COFFObjectFile) {
    auto &WindowsRoot = Root.osSpecific<revng::WindowsRoot>();

    std::vector<QueueEntry> Results;
    for (const ImportDirectoryEntryRef &I :
         COFFObjectFile.import_directories()) {
      StringRef LibraryName;
      if (Error E = I.getName(LibraryName)) {
        revng_log(LDDTreeLog, "Found an imported symbol without a name.");
        continue;
      }

      auto LibraryNameAsString = LibraryName.str();

      revng_log(LDDTreeLog, "Considering library " << LibraryNameAsString);
      LoggerIndent Indent(LDDTreeLog);

      // Get redirections from apisetschema.dll for these libraries
      auto Redirections = WindowsRoot.getRedirectionsFor(LibraryNameAsString);
      if (not Redirections.empty()) {
        revng_log(LDDTreeLog, "Redirecting to:");
        LoggerIndent Indent(LDDTreeLog);

        for (auto &Redirection : Redirections) {
          revng_log(LDDTreeLog, Redirection.str());
          Results.push_back({ Redirection.str(),
                              ImporterIdentifier.str(),
                              ImporterCanonicalPath.str(),
                              { COFFObjectFile.getArch() } });
        }
      } else {
        Results.push_back({ LibraryNameAsString,
                            ImporterIdentifier.str(),
                            ImporterCanonicalPath.str(),
                            { COFFObjectFile.getArch() } });
      }
    }

    return Results;
  }

  /// \see https://tinyurl.com/4yduz682
  static SmallVector<std::string, 16>
  computeSearchPaths(const revng::RootEntry &Root, QueueEntry &Entry) {
    // Note: despite taking Root, the results should not be prefixed with root!

    auto &WindowsRoot = Root.osSpecific<revng::WindowsRoot>();
    bool InputIs64 = Entry.Dependency.is64();

    using namespace llvm::sys::path;

    SmallVector<std::string, 16> SearchPaths;

    // Register the directory of the binary itself
    SmallString<16> Path;
    append(Path, Entry.ImporterCanonicalPath);
    remove_filename(Path);
    if (not Path.empty())
      SearchPaths.push_back(Path.str().str());

    // Note: in "non-safe" DLL search mode we should look in the *current*
    // directory. We really don't want to do that.

    // Register the "system directory"
    SearchPaths.push_back(WindowsRoot.getSystemDirectory());

    // Note: we should now look in the "16-bit system directory". We really
    // don't want to do that.

    // Register the "Windows directory"
    Path.clear();
    append(Path, StringRef("Windows"));
    SearchPaths.push_back(Path.str().str());

#if defined(__WIN32__) || defined(__MINGW32__) || defined(_WIN32)
    // Process `PATH` environment variable
    if (Root.isHost()) {
      if (auto MaybeLibraryPath = llvm::sys::Process::GetEnv("PATH")) {
        for (StringRef PathFromPath : split(*MaybeLibraryPath, ":")) {
          Path.clear();
          append(Path, PathFromPath);
          SearchPaths.push_back(Path.str().str());
        }
      }
    }
#endif

    return SearchPaths;
  }

  static bool isValidObjectFor(QueueEntry &Entry,
                               const COFFObjectFile &COFFObjectFile) {
    if (COFFObjectFile.getArch() != Entry.Dependency.Architecture) {
      revng_log(LDDTreeLog, "Incompatible architecture, ignoring.");
      return false;
    }

    return true;
  }
};

using Trait = LDDTreeObjectFileTraits<COFFObjectFile>;
static_assert(LDDTreeObjectFileTraitsConcept<Trait>);

template const LDDTree
LDDTree::fromPath<COFFObjectFile>(const revng::RootEntry &Root,
                                  StringRef Path,
                                  const COFFObjectFile &Binary,
                                  std::optional<LDDTree::SymbolSet> Symbols);
