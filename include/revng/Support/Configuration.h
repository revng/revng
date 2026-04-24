#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/RadixTree.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/OperatingSystem.h"
#include "revng/Support/Debug.h"

namespace revng {

class RootEntry;

class DebugInfoServerURLs {
public:
  // Note: remember to update the merge method if you change anything
  std::vector<std::string> Dwarf;
  std::vector<std::string> PE;

public:
  llvm::Error merge(DebugInfoServerURLs &&Other);

public:
  void finalize();
};

class OperatingSystemRoot {
public:
  enum class Kind {
    Windows,
    Linux,
    MacOS
  };

protected:
  OperatingSystemRoot::Kind Kind;
  const revng::RootEntry &Parent;

protected:
  OperatingSystemRoot(enum OperatingSystemRoot::Kind Kind,
                      const revng::RootEntry &Parent) :
    Kind(Kind), Parent(Parent) {}

public:
  auto getKind() const { return Kind; }

public:
  virtual void clear() = 0;
  virtual ~OperatingSystemRoot() {}
};

struct WindowsLibraryMap {
  llvm::RadixTree<std::string, std::string> Map;

  /// v2 and v4 strip the first 4 bytes (`api-`)
  unsigned PrefixLength = 0;
};

class WindowsRoot : public OperatingSystemRoot {
private:
  std::optional<WindowsLibraryMap> Redirections;
  std::optional<std::string> SystemDirectory;

public:
  WindowsRoot(const revng::RootEntry &Parent) :
    OperatingSystemRoot(OperatingSystemRoot::Kind::Windows, Parent) {}
  ~WindowsRoot() override {}

public:
  void clear() override { Redirections.reset(); }

public:
  static bool classof(const OperatingSystemRoot *S) {
    return S->getKind() == OperatingSystemRoot::Kind::Windows;
  }

public:
  llvm::SmallVector<llvm::StringRef, 2>
  getRedirectionsFor(const std::string &LibraryName);

  std::string getSystemDirectory();

private:
  void loadRedirections();
};

class LinuxRootFS : public OperatingSystemRoot {
private:
  std::optional<llvm::SmallVector<std::string>> LdSoConfPaths;

public:
  LinuxRootFS(const revng::RootEntry &Parent) :
    OperatingSystemRoot(OperatingSystemRoot::Kind::Linux, Parent) {}

public:
  static bool classof(const OperatingSystemRoot *S) {
    return S->getKind() == OperatingSystemRoot::Kind::Linux;
  }

public:
  void addLdSoConfPaths(llvm::SmallVectorImpl<std::string> &Paths);

public:
  void clear() override { LdSoConfPaths.reset(); }
  ~LinuxRootFS() override {}

private:
  void loadLdSoConfPaths();
};

class MacOSRoot : public OperatingSystemRoot {
public:
  MacOSRoot(const revng::RootEntry &Parent) :
    OperatingSystemRoot(OperatingSystemRoot::Kind::MacOS, Parent) {}

public:
  static bool classof(const OperatingSystemRoot *S) {
    return S->getKind() == OperatingSystemRoot::Kind::MacOS;
  }

public:
  void clear() override {}
  ~MacOSRoot() override {}
};

class RootEntry {
private:
  std::unique_ptr<OperatingSystemRoot> OSSpecific;

public:
  // Note: remember to update the merge method if you change anything
  std::string Path;
  model::Architecture::Values Architecture = model::Architecture::Invalid;
  model::OperatingSystem::Values
    OperatingSystem = model::OperatingSystem::Invalid;

public:
  bool isHost() const { return Path == "/"; }

  bool isCaseSensitive() const {
    return OperatingSystem != model::OperatingSystem::Windows;
  }

  /// \return the full path of the existing file that you can open, or nullopt
  ///         if it doesn't exist.
  std::optional<std::string>
  getExistingPath(llvm::StringRef RootRelativePath) const;

  template<typename T>
  T &osSpecific() const {
    return *llvm::cast<T>(OSSpecific.get());
  }

public:
  llvm::Error merge(RootEntry &&Other);

public:
  void finalize();
};

class Configuration {
public:
  using RootsMap = std::map<std::string, revng::RootEntry>;

public:
  // Note: remember to update the merge method if you change anything
  std::vector<std::string> TranslationLDFlags;
  std::string CachePath;
  DebugInfoServerURLs DebugInfoServerURLs;
  RootsMap Root;

public:
  /// Parse the configuration in \p Path, if it exists.
  static llvm::Expected<Configuration> fromPath(const llvm::Twine &Path);
  static llvm::Error reload();

public:
  llvm::Error merge(Configuration &&Other);

public:
  void dump() const debug_function;

public:
  void finalize();
};

const Configuration &configuration();

} // namespace revng
