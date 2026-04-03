//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

extern "C" {
#include "glob.h"
}

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/OperatingSystem.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/Generator.h"
#include "revng/Support/PathList.h"
#include "revng/Support/WindowsApiSetSchemaParser.h"
#include "revng/Support/YAMLTraits.h"

using namespace llvm;
using yaml::IO;
using yaml::MappingTraits;

static Logger Log("configuration");

// Keep the configuration static, so we can give it away as const
static revng::Configuration TheConfiguration;
static bool ConfigurationLoaded = false;

const revng::Configuration &revng::configuration() {
  revng_assert(ConfigurationLoaded);
  return TheConfiguration;
}

template<>
struct MappingTraits<revng::DebugInfoServerURLs> {
  static void mapping(IO &IO, revng::DebugInfoServerURLs &Object) {
    IO.mapOptional("dwarf", Object.Dwarf);
    IO.mapOptional("pe", Object.PE);
  }
};

template<>
struct MappingTraits<revng::RootEntry> {
  static void mapping(IO &IO, revng::RootEntry &Object) {
    IO.mapOptional("path", Object.Path);
    IO.mapOptional("architecture", Object.Architecture);
    IO.mapOptional("operating-system", Object.OperatingSystem);
  }
};

template<>
struct MappingTraits<revng::Configuration> {
  static void mapping(IO &IO, revng::Configuration &Object) {
    IO.mapOptional("translation-ldflags", Object.TranslationLDFlags);
    IO.mapOptional("cache-path", Object.CachePath);
    IO.mapOptional("debug-info-server-urls", Object.DebugInfoServerURLs);
    IO.mapOptional("rootfs", Object.Root);
  }
};

template<>
struct yaml::CustomMappingTraits<revng::Configuration::RootsMap> {
  static void
  inputOne(IO &IO, StringRef Key, revng::Configuration::RootsMap &Data) {
    IO.mapRequired(Key.str().c_str(), Data[Key.str()]);
  }

  static void output(IO &IO, revng::Configuration::RootsMap &Data) {
    for (auto &[Key, Value] : Data)
      IO.mapRequired(Key.c_str(), Value);
  }
};

void revng::Configuration::dump() const {
  serialize(dbg, *this);
}

template<typename T>
static bool setEitherOr(T &Destination, T &&Other) {
  if (Destination.empty()) {
    Destination = std::move(Other);
    return true;
  } else {
    return false;
  }
}

Error revng::DebugInfoServerURLs::merge(DebugInfoServerURLs &&Other) {
  appendTo(std::move(Other.Dwarf), Dwarf);
  appendTo(std::move(Other.PE), PE);
  return Error::success();
}

Error revng::Configuration::merge(Configuration &&Other) {
  appendTo(std::move(Other.TranslationLDFlags), TranslationLDFlags);

  if (not setEitherOr(CachePath, std::move(Other.CachePath))) {
    const char *Message = "Conflicting values found for cache-path while "
                          "merging configurations.";
    return revng::createError(Message);
  }

  if (auto Error = DebugInfoServerURLs
                     .merge(std::move(Other.DebugInfoServerURLs))) {
    return Error;
  }

  for (auto &[Name, Entry] : Other.Root) {
    if (Root.contains(Name)) {
      const char *Message = "Conflicting rootfs identifiers found while merging"
                            "configurations.";
      return revng::createError(Message);
    }

    Root[std::move(Name)] = std::move(Entry);
  }

  return Error::success();
}

static void expandString(std::string &String) {
  replaceAll(String, "$INSTALL_ROOT", getCurrentRoot().str());
}

static void expandStrings(std::vector<std::string> &Strings) {
  for_each(Strings, expandString);
}

void revng::DebugInfoServerURLs::finalize() {
  expandStrings(Dwarf);
  expandStrings(PE);
}

void revng::RootEntry::finalize() {
  expandString(Path);

  switch (OperatingSystem) {
  case model::OperatingSystem::Linux:
    OSSpecific = std::make_unique<LinuxRootFS>(*this);
    break;
  case model::OperatingSystem::Windows:
    OSSpecific = std::make_unique<WindowsRoot>(*this);
    break;
  case model::OperatingSystem::MacOS:
    OSSpecific = std::make_unique<MacOSRoot>(*this);
    break;
  case model::OperatingSystem::Invalid:
  case model::OperatingSystem::Count:
    revng_abort();
  }
}

void revng::Configuration::finalize() {
  expandStrings(TranslationLDFlags);
  expandString(CachePath);
  DebugInfoServerURLs.finalize();

  for (auto &[Key, Value] : Root)
    Value.finalize();
}

Expected<revng::Configuration>
revng::Configuration::fromPath(const Twine &Path) {
  std::string ThePath = Path.str();

  if (not sys::fs::exists(Path))
    return revng::Configuration();

  auto MaybeBuffer = errorOrToExpected(MemoryBuffer::getFileOrSTDIN(ThePath));

  if (not MaybeBuffer)
    return MaybeBuffer.takeError();

  using revng::Configuration;
  auto MaybeResult = fromString<Configuration>((*MaybeBuffer)->getBuffer());

  if (not MaybeResult)
    return MaybeResult.takeError();

  return MaybeResult;
}

Error revng::Configuration::reload() {
  std::string Path = joinPath(getCurrentRoot(),
                              "etc",
                              "revng",
                              "configuration.yml");
  auto MaybeNewConfiguration = fromPath(Path);

  if (not MaybeNewConfiguration) {
    return createFileError(Path, MaybeNewConfiguration.takeError());
  }

  Path = joinPath(getConfigDirectory(), "configuration.yml");
  auto MaybeUserConfiguration = fromPath(Path);

  if (not MaybeUserConfiguration) {
    return createFileError(Path, MaybeUserConfiguration.takeError());
  }

  auto &UserConfiguration = *MaybeUserConfiguration;
  auto &NewConfiguration = *MaybeNewConfiguration;

  // TODO: wrap error providing more context
  if (auto Error = NewConfiguration.merge(std::move(UserConfiguration)))
    return Error;

  TheConfiguration = std::move(NewConfiguration);

  TheConfiguration.finalize();

  ConfigurationLoaded = true;

  return Error::success();
}

std::optional<std::string>
revng::RootEntry::getExistingPath(StringRef RootRelativePath) const {
  revng_assert(not RootRelativePath.empty());

  if (isCaseSensitive()) {
    SmallString<16> FullPath;
    sys::path::append(FullPath, Path, RootRelativePath);
    if (sys::fs::exists(FullPath))
      return FullPath.str().str();
    else
      return std::nullopt;
  } else {
    return findPathCaseInsensitive(Path, RootRelativePath);
  }
}

SmallVector<StringRef, 2>
revng::WindowsRoot::getRedirectionsFor(const std::string &LibraryName) {
  loadRedirections();

  revng_log(Log, "Getting redirections for " << LibraryName);
  LoggerIndent Indent(Log);

  auto Get = [&LibraryName](const revng::WindowsLibraryMap &Redirections) {
    SmallVector<StringRef, 2> Result;

    std::string LowerCase = StringRef(LibraryName).lower();
    StringRef Normalized(LowerCase);
    bool EndsWithDLL = Normalized.consume_back(".dll");

    if (not EndsWithDLL) {
      revng_log(Log, "No .dll suffix, bailing out");
      return Result;
    }

    auto NormalizedString = Normalized.substr(Redirections.PrefixLength);

    for (const auto &[Key, Value] :
         Redirections.Map.find_prefixes(NormalizedString.str())) {
      revng_log(Log, Value);
      Result.push_back(Value);
    }

    return Result;
  };

  return Get(Redirections.value());
}

std::string revng::WindowsRoot::getSystemDirectory() {
  if (SystemDirectory.has_value())
    return *SystemDirectory;

  using namespace sys::path;
  SmallString<16> Result;
  append(Result, StringRef("Windows"), StringRef("SysWoW64"));

  bool Is64 = model::Architecture::getPointerSize(Parent.Architecture) != 8;
  if (Parent.getExistingPath(Result) and not Is64) {
    // SysWoW64 contains 32-bit binaries on a 64-bit Windows installation
    append(Result, StringRef("SysWoW64"));
  } else {
    // System32 is called 32 for historical reasons, it contains 32-bit
    // binaries on 32-bit installation and 64-bit binaries on 64-bit
    // installations
    Result.clear();
    append(Result, StringRef("Windows"), StringRef("System32"));
  }

  return Result.str().str();
}

void revng::WindowsRoot::loadRedirections() {
  if (Redirections.has_value()) {
    return;
  }

  revng_log(Log, "Loading redirections");
  LoggerIndent Indent(Log);

  SmallString<16> ApiSetSchemaPath;
  sys::path::append(ApiSetSchemaPath, getSystemDirectory(), "apisetschema.dll");

  auto MaybePath = Parent.getExistingPath(ApiSetSchemaPath);
  if (not MaybePath.has_value()) {
    revng_log(Log, "File not found: " << ApiSetSchemaPath.str().str());
    Redirections.emplace();
    return;
  }

  Redirections = parseWindowsApiSetSchemaFromDLL(*MaybePath);
}

void revng::LinuxRootFS::addLdSoConfPaths(SmallVectorImpl<std::string> &Paths) {
  loadLdSoConfPaths();

  for (const std::string &Path : *LdSoConfPaths)
    Paths.push_back(Path);
}

/// \see ldconfig.c from glibc
class LdSoConfParser {
private:
  static constexpr unsigned MaxIncludeDepth = 5;

private:
  const revng::RootEntry &Root;
  SmallVectorImpl<std::string> &SearchPaths;
  std::set<std::string> VisitedFiles;

public:
  LdSoConfParser(const revng::RootEntry &Root,
                 SmallVectorImpl<std::string> &SearchPaths) :
    Root(Root), SearchPaths(SearchPaths) {}

public:
  void parse() {
    std::set<std::string> VisitedFiles;
    parseImpl("/etc/ld.so.conf", 0);
  }

private:
  void parseImpl(StringRef Path, unsigned Depth) {
    revng_log(Log, "Parsing " << Path);
    LoggerIndent Indent(Log);

    if (Depth > MaxIncludeDepth) {
      revng_log(Log,
                "More than " << MaxIncludeDepth
                             << " nested includes in ld.so.conf");
      return;
    }

    bool IsNew = VisitedFiles.insert(Path.str()).second;
    if (not IsNew) {
      revng_log(Log, "We already visited " << Path.str() << ". Ignoring.");
      return;
    }

    using namespace sys;
    StringRef Directory = path::parent_path(Path);

    SmallString<16> FullPath;
    sys::path::append(FullPath, Root.Path, Path);
    auto MaybeBuffer = MemoryBuffer::getFile(FullPath);
    if (not MaybeBuffer) {
      serialize(dbg, Root);
      revng_log(Log, "Can't open " << FullPath.str().str());
      return;
    }

    StringRef Data = MaybeBuffer->get()->getBuffer();

    // Split in lines
    SmallVector<StringRef, 8> Lines;
    Data.split(Lines, "\n");

    for (StringRef Line : Lines) {
      // Remove comments
      Line = Line.split("#").first;

      // Strip white spaces
      Line = Line.trim();

      // Ignore empty lines
      if (Line.size() == 0)
        continue;

      if (Line.consume_front("include ")) {
        Line = Line.trim();

        SmallString<32> GlobExpression(Root.Path);
        sys::path::append(GlobExpression, makeAbsolute(Line, Directory));
        for (StringRef File : glob(GlobExpression)) {
          // Strip root path
          bool Done = File.consume_front(Root.Path);
          revng_assert(Done);

          // Proceed
          parseImpl(File, Depth + 1);
        }

      } else if (Line.consume_front("hwcap ")) {

        revng_log(Log, "Ignoring hwcap line");

      } else {

        // Add the path
        SmallString<32> SearchPath = makeAbsolute(Line, Directory);
        revng_log(Log, "Registering " << SearchPath.str().str());
        SearchPaths.push_back(SearchPath.str().str());
      }
    }
  }

private:
  SmallString<32> makeAbsolute(StringRef Path, StringRef CurrentDirectory) {
    SmallString<32> Result;

    if (Path.starts_with("/")) {
      Result.append(Path);
    } else {
      Result = CurrentDirectory;
      sys::path::append(Result, Path);
    }

    return Result;
  }

  static cppcoro::generator<StringRef> glob(StringRef Pattern) {
    glob64_t GlobResults;
    int Result = glob64(Pattern.str().data(), 0, NULL, &GlobResults);

    switch (Result) {
    case 0:
      for (size_t I = 0; I < GlobResults.gl_pathc; ++I)
        co_yield StringRef(GlobResults.gl_pathv[I]);
      globfree64(&GlobResults);
      break;

    case GLOB_NOMATCH:
      break;

    case GLOB_NOSPACE:
    case GLOB_ABORTED:
      revng_log(Log, "Cannot read directory");
      break;

    default:
      revng_abort();
      break;
    }
  }
};

void revng::LinuxRootFS::loadLdSoConfPaths() {
  if (LdSoConfPaths.has_value())
    return;

  LdSoConfParser(Parent, LdSoConfPaths.emplace()).parse();
}
