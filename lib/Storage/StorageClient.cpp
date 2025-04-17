/// \file LocalStorageClient.cpp
/// \brief Implementation of StorageClient operations on the local filesystem

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Storage/StorageClient.h"
#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"

/// Utility function that checks the "cleanliness" of a path, returning false if
/// the path is "dirty".
/// Things that will result in a dirty path:
/// * Use of a leading '/'
/// * Use of './'
/// * Use of '../'
/// * Use of '//'
static bool checkPath(llvm::StringRef InputPath, llvm::sys::path::Style Style) {
  llvm::StringRef Separator = llvm::sys::path::get_separator(Style);

  // TODO: how to make this windows-compatible?
  if (InputPath.starts_with(Separator))
    return false;

  bool TrailingSlash = InputPath.ends_with(Separator);

  llvm::SmallString<128> Temp(joinPath(Style, Separator, InputPath.str()));
  llvm::sys::path::remove_dots(Temp, true, Style);
  if (TrailingSlash)
    Temp.append(Separator);

  return InputPath == Temp.substr(1);
}

namespace revng {

std::string StorageClient::resolvePath(llvm::StringRef Path) {
  if (Path.empty()) {
    return Root;
  } else {
    revng_assert(checkPath(Path, getStyle()));
    return joinPath(getStyle(), Root, Path);
  }
}

StorageClient::StorageClient(llvm::StringRef Root) : Root(Root.str()) {
  revng_assert(not Root.empty());
};

std::string StorageClient::dumpString() const {
  return Root;
}

llvm::Expected<PathType> StorageClient::type(llvm::StringRef Path) {
  std::string ResolvedPath = resolvePath(Path);
  if (llvm::sys::fs::exists(ResolvedPath)) {
    if (llvm::sys::fs::is_directory(ResolvedPath)) {
      return PathType::Directory;
    } else {
      // If Path is not in files it means that the file existed before the
      // StorageClient was created, add an entry with index 0.
      if (Metadata.Files.count(Path) == 0)
        Metadata.Files[Path] = { .Index = 0 };
      return PathType::File;
    }
  } else {
    return PathType::Missing;
  }
}

llvm::Error StorageClient::createDirectory(llvm::StringRef Path) {
  std::string ResolvedPath = resolvePath(Path);
  std::error_code EC = llvm::sys::fs::create_directory(ResolvedPath);
  if (EC) {
    return llvm::createStringError(EC,
                                   "Could not create directory %s",
                                   ResolvedPath.c_str());
  }

  return llvm::Error::success();
}

llvm::Error StorageClient::remove(llvm::StringRef Path) {
  std::string ResolvedPath = resolvePath(Path);
  std::error_code EC = llvm::sys::fs::remove(ResolvedPath);
  if (EC) {
    return llvm::createStringError(EC,
                                   "Could not remove file %s",
                                   ResolvedPath.c_str());
  }

  Metadata.Files.erase(Path);
  return llvm::Error::success();
}

llvm::sys::path::Style StorageClient::getStyle() const {
  return llvm::sys::path::Style::native;
}

llvm::Error StorageClient::copy(llvm::StringRef Source,
                                llvm::StringRef Destination) {
  std::string ResolvedSource = resolvePath(Source);
  std::string ResolvedDestination = resolvePath(Destination);
  std::error_code EC = llvm::sys::fs::copy_file(ResolvedSource.c_str(),
                                                ResolvedDestination.c_str());
  if (EC) {
    return llvm::createStringError(EC,
                                   "Could not copy file %s to %s",
                                   ResolvedSource.c_str(),
                                   ResolvedDestination.c_str());
  }

  Metadata.Files[Destination] = { .Index = Metadata.TransactionIndex,
                                  .Encoding = Metadata.Files[Source].Encoding };
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<ReadableFile>>
StorageClient::getReadableFile(llvm::StringRef Path) {
  std::string ResolvedPath = resolvePath(Path);
  auto MaybeBuffer = llvm::MemoryBuffer::getFile(ResolvedPath);
  if (not MaybeBuffer) {
    return llvm::createStringError(MaybeBuffer.getError(),
                                   "Could not open file %s for reading",
                                   ResolvedPath.c_str());
  }

  // If Path is not in files it means that the file existed before the
  // StorageClient was created, add an entry with index 0.
  if (Metadata.Files.count(Path) == 0)
    Metadata.Files[Path] = { .Index = 0 };
  return std::make_unique<ReadableFile>(std::move(MaybeBuffer.get()));
}

llvm::Expected<std::unique_ptr<WritableFile>>
StorageClient::getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) {
  std::string ResolvedPath = resolvePath(Path);
  std::error_code EC;
  auto OS = std::make_unique<llvm::raw_fd_ostream>(ResolvedPath,
                                                   EC,
                                                   llvm::sys::fs::OF_None);
  if (EC) {
    return llvm::createStringError(EC,
                                   "Could not open file %s for writing",
                                   ResolvedPath.c_str());
  }

  Metadata.Files[Path] = { .Index = Metadata.TransactionIndex,
                           .Encoding = Encoding };

  return std::make_unique<WritableFile>(std::move(OS));
}

} // namespace revng

template<>
struct llvm::yaml::MappingTraits<revng::StorageClient::FileMetadata> {
  static void mapping(IO &IO, revng::StorageClient::FileMetadata &Obj) {
    IO.mapRequired("Index", Obj.TransactionIndex);
    IO.mapRequired("Files", Obj.Files);
  }
};

using FileDataMap = llvm::StringMap<revng::StorageClient::File>;
template<>
struct llvm::yaml::CustomMappingTraits<FileDataMap> {
  static void inputOne(IO &, StringRef, FileDataMap &) {
    revng_abort("Output only");
  }

  static void output(IO &IO, FileDataMap &Obj) {
    for (auto &&[Key, Value] : Obj) {
      IO.mapRequired(Key.str().c_str(), Value);
    }
  }
};

LLVM_YAML_IS_STRING_MAP(revng::StorageClient::File)

template<>
struct llvm::yaml::MappingTraits<revng::StorageClient::File> {
  static void mapping(IO &IO, revng::StorageClient::File &Obj) {
    IO.mapRequired("Index", Obj.Index);
    IO.mapRequired("Encoding", Obj.Encoding);
  }
};

template<>
struct llvm::yaml::ScalarEnumerationTraits<revng::ContentEncoding> {
  static void enumeration(IO &IO, revng::ContentEncoding &Value) {
    IO.enumCase(Value, "None", revng::ContentEncoding::None);
    IO.enumCase(Value, "Gzip", revng::ContentEncoding::Gzip);
  }
};

llvm::Error revng::StorageClient::commit() {
  // Write the `file-list.yml` file, this file contains
  // * The transaction index
  // * A serialization of the Files property
  // These are needed for downstream synchronizers to effectively save the
  // resume directory externally.
  std::string ResolvedPath = resolvePath("file-list.yml");
  std::error_code EC;
  llvm::raw_fd_ostream OS(ResolvedPath, EC);
  if (EC)
    return llvm::createStringError(EC, "Could not write file list");

  llvm::yaml::Output Out(OS);
  Out << Metadata;
  OS.flush();

  // Increase the transaction index
  Metadata.TransactionIndex++;
  return llvm::Error::success();
}
