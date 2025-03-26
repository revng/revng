/// \file LocalStorageClient.cpp
/// \brief Implementation of StorageClient operations on the local filesystem

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/YAMLTraits.h"

#include "revng/Support/PathList.h"

#include "LocalFile.h"
#include "LocalStorageClient.h"
#include "Utils.h"

namespace revng {

std::string LocalStorageClient::resolvePath(llvm::StringRef Path) {
  if (Path.empty()) {
    return Root;
  } else {
    revng_assert(checkPath(Path, getStyle()));
    return joinPath(getStyle(), Root, Path);
  }
}

LocalStorageClient::LocalStorageClient(llvm::StringRef Root) :
  Root(Root.str()) {
  revng_assert(not Root.empty());
};

std::string LocalStorageClient::dumpString() const {
  return Root;
}

llvm::Expected<PathType> LocalStorageClient::type(llvm::StringRef Path) {
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

llvm::Error LocalStorageClient::createDirectory(llvm::StringRef Path) {
  std::string ResolvedPath = resolvePath(Path);
  std::error_code EC = llvm::sys::fs::create_directory(ResolvedPath);
  if (EC) {
    return llvm::createStringError(EC,
                                   "Could not create directory %s",
                                   ResolvedPath.c_str());
  }

  return llvm::Error::success();
}

llvm::Error LocalStorageClient::remove(llvm::StringRef Path) {
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

llvm::sys::path::Style LocalStorageClient::getStyle() const {
  return llvm::sys::path::Style::native;
}

llvm::Error LocalStorageClient::copy(llvm::StringRef Source,
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
LocalStorageClient::getReadableFile(llvm::StringRef Path) {
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
  return std::make_unique<LocalReadableFile>(std::move(MaybeBuffer.get()));
}

llvm::Expected<std::unique_ptr<WritableFile>>
LocalStorageClient::getWritableFile(llvm::StringRef Path,
                                    ContentEncoding Encoding) {
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

  return std::make_unique<LocalWritableFile>(std::move(OS));
}

} // namespace revng

template<>
struct llvm::yaml::MappingTraits<revng::LocalStorageClient::FileMetadata> {
  static void mapping(IO &IO, revng::LocalStorageClient::FileMetadata &Obj) {
    IO.mapRequired("Index", Obj.TransactionIndex);
    IO.mapRequired("Files", Obj.Files);
  }
};

using FileDataMap = llvm::StringMap<revng::LocalStorageClient::File>;
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

LLVM_YAML_IS_STRING_MAP(revng::LocalStorageClient::File)

template<>
struct llvm::yaml::MappingTraits<revng::LocalStorageClient::File> {
  static void mapping(IO &IO, revng::LocalStorageClient::File &Obj) {
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

llvm::Error revng::LocalStorageClient::commit() {
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
