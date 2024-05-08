#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

#include "revng/Storage/StorageClient.h"
#include "revng/Support/Debug.h"
#include "revng/Support/PathList.h"

namespace revng {

class PathBase {
protected:
  revng::StorageClient *Client = nullptr;
  std::string SubPath;

public:
  explicit PathBase(StorageClient *Client, llvm::StringRef SubPath) :
    Client(Client), SubPath(SubPath.str()){};

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &OS) const debug_function {
    OS << "Client: ";
    Client->dump(OS);
    OS << " Path: " << SubPath << "\n";
  }

  bool isValid() const { return Client != nullptr; }
};

class FilePath : public PathBase {
public:
  using PathBase::PathBase;

  /// Utility function to get a FilePath that can read stdin
  static FilePath stdin();

  /// Utility function to get a FilePath that can write stdout
  static FilePath stdout();

  /// Utility function to get a FilePath on the local filesystem
  static FilePath fromLocalStorage(llvm::StringRef Path);

  /// This function will return another FilePath with the extension added
  FilePath addExtension(llvm::StringRef Extension) const {
    using llvm::sys::path::get_separator;
    llvm::StringRef Separator = get_separator(Client->getStyle());
    revng_assert(Extension.find('.') == llvm::StringRef::npos);
    revng_assert(Extension.find(Separator) == llvm::StringRef::npos);
    return FilePath(Client, SubPath + '.' + Extension.str());
  }

  llvm::Expected<bool> exists() const {
    auto MaybeResult = Client->type(SubPath);
    if (!MaybeResult)
      return MaybeResult.takeError();

    revng_assert(MaybeResult.get() != PathType::Directory);
    return MaybeResult.get() == PathType::File;
  }

  llvm::Error remove() const { return Client->remove(SubPath); }

  /// This function will allow the user of a FilePath to obtain a wrapped
  /// llvm::MemoryBuffer that can be used for reading
  llvm::Expected<std::unique_ptr<ReadableFile>> getReadableFile() const {
    return Client->getReadableFile(SubPath);
  };

  /// This function will allow the user of a FilePath to obtain a wrapped
  /// llvm::raw_ostream that can be used to write to the file (reminder to then
  /// call WritableFile::commit). The Encoding parameter is useful only on some
  /// backend (currently, S3) which tell the storage provider the encoding of
  /// the file.
  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(ContentEncoding Encoding = ContentEncoding::None) const {
    return Client->getWritableFile(SubPath, Encoding);
  }

  llvm::Error copyTo(const FilePath &Destination) const {
    if (Client == Destination.Client)
      return Client->copy(SubPath, Destination.SubPath);

    auto MaybeReadableFile = getReadableFile();
    if (not MaybeReadableFile)
      return MaybeReadableFile.takeError();

    auto MaybeWritableFile = Destination.getWritableFile();
    if (not MaybeWritableFile)
      return MaybeWritableFile.takeError();

    auto &ReadableFile = MaybeReadableFile.get();
    auto &WritableFile = MaybeWritableFile.get();
    WritableFile->os() << ReadableFile->buffer().getBuffer();

    return WritableFile->commit();
  }
};

class DirectoryPath : public PathBase {
public:
  using PathBase::PathBase;

  static DirectoryPath fromLocalStorage(llvm::StringRef Path);

public:
  DirectoryPath getDirectory(llvm::StringRef DirectoryName) const {
    std::string Path = joinPath(Client->getStyle(), SubPath, DirectoryName);
    return DirectoryPath(Client, Path);
  }

  FilePath getFile(llvm::StringRef Filename) const {
    return FilePath(Client, joinPath(Client->getStyle(), SubPath, Filename));
  }

  llvm::Expected<bool> exists() const {
    auto MaybeResult = Client->type(SubPath);
    if (!MaybeResult)
      return MaybeResult.takeError();

    revng_assert(MaybeResult.get() != PathType::File);
    return MaybeResult.get() == PathType::Directory;
  }

  llvm::Error create() const { return Client->createDirectory(SubPath); }
};

} // namespace revng
