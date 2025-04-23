#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

#include "revng/Storage/StorageClient.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"
#include "revng/Support/PathList.h"

namespace revng {

class PathBase {
protected:
  revng::StorageClient *Client = nullptr;
  std::string SubPath;

public:
  explicit PathBase(StorageClient *Client, llvm::StringRef SubPath) :
    Client(Client), SubPath(SubPath.str()){};

protected:
  explicit PathBase(std::nullptr_t, llvm::StringRef SubPath) :
    SubPath(SubPath.str()){};

public:
  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &OS) const debug_function {
    if (Client != nullptr) {
      OS << "Client: ";
      Client->dump(OS);
    } else {
      OS << "Client: <nullptr>\n";
    }
    OS << " Path: " << SubPath << "\n";
  }
};

class FilePath : public PathBase {
public:
  using PathBase::PathBase;

  /// Utility function to get a FilePath that can read stdin
  static FilePath stdin() { return FilePath(nullptr, "stdin"); }

  /// Utility function to get a FilePath that can write stdout
  static FilePath stdout() { return FilePath(nullptr, "stdout"); }

  /// Utility function to get a FilePath on the local filesystem
  static FilePath fromLocalStorage(llvm::StringRef Path);

  /// This function will return another FilePath with the extension added
  FilePath addExtension(llvm::StringRef Extension) const {
    if (Client == nullptr)
      return *this;

    using llvm::sys::path::get_separator;
    llvm::StringRef Separator = get_separator(Client->getStyle());
    revng_assert(Extension.find('.') == llvm::StringRef::npos);
    revng_assert(Extension.find(Separator) == llvm::StringRef::npos);
    return FilePath(Client, SubPath + '.' + Extension.str());
  }

  llvm::Error check() const {
    if (Client == nullptr)
      return llvm::Error::success();

    auto MaybeResult = Client->type(SubPath);
    if (!MaybeResult)
      return MaybeResult.takeError();

    if (MaybeResult.get() == PathType::Directory)
      return revng::createError("Provided path is a directory");

    return llvm::Error::success();
  }

  llvm::Expected<bool> exists() const {
    if (Client == nullptr)
      return false;

    auto MaybeResult = Client->type(SubPath);
    if (!MaybeResult)
      return MaybeResult.takeError();

    revng_assert(MaybeResult.get() != PathType::Directory);
    return MaybeResult.get() == PathType::File;
  }

  llvm::Error remove() const {
    revng_assert(Client != nullptr);
    return Client->remove(SubPath);
  }

  /// This function will allow the user of a FilePath to obtain a wrapped
  /// llvm::MemoryBuffer that can be used for reading
  llvm::Expected<std::unique_ptr<ReadableFile>> getReadableFile() const {
    if (Client != nullptr)
      return Client->getReadableFile(SubPath);

    revng_assert(SubPath == "stdin");
    auto MaybeBuffer = llvm::MemoryBuffer::getSTDIN();
    if (not MaybeBuffer) {
      return llvm::createStringError(MaybeBuffer.getError(),
                                     "Could not open stdin");
    }

    return std::make_unique<ReadableFile>(std::move(MaybeBuffer.get()));
  };

  /// This function will allow the user of a FilePath to obtain a wrapped
  /// llvm::raw_ostream that can be used to write to the file (reminder to then
  /// call WritableFile::commit). The Encoding parameter is useful only on some
  /// backend (currently, S3) which tell the storage provider the encoding of
  /// the file.
  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(ContentEncoding Encoding = ContentEncoding::None) const {
    if (Client != nullptr)
      return Client->getWritableFile(SubPath, Encoding);

    revng_assert(SubPath == "stdout");
    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>("-",
                                                     EC,
                                                     llvm::sys::fs::OF_None);
    if (EC)
      return llvm::createStringError(EC, "Could not open stdout");

    return std::make_unique<WritableFile>(std::move(OS));
  }

  llvm::Error copyTo(const FilePath &Destination) const {
    if (Client != nullptr && Destination.Client != nullptr
        && Client == Destination.Client)
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
    return llvm::Error::success();
  }
};

class DirectoryPath : public PathBase {
public:
  using PathBase::PathBase;

  static DirectoryPath fromLocalStorage(llvm::StringRef Path);

public:
  DirectoryPath getDirectory(llvm::StringRef DirectoryName) const {
    revng_assert(Client != nullptr);
    std::string Path = joinPath(Client->getStyle(), SubPath, DirectoryName);
    return DirectoryPath(Client, Path);
  }

  FilePath getFile(llvm::StringRef Filename) const {
    revng_assert(Client != nullptr);
    return FilePath(Client, joinPath(Client->getStyle(), SubPath, Filename));
  }

  llvm::Expected<bool> exists() const {
    revng_assert(Client != nullptr);
    auto MaybeResult = Client->type(SubPath);
    if (!MaybeResult)
      return MaybeResult.takeError();

    revng_assert(MaybeResult.get() != PathType::File);
    return MaybeResult.get() == PathType::Directory;
  }

  llvm::Error create() const {
    revng_assert(Client != nullptr);
    return Client->createDirectory(SubPath);
  }

  bool isValid() const { return Client != nullptr; }
};

} // namespace revng
