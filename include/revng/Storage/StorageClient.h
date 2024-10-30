#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "revng/Storage/ReadableFile.h"
#include "revng/Storage/WritableFile.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"

namespace revng {

enum class ContentEncoding {
  None,
  Gzip
};

enum class PathType {
  Missing,
  File,
  Directory
};

/// Class that abstracts storage from its users.
/// This allows scoped access to a specific "directory", in the sense that no
/// file parent of the root directory can be accessed. This has been done
/// deliberately since it synergises well with PipelineManager's work directory.
///
/// All methods are failable since the storage might not be on the local
/// filesystem and, as such, is subject to network errors. Moreover, it is not
/// guaranteed that the buffers/streams returned by ::getWritableFile or
/// ::getReadableFile are actually backed by a file in the filesystem.
///
/// Currently the following backends are supported:
/// * Local filesystem via ordinary unix paths
/// * S3 storage via the 's3://' and 's3s://` uris
///
/// Writing files happens in 3 steps:
/// * A writable file is requested via ::getWritableFile
/// * The writable file is committed via WritableFile::commit
/// * Once all writes are done, a final ::commit must be
///   issued, otherwise all changes might be lost
///
/// Example:
/// \code{.cpp}
/// llvm::Error serialize() {
///   for (auto &Path : pathsToSave()) {
///     auto MaybeFile = Client.getWritableFile(Path);
///     if (not MaybeFile)
///       return MaybeFile.takeError();
///
///     serialize(Path, MaybeFile.get()->os());
///
///     if (auto Error = MaybeFile->commit())
///       return Error;
///   }
///
///   return Client.commit();
/// }
/// \endcode
class StorageClient {
protected:
  StorageClient() = default;

public:
  virtual ~StorageClient() = default;
  StorageClient(const StorageClient &Other) = delete;
  StorageClient &operator=(const StorageClient &Other) = delete;
  StorageClient(const StorageClient &&Other) = delete;
  StorageClient &operator=(StorageClient &&Other) = delete;

public:
  static llvm::Expected<std::unique_ptr<StorageClient>>
  fromPathOrURL(llvm::StringRef URL);

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &OS) const debug_function {
    OS << dumpString();
  }

  virtual llvm::Expected<PathType> type(llvm::StringRef Path) = 0;
  virtual llvm::Error createDirectory(llvm::StringRef Path) = 0;
  virtual llvm::Error remove(llvm::StringRef Path) = 0;
  virtual llvm::sys::path::Style getStyle() const = 0;

  virtual llvm::Error copy(llvm::StringRef Source,
                           llvm::StringRef Destination) = 0;

  virtual llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path) = 0;

  virtual llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) = 0;

  virtual llvm::Error commit() { return llvm::Error::success(); };

  virtual llvm::Error setCredentials(llvm::StringRef Credentials) {
    return revng::createError("Not Supported");
  }

private:
  virtual std::string dumpString() const = 0;
};

} // namespace revng
