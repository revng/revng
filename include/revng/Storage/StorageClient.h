#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Storage/StorageClient.h"
#include "revng/Support/Debug.h"

namespace revng {

class ReadableFile {
private:
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

public:
  ReadableFile(std::unique_ptr<llvm::MemoryBuffer> &&Buffer) :
    Buffer(std::move(Buffer)) {}
  ~ReadableFile() = default;

  llvm::MemoryBuffer &buffer() { return *Buffer; };
};

class WritableFile {
private:
  std::unique_ptr<llvm::raw_fd_ostream> OS;

public:
  WritableFile(std::unique_ptr<llvm::raw_fd_ostream> &&OS) :
    OS(std::move(OS)) {}
  ~WritableFile() = default;
  llvm::raw_pwrite_stream &os() { return *OS; }
  llvm::Error commit() { return llvm::Error::success(); }
};

enum class ContentEncoding {
  None,
  Gzip
};

enum class PathType {
  Missing,
  File,
  Directory
};

class StorageClient {
public:
  struct File {
    /// TransactionIndex when the file was last written by the Client, if the
    /// value is '0' then the file existed before the Client was created.
    uint64_t Index = 0;
    /// Content Encoding of the file, this is used downstream.
    ContentEncoding Encoding = ContentEncoding::None;
  };

  struct FileMetadata {
    /// Index of the commit that will be written when calling the `commit`
    /// method of this class. Increases by one every time `commit()` is called.
    uint64_t TransactionIndex = 1;
    /// Map that stores the state of the files managed by this StorageClient. It
    /// records the last TransactionIndex the files were written and the
    /// encoding used.
    llvm::StringMap<File> Files;
  };

private:
  std::string Root;
  FileMetadata Metadata;

public:
  StorageClient(llvm::StringRef Root);
  ~StorageClient() = default;

  llvm::Expected<PathType> type(llvm::StringRef Path);
  llvm::Error createDirectory(llvm::StringRef Path);
  llvm::Error remove(llvm::StringRef Path);
  llvm::sys::path::Style getStyle() const;

  llvm::Error copy(llvm::StringRef Source, llvm::StringRef Destination);

  llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path);

  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding);

  llvm::Error commit();

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &OS) const debug_function {
    OS << dumpString();
  }

private:
  std::string dumpString() const;
  std::string resolvePath(llvm::StringRef Path);
};

} // namespace revng
