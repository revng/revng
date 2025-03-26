#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringMap.h"

#include "revng/Storage/StorageClient.h"

namespace revng {

class LocalStorageClient : public StorageClient {
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
  LocalStorageClient(llvm::StringRef Root);
  ~LocalStorageClient() override = default;

  llvm::Expected<PathType> type(llvm::StringRef Path) override;
  llvm::Error createDirectory(llvm::StringRef Path) override;
  llvm::Error remove(llvm::StringRef Path) override;
  llvm::sys::path::Style getStyle() const override;

  llvm::Error copy(llvm::StringRef Source,
                   llvm::StringRef Destination) override;

  llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path) override;

  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) override;

  llvm::Error commit() override;

private:
  std::string dumpString() const override;
  std::string resolvePath(llvm::StringRef Path);
};

} // namespace revng
