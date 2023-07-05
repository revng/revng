#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Storage/StorageClient.h"
#include "revng/Support/Debug.h"

namespace revng {

class LocalStorageClient : public StorageClient {
private:
  std::string Root;

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

private:
  std::string dumpString() const override;
  std::string resolvePath(llvm::StringRef Path);
};

} // namespace revng
