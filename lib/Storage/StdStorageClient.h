#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Storage/StorageClient.h"

#include "LocalFile.h"

namespace revng {

namespace detail {

class StdStorageClient : public StorageClient {
public:
  StdStorageClient() = default;
  ~StdStorageClient() override = default;

  llvm::Expected<PathType> type(llvm::StringRef Path) override {
    return PathType::Missing;
  };

  llvm::Error createDirectory(llvm::StringRef Path) override { revng_abort(); };
  llvm::Error remove(llvm::StringRef Path) override { revng_abort(); };
  llvm::sys::path::Style getStyle() const override {
    return llvm::sys::path::Style::posix;
  };

  llvm::Error copy(llvm::StringRef Source,
                   llvm::StringRef Destination) override {
    revng_abort();
  };
};

} // namespace detail

class StdinStorageClient : public detail::StdStorageClient {
public:
  StdinStorageClient() = default;

  llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path) override {
    revng_assert(Path == "");
    auto MaybeBuffer = llvm::MemoryBuffer::getSTDIN();
    if (not MaybeBuffer) {
      return llvm::createStringError(MaybeBuffer.getError(),
                                     "Could not open stdin");
    }

    return std::make_unique<LocalReadableFile>(std::move(MaybeBuffer.get()));
  }

  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) override {
    revng_abort();
  }

private:
  virtual std::string dumpString() const override { return "stdin"; }
};

class StdoutStorageClient : public detail::StdStorageClient {
public:
  StdoutStorageClient() = default;

  llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path) override {
    revng_abort();
  }

  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) override {
    using llvm::sys::fs::OF_None;
    if (Path != "") {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot open file");
    }

    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>("-", EC, OF_None);
    if (EC)
      return llvm::createStringError(EC, "Could not open stdout");

    return std::make_unique<LocalWritableFile>(std::move(OS));
  }

private:
  virtual std::string dumpString() const override { return "stdout"; }
};

} // namespace revng
