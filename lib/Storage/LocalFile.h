#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/Error.h"

#include "revng/Storage/ReadableFile.h"
#include "revng/Storage/WritableFile.h"

namespace revng {

class LocalReadableFile : public ReadableFile {
private:
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

public:
  LocalReadableFile(std::unique_ptr<llvm::MemoryBuffer> &&Buffer) :
    Buffer(std::move(Buffer)) {}
  ~LocalReadableFile() override = default;

  llvm::MemoryBuffer &buffer() override { return *Buffer; };
};

class LocalWritableFile : public WritableFile {
private:
  std::unique_ptr<llvm::raw_fd_ostream> OS;

public:
  LocalWritableFile(std::unique_ptr<llvm::raw_fd_ostream> &&OS) :
    OS(std::move(OS)) {}
  ~LocalWritableFile() override = default;
  llvm::raw_pwrite_stream &os() override { return *OS; }
  llvm::Error commit() override { return llvm::Error::success(); }
};

} // namespace revng
