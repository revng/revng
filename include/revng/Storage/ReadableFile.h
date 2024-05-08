#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/MemoryBuffer.h"

namespace revng {

class ReadableFile {
protected:
  ReadableFile() = default;

public:
  virtual ~ReadableFile() = default;

  ReadableFile(const ReadableFile &Other) = delete;
  ReadableFile &operator=(const ReadableFile &Other) = delete;
  ReadableFile(ReadableFile &&Other) = delete;
  ReadableFile &operator=(ReadableFile &&Other) = delete;

  virtual llvm::MemoryBuffer &buffer() = 0;
};

} // namespace revng
