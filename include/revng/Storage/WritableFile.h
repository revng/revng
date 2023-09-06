#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

namespace revng {

class WritableFile {
protected:
  WritableFile() = default;

public:
  virtual ~WritableFile() = default;

  WritableFile(const WritableFile &Other) = delete;
  WritableFile &operator=(const WritableFile &Other) = delete;
  WritableFile(WritableFile &&Other) = delete;
  WritableFile &operator=(WritableFile &&Other) = delete;

  virtual llvm::raw_pwrite_stream &os() = 0;
  virtual llvm::Error commit() = 0;
};

} // namespace revng
