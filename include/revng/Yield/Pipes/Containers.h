#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/Yield/Function.h"

namespace revng::pypeline {

class AssemblyInternalContainer
  : public TupleTreeContainer<yield::Function, Kinds::Function> {
public:
  static constexpr llvm::StringRef Name = "AssemblyInternalContainer";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

} // namespace revng::pypeline
