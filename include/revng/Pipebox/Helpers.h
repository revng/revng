#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RawBinaryView.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline {

/// Utility piperun argument for LLVM pass-like pipes that take a single
/// LLVMFunctionContainer and also need the model to run
using SingleLLVMFunctionsArgument = TypeList<
  PipeRunArgument<LLVMFunctionContainer, "Module", "function LLVM module(s)">>;

inline RawBinaryView makeBinaryView(const Model &Model,
                                    const BinariesContainer &Binaries) {
  llvm::ArrayRef<char> BinaryBuffer = Binaries.getFile(0);
  return RawBinaryView(*Model.get().get(),
                       llvm::StringRef{ BinaryBuffer.data(),
                                        BinaryBuffer.size() });
}

} // namespace revng::pypeline
