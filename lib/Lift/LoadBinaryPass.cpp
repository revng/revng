/// \file LoadBinaryPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/RawBinaryView.h"

using namespace llvm;

char LoadBinaryWrapperPass::ID;

using Register = llvm::RegisterPass<LoadBinaryWrapperPass>;
static Register X("load-binary", "Load Binary Pass", true, true);

bool LoadBinaryWrapperPass::runOnModule(llvm::Module &M) {
  auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
  BinaryView.emplace(*Model, Data);
  return false;
}

llvm::Expected<std::pair<RawBinaryView, std::unique_ptr<llvm::MemoryBuffer>>>
loadBinary(const model::Binary &Model, llvm::StringRef BinaryPath) {
  revng_assert(Model.verify(true));

  auto FileContentsBuffer = llvm::MemoryBuffer::getFileOrSTDIN(BinaryPath);
  if (auto ErrorCode = FileContentsBuffer.getError())
    return llvm::errorCodeToError(std::move(ErrorCode));

  std::unique_ptr<llvm::MemoryBuffer> Result = std::move(*FileContentsBuffer);
  RawBinaryView ResultView(Model, toArrayRef(Result->getBuffer()));
  return std::pair{ std::move(ResultView), std::move(Result) };
}
