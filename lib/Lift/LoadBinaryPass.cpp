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

namespace {
using namespace llvm::cl;
opt<std::string> RawBinaryPath("binary-path", desc("<raw binary path>"));
} // namespace

char LoadBinaryWrapperPass::ID;

using Register = llvm::RegisterPass<LoadBinaryWrapperPass>;
static Register X("load-binary", "Load Binary Pass", true, true);

LoadBinaryWrapperPass::LoadBinaryWrapperPass() : llvm::ModulePass(ID) {
  revng_check(RawBinaryPath.getNumOccurrences() == 1);
  auto Result = MemoryBuffer::getFileOrSTDIN(RawBinaryPath);
  MaybeBuffer = cantFail(errorOrToExpected(std::move(Result)));
  Data = toArrayRef(MaybeBuffer->getBuffer());
}

bool LoadBinaryWrapperPass::runOnModule(llvm::Module &M) {
  auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
  BinaryView.emplace(*Model, Data);
  return false;
}

llvm::Expected<std::pair<RawBinaryView, std::unique_ptr<llvm::MemoryBuffer>>>
loadBinary(const model::Binary &Model, llvm::StringRef BinaryPath) {
  revng_assert(Model.verify());

  auto FileContentsBuffer = llvm::MemoryBuffer::getFileOrSTDIN(BinaryPath);
  if (auto ErrorCode = FileContentsBuffer.getError())
    return llvm::errorCodeToError(std::move(ErrorCode));

  std::unique_ptr<llvm::MemoryBuffer> Result = std::move(*FileContentsBuffer);
  RawBinaryView ResultView(Model, toArrayRef(Result->getBuffer()));
  return std::pair{ std::move(ResultView), std::move(Result) };
}
