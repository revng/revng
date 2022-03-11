/// \file LoadBinaryPass.cpp
/// \brief

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
