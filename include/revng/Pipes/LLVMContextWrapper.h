#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/SavableObject.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/IsolatedKind.h"
#include "revng/Pipes/RootKind.h"

namespace revng::pipes {
class LLVMContextWrapper : public pipeline::SavableObject<LLVMContextWrapper> {
public:
  static const char ID;
  const llvm::LLVMContext &getContext() const { return Ctx; }

  llvm::LLVMContext &getContext() { return Ctx; }

  ~LLVMContextWrapper() override = default;

  llvm::Error storeToDisk(llvm::StringRef path) const final {
    return llvm::Error::success();
  }
  llvm::Error loadFromDisk(llvm::StringRef path) final {
    return llvm::Error::success();
  }

private:
  llvm::LLVMContext Ctx;
};

} // namespace revng::pipes
