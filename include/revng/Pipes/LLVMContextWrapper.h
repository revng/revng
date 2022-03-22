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
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::pipes {
class LLVMContextWrapper : public pipeline::SavableObject<LLVMContextWrapper> {
public:
  static const char ID;
  const llvm::LLVMContext &getContext() const { return Ctx; }

  llvm::LLVMContext &getContext() { return Ctx; }

  ~LLVMContextWrapper() override = default;

  llvm::Error storeToDisk(llvm::StringRef Path) const final {
    return llvm::Error::success();
  }
  llvm::Error loadFromDisk(llvm::StringRef Path) final {
    return llvm::Error::success();
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {
    return llvm::Error::success();
  }
  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {
    return llvm::Error::success();
  }
  void clear() final {}

private:
  llvm::LLVMContext Ctx;
};

} // namespace revng::pipes
