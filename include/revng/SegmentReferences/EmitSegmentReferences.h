#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class EmitSegmentReferences {
private:
  const model::Binary &Binary;
  LLVMRootContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "emit-segment-references";
  using Arguments = TypeList<
    PipeRunArgument<LLVMRootContainer, "Module", "Root LLVM module">>;

  EmitSegmentReferences(const class Model &Model,
                        llvm::StringRef StaticConfiguration,
                        llvm::StringRef Configuration,
                        LLVMRootContainer &ModuleContainer) :
    Binary(*Model.get().get()), ModuleContainer(ModuleContainer) {}

  void run();
};

} // namespace revng::pypeline::piperuns
