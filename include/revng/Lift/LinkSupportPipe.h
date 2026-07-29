#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class LinkSupport {
private:
  const model::Binary &Binary;
  LLVMRootContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "link-support";
  using Arguments = TypeList<PipeRunArgument<LLVMRootContainer,
                                             "Module",
                                             "Module to link support into">>;

public:
  LinkSupport(const class Model &Model,
              llvm::StringRef Config,
              llvm::StringRef DynamicConfig,
              LLVMRootContainer &ModuleContainer);

  void run();
};

} // namespace revng::pypeline::piperuns
