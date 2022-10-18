#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

class LinkForTranslation {
public:
  static constexpr auto Name = "LinkForTranslation";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract BinaryPart(kinds::Binary,
                                  0,
                                  kinds::Translated,
                                  2,
                                  pipeline::InputPreservation::Preserve);
    pipeline::Contract ObjectPart(kinds::Object,
                                  1,
                                  kinds::Translated,
                                  2,
                                  pipeline::InputPreservation::Preserve);
    return { pipeline::ContractGroup({ BinaryPart, ObjectPart }) };
  }

  void run(const pipeline::Context &Ctx,
           BinaryFileContainer &InputBinary,
           ObjectFileContainer &ObjectFile,
           TranslatedFileContainer &OutputBinary);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
