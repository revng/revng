#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Recompile/CompileModulePipe.h"

namespace revng::pipes {

class LinkForTranslation {
public:
  static constexpr auto Name = "link-for-translation";

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

  void run(pipeline::ExecutionContext &EC,
           BinaryFileContainer &InputBinary,
           ObjectFileContainer &ObjectFile,
           TranslatedFileContainer &OutputBinary);
};

} // namespace revng::pipes

namespace revng::pypeline {

using TranslatedContainer = BytesContainer<"TranslatedContainer",
                                           "application/x-executable">;

namespace piperuns {

class LinkForTranslation {
public:
  static constexpr llvm::StringRef Name = "LinkForTranslation";
  using Arguments = TypeList<
    PipeArgument<"Binaries", "The input binaries">,
    PipeArgument<"ObjectFile", "The complied object file">,
    PipeArgument<"Output", "The output executable", Access::Write>>;

  static llvm::Error checkPrecondition(const class Model &Model);

  static void run(const Model &TheModel,
                  llvm::StringRef StaticConfig,
                  llvm::StringRef DynamicConfig,
                  const BinariesContainer &Binaries,
                  const ObjectFileContainer &ObjectFile,
                  TranslatedContainer &Output);
};

} // namespace piperuns

} // namespace revng::pypeline
