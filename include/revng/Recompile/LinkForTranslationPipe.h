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

class TranslatedContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "TranslatedContainer";
  static constexpr llvm::StringRef MimeType = "application/x-executable";
};

namespace piperuns {

class LinkForTranslation {
private:
  const model::Binary &Binary;
  const BinariesContainer &Binaries;
  const ObjectFileContainer &ObjectFile;
  TranslatedContainer &Output;

public:
  static constexpr llvm::StringRef Name = "link-for-translation";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Binaries", "The input binaries">,
    PipeRunArgument<const ObjectFileContainer,
                    "ObjectFile",
                    "The complied object file">,
    PipeRunArgument<TranslatedContainer,
                    "Output",
                    "The output executable",
                    Access::Write>>;

  static llvm::Error checkPrecondition(const class Model &Model);

  LinkForTranslation(const Model &TheModel,
                     llvm::StringRef StaticConfig,
                     llvm::StringRef DynamicConfig,
                     const BinariesContainer &Binaries,
                     const ObjectFileContainer &ObjectFile,
                     TranslatedContainer &Output);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
