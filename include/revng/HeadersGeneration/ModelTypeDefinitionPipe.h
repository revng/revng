#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

using PTMLCTypeContainer = TypeDefinitionToBytesContainer<"PTMLCTypeContainer",
                                                          "text/x.c+ptml">;

namespace piperuns {

class GenerateModelTypeDefinition {
private:
  const Model &Model;
  PTMLCTypeContainer &Output;

public:
  static constexpr llvm::StringRef Name = "GenerateModelTypeDefinition";
  using Arguments = TypeList<PipeRunArgument<PTMLCTypeContainer,
                                             "Output",
                                             "The output C headers of each "
                                             "Type Definition",
                                             Access::Write>>;

  GenerateModelTypeDefinition(const class Model &Model,
                              llvm::StringRef Config,
                              llvm::StringRef DynamicConfig,
                              PTMLCTypeContainer &Output) :
    Model(Model), Output(Output) {}

  void runOnTypeDefinition(const UpcastablePointer<model::TypeDefinition>
                             &TypeDefinition);
};

} // namespace piperuns

} // namespace revng::pypeline
