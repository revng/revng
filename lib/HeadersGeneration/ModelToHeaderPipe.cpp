//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/ConfigurationHelpers.h"
#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pypeline::piperuns {

ModelToHeader::ModelToHeader(const Model &TheModel,
                             llvm::StringRef StaticConfig,
                             llvm::StringRef DynamicConfig,
                             PTMLCBytesContainer &Buffer) :
  Binary(*TheModel.get().get()), Buffer(Buffer){};

void ModelToHeader::run() {
  std::unique_ptr<llvm::raw_ostream> Out = Buffer.getOStream(ObjectID());
  ptml::ModelCBuilder
    B(*Out,
      Binary,
      /* EnableTaglessMode = */ false,
      { .EnableStackFrameInlining = revng::options::EnableStackFrameInlining,
        .EnablePrintingOfTheMaximumEnumValue = true,
        .ExplicitTargetPointerSize = getExplicitPointerSize(Binary) });
  ptml::HeaderBuilder(B).printModelHeader(/*DefineOpaqueTypes*/ true);
  Out->flush();
}

} // namespace revng::pypeline::piperuns
