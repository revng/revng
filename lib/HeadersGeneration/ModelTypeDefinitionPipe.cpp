//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompileFunction.h"
#include "revng/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pypeline::piperuns {

using TD = UpcastablePointer<model::TypeDefinition>;
void GenerateModelTypeDefinition::runOnTypeDefinition(const TD
                                                        &TypeDefinition) {
  auto OS = Output.getOStream(ObjectID(TypeDefinition->key()));
  ptml::ModelCBuilder B(*OS,
                        *Model.get().get(),
                        true,
                        { .EnablePrintingOfTheMaximumEnumValue = true,
                          .EnableExplicitPadding = false });

  upcast(TypeDefinition,
         [&B]<typename T>(const T &Upcasted) { B.printDefinition(Upcasted); });
}

} // namespace revng::pypeline::piperuns
