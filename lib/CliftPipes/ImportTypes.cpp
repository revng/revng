//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftPipes/ImportTypes.h"
#include "revng/Ranks/Location.h"

namespace revng::pypeline::piperuns {

void ImportTypes::run() {
  clift::importAllModelTypes(Binary, Output.getModule());
}

void ImportFunctionDeclarations::run() {
  clift::importAllModelFunctionDeclarations(Binary, Module.getModule());
}

void ImportSegmentDeclarations::run() {
  clift::importAllModelSegmentDeclarations(Binary, Module.getModule());
}

} // namespace revng::pypeline::piperuns
