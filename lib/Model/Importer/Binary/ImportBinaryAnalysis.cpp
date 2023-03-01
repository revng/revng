/// \file ImportBinaryAnalysis.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/ImportBinaryAnalysis.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/TupleTree/TupleTree.h"

using namespace revng::pipes;

llvm::Error ImportBinaryAnalysis::run(pipeline::Context &Context,
                                      const BinaryFileContainer &SourceBinary) {
  if (not SourceBinary.exists())
    return llvm::Error::success();

  TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

  const ImporterOptions &Options = importerOptions();
  if (llvm::Error Error = importBinary(Model, *SourceBinary.path(), Options))
    return Error;

  if (!Options.AdditionalDebugInfoPaths.empty()) {
    DwarfImporter Importer(Model);
    for (const std::string &Path : Options.AdditionalDebugInfoPaths)
      Importer.import(Path, Options);
  }

  return llvm::Error::success();
}

static pipeline::RegisterAnalysis<ImportBinaryAnalysis> E;
