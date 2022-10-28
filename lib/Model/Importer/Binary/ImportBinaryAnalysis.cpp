/// \file ImportBinaryAnalysis.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterOptions.h"
#include "revng/Model/Importer/Binary/ImportBinaryAnalysis.h"
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

  if (auto Error = importBinary(Model, *SourceBinary.path(), BaseAddress);
      Error)
    return Error;

  if (ImportDebugInfo.size() > 0) {
    DwarfImporter Importer(Model, BaseAddress);
    for (const std::string &Path : ImportDebugInfo)
      Importer.import(Path);
  }

  return llvm::Error::success();
}

static pipeline::RegisterAnalysis<ImportBinaryAnalysis> E;
