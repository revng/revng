/// \file ImportBinaryPipe.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterOptions.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ImportBinaryAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/TupleTree/TupleTree.h"

using namespace revng::pipes;

void ImportBinaryAnalysis::run(pipeline::Context &Context,
                               const FileContainer &SourceBinary) {
  if (not SourceBinary.exists())
    return;

  TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

  revng_check(not importBinary(Model, *SourceBinary.path(), BaseAddress));

  if (ImportDebugInfo.size() > 0) {
    DwarfImporter Importer(Model);
    for (const std::string &Path : ImportDebugInfo)
      Importer.import(Path);
  }
}

static pipeline::RegisterAnalysis<ImportBinaryAnalysis> E;
