/// \file ImportBinaryAnalysis.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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

llvm::Error ImportBinaryAnalysis::run(pipeline::ExecutionContext &Context,
                                      const BinaryFileContainer &SourceBinary) {
  if (not SourceBinary.exists())
    return llvm::Error::success();

  TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

  const ImporterOptions &Options = importerOptions();

  llvm::Task T(2, "Import binary");
  T.advance("Import main binary", true);

  if (llvm::Error Error = importBinary(Model, *SourceBinary.path(), Options))
    return Error;

  T.advance("Import additional debug info", true);
  if (!Options.AdditionalDebugInfoPaths.empty()) {
    DwarfImporter Importer(Model);
    llvm::Task T2(Options.AdditionalDebugInfoPaths.size(),
                  "Import additional debug info");
    for (const std::string &Path : Options.AdditionalDebugInfoPaths) {
      T2.advance(Path, true);
      Importer.import(Path, Options);
    }
  }

  return llvm::Error::success();
}

static pipeline::RegisterAnalysis<ImportBinaryAnalysis> E;
