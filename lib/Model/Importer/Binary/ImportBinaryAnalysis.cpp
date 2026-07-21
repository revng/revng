//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/ImportBinaryAnalysis.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pypeline::analyses {

// TODO: have a configuration to list the "preferred" roots to use for import
llvm::Error ParseBinaryAnalysis::run(Model &Model,
                                     const Request &Incoming,
                                     llvm::StringRef Configuration,
                                     const BinariesContainer &Binaries) {
  const ImporterOptions &Options = importerOptions();

  llvm::Task T(1, "Import binary");
  T.advance("Import main binary", true);

  for (size_t I = 0; I < Binaries.size(); I++) {

    auto Reference = Model.get()->getBinaryIdentifierReference(I);
    if (llvm::Error Error = importBinary(Model.get(),
                                         Binaries.getFilePath(I),
                                         Options,
                                         Reference))
      return Error;
  }

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
