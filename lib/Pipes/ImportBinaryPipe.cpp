/// \file ImportBinaryPipe.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterOptions.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/ImportBinaryPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/TupleTree/TupleTree.h"

using namespace revng::pipes;

llvm::Error
revng::pipes::ImportBinaryPipe::run(pipeline::Context &Context,
                                    const FileContainer &SourceBinary) {
  if (not SourceBinary.exists())
    return llvm::Error::success();

  TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

  if (auto Error = importBinary(Model, *SourceBinary.path(), BaseAddress);
      Error)
    return Error;

  if (ImportDebugInfo.size() > 0) {
    DwarfImporter Importer(Model);
    for (const std::string &Path : ImportDebugInfo)
      Importer.import(Path);
  }

  return llvm::Error::success();
}

void ImportBinaryPipe::print(const pipeline::Context &Ctx,
                             llvm::raw_ostream &OS,
                             llvm::ArrayRef<std::string> Names) const {
  OS << *revng::ResourceFinder.findFile("bin/revng");
  OS << " model import binary " << Names[0] << " -o model.yml";
}

static pipeline::RegisterPipe<ImportBinaryPipe> E;
