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

static model::BinaryReference makeReference(model::Binary &Binary,
                                            size_t Index) {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  TupleTreePath BinaryPath;
  BinaryPath.push_back(static_cast<size_t>(Fields::Binaries));
  BinaryPath.push_back(Index);
  return model::BinaryReference{ &Binary, BinaryPath };
}

llvm::Error ImportBinaryAnalysis::run(pipeline::ExecutionContext &Context,
                                      const BinaryFileContainer &SourceBinary) {
  if (not SourceBinary.exists())
    return llvm::Error::success();

  TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

  const ImporterOptions &Options = importerOptions();
  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(*SourceBinary.path(),
                                                        false,
                                                        false);
  if (not MaybeBuffer)
    return llvm::errorCodeToError(MaybeBuffer.getError());

  llvm::Task T(2, "Import binary");
  T.advance("Import main binary", true);

  model::BinaryReference Reference;
  if (Model->Binaries().size() > 0)
    Reference = makeReference(*Model.get(), 0);

  if (llvm::Error Error = importBinary(Model,
                                       **MaybeBuffer,
                                       Options,
                                       Reference))
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

namespace revng::pypeline::analyses {

llvm::Error ParseBinaryAnalysis::run(Model &Model,
                                     const Request &Incoming,
                                     llvm::StringRef Configuration,
                                     const BinariesContainer &Binaries) {
  const ImporterOptions &Options = importerOptions();

  llvm::Task T(2, "Import binary");
  T.advance("Import main binary", true);

  for (size_t I = 0; I < Binaries.size(); I++) {
    llvm::ArrayRef<char> Ref = Binaries.getFile(I);
    auto Buffer = llvm::MemoryBuffer::getMemBuffer({ Ref.begin(), Ref.size() },
                                                   "",
                                                   false);

    model::BinaryReference Reference = makeReference(*Model.get().get(), I);
    if (llvm::Error Error = importBinary(Model.get(),
                                         *Buffer,
                                         Options,
                                         Reference))
      return Error;
  }

  T.advance("Import additional debug info", true);
  if (!Options.AdditionalDebugInfoPaths.empty()) {
    DwarfImporter Importer(Model.get());
    llvm::Task T2(Options.AdditionalDebugInfoPaths.size(),
                  "Import additional debug info");
    for (const std::string &Path : Options.AdditionalDebugInfoPaths) {
      T2.advance(Path, true);
      Importer.import(Path, Options);
    }
  }

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
