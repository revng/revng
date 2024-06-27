//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Support/GzipTarFile.h"
#include "revng/Support/ResourceFinder.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/Backend/DecompileToDirectoryPipe.h"
#include "revng-c/Backend/DecompileToSingleFile.h"
#include "revng-c/HeadersGeneration/HelpersToHeader.h"
#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/Support/PTMLC.h"

namespace revng::pipes {

using namespace pipeline;

static RegisterDefaultConstructibleContainer<RecompilableArchiveContainer> Reg;

void DecompileToDirectory::run(const pipeline::ExecutionContext &Ctx,
                               pipeline::LLVMContainer &IRContainer,
                               const revng::pipes::CFGMap &CFGMap,
                               RecompilableArchiveContainer &OutTarFile) {

  std::error_code EC;
  llvm::raw_fd_ostream OutputStream{ OutTarFile.getOrCreatePath(), EC };
  if (EC)
    revng_abort(EC.message().c_str());

  GzipTarWriter TarWriter{ OutputStream };

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(Ctx);
  {

    ControlFlowGraphCache Cache{ CFGMap };

    std::string DecompiledC;

    llvm::raw_string_ostream Out{ DecompiledC };

    DecompileStringMap DecompiledFunctions("tmp");
    decompile(Cache,
              Module,
              Model,
              DecompiledFunctions,
              /* GeneratePlainC = */ true);

    ptml::PTMLCBuilder B{ /* GeneratePlainC = */ true };
    printSingleCFile(Out, B, DecompiledFunctions, {} /* Targets */);

    Out.flush();

    TarWriter.append("decompiled/functions.c",
                     llvm::ArrayRef{ DecompiledC.data(),
                                     DecompiledC.length() });
  }

  {
    std::string ModelHeader;
    llvm::raw_string_ostream Out{ ModelHeader };

    dumpModelToHeader(Model,
                      Out,
                      ModelToHeaderOptions{ .GeneratePlainC = true });

    Out.flush();

    TarWriter.append("decompiled/types-and-globals.h",
                     llvm::ArrayRef{ ModelHeader.data(),
                                     ModelHeader.length() });
  }

  {
    std::string HelpersHeader;
    llvm::raw_string_ostream Out{ HelpersHeader };

    dumpHelpersToHeader(Module,
                        Out,
                        /* GeneratePlainC = */ true);

    Out.flush();

    TarWriter.append("decompiled/helpers.h",
                     llvm::ArrayRef{ HelpersHeader.data(),
                                     HelpersHeader.length() });
  }

  {
    auto Path = revng::ResourceFinder.findFile("share/revng-c/include/"
                                               "attributes.h");

    if (not Path or Path->empty())
      revng_abort("can't find attributes.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/attributes.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  {
    auto Path = revng::ResourceFinder.findFile("share/revng-c/include/"
                                               "primitive-types.h");

    if (not Path or Path->empty())
      revng_abort("can't find primitive-types.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/primitive-types.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  TarWriter.close();
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompileToDirectory> Y;
