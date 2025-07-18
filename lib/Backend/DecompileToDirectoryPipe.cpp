//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Backend/DecompileFunction.h"
#include "revng/Backend/DecompileToDirectoryPipe.h"
#include "revng/Backend/DecompileToSingleFile.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/PTML/CBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Support/GzipTarFile.h"
#include "revng/Support/ResourceFinder.h"

namespace revng::pipes {

using namespace pipeline;

static RegisterDefaultConstructibleContainer<RecompilableArchiveContainer> Reg;

void DecompileToDirectory::run(pipeline::ExecutionContext &EC,
                               pipeline::LLVMContainer &IRContainer,
                               const revng::pipes::CFGMap &CFGMap,
                               RecompilableArchiveContainer &OutTarFile) {

  std::error_code ErrorCode;
  llvm::raw_fd_ostream OutputStream{ OutTarFile.getOrCreatePath(), ErrorCode };
  if (ErrorCode)
    revng_abort(ErrorCode.message().c_str());

  GzipTarWriter TarWriter{ OutputStream };

  llvm::Module &Module = IRContainer.getModule();
  const model::Binary &Model = *getModelFromContext(EC);

  namespace options = revng::options;
  ptml::ModelCBuilder B(llvm::nulls(),
                        Model,
                        /* EnableTaglessMode = */ true,
                        // Disable stack frame inlining because enabling it
                        // could break the property that we emit syntactically
                        // valid C code, due to the stack frame type definition
                        // being duplicated in the global header and
                        // in the function's body. In this artifact, where all
                        // the decompiled functions are put in a single .c file,
                        // recompilability is still important.
                        { .EnableStackFrameInlining = false });

  {
    ControlFlowGraphCache Cache{ CFGMap };
    DecompileStringMap DecompiledFunctions("tmp");
    for (pipeline::Target &Target : CFGMap.enumerate()) {
      auto Entry = MetaAddress::fromString(Target.getPathComponents()[0]);
      const model::Function &Function = Model.Functions().at(Entry);
      auto *F = Module.getFunction(B.NameBuilder.llvmName(Function));
      std::string CCode = decompile(Cache, *F, Model, B);
      DecompiledFunctions.insert_or_assign(Entry, std::move(CCode));
    }

    std::string DecompiledC;
    llvm::raw_string_ostream Out{ DecompiledC };
    B.setOutputStream(Out);
    printSingleCFile(B, DecompiledFunctions, {} /* Targets */);

    Out.flush();

    TarWriter.append("decompiled/functions.c",
                     llvm::ArrayRef{ DecompiledC.data(),
                                     DecompiledC.length() });
  }

  {
    std::string ModelHeader;
    llvm::raw_string_ostream Out{ ModelHeader };
    B.setOutputStream(Out);

    ptml::HeaderBuilder HB = B;
    HB.printModelHeader();

    Out.flush();

    TarWriter.append("decompiled/types-and-globals.h",
                     llvm::ArrayRef{ ModelHeader.data(),
                                     ModelHeader.length() });
  }

  {
    std::string HelpersHeader;
    llvm::raw_string_ostream Out{ HelpersHeader };
    B.setOutputStream(Out);

    ptml::HeaderBuilder HB = B;
    HB.printHelpersHeader(Module);

    Out.flush();

    TarWriter.append("decompiled/helpers.h",
                     llvm::ArrayRef{ HelpersHeader.data(),
                                     HelpersHeader.length() });
  }

  {
    auto Path = revng::ResourceFinder.findFile("share/revng/include/"
                                               "attributes.h");

    if (not Path or Path->empty())
      revng_abort("can't find attributes.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/attributes.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  {
    auto Path = revng::ResourceFinder.findFile("share/revng/include/"
                                               "primitive-types.h");

    if (not Path or Path->empty())
      revng_abort("can't find primitive-types.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/primitive-types.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  TarWriter.close();

  EC.commitUniqueTarget(OutTarFile);
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::DecompileToDirectory> Y;
