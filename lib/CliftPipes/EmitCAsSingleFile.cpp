//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/EmitCAsSingleFile.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Containers.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringBufferContainer.h"

static void printIncludes(ptml::CTokenEmitter &Tokens) {

  Tokens.emitIncludeDirective("types-and-globals.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitIncludeDirective("helpers.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitNewline();
}

namespace revng::pipes {

inline constexpr char DecompiledMIMEType[] = "text/x.c+ptml";
inline constexpr char DecompiledSuffix[] = ".c";
inline constexpr char DecompiledName[] = "decompiled-c-code";
using DecompiledFileContainer = StringBufferContainer<&kinds::DecompiledToC,
                                                      DecompiledName,
                                                      DecompiledMIMEType,
                                                      DecompiledSuffix>;

static pipeline::RegisterDefaultConstructibleContainer<DecompiledFileContainer>
  Reg;

} // namespace revng::pipes

class EmitCAsSingleFile {
public:
  static constexpr auto Name = "emit-c-as-single-file";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(Decompiled,
                                      0,
                                      DecompiledToC,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::DecompileStringMap &DecompiledFunctions,
           revng::pipes::DecompiledFileContainer &OutCFile) {
    {
      llvm::raw_string_ostream Out = OutCFile.asStream();
      static constexpr ptml::Tagging Tags = ptml::Tagging::Enabled;

      ptml::CTokenEmitter Tokens(Out, Tags);
      printIncludes(Tokens);

      ptml::StreamEmitter RawEmitter(Out);
      for (const auto &[MetaAddress, CFunction] : DecompiledFunctions)
        RawEmitter.emit(CFunction + "\n");
    }

    EC.commitUniqueTarget(OutCFile);
  }
};

static pipeline::RegisterPipe<EmitCAsSingleFile> Y;

namespace revng::pypeline::piperuns {

EmitCAsSingleFile::EmitCAsSingleFile(const class Model &Model,
                                     llvm::StringRef Configuration,
                                     llvm::StringRef DynamicConfig,
                                     const PTMLCFunctionBytesContainer &Input,
                                     PTMLCBytesContainer &Output) :
  Binary(*Model.get().get()),
  Input(Input),
  Output(Output),
  Configuration(parseCEmissionPipeConfiguration(Configuration)) {
}

void EmitCAsSingleFile::run() {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = Output.getOStream(ObjectID());

  ptml::CTokenEmitter Tokens(*Out,
                             Configuration.DisableMarkup ?
                               ptml::Tagging::Disabled :
                               ptml::Tagging::Enabled);
  printIncludes(Tokens);

  ptml::StreamEmitter RawEmitter(*Out);
  for (const auto &Object : Input.objects()) {
    auto Buffer = Input.getMemoryBuffer(Object);
    RawEmitter.emit(Buffer->getBuffer().str() + "\n");
  }
}

} // namespace revng::pypeline::piperuns
