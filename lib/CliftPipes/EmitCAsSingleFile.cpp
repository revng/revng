//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/EmitCAsSingleFile.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"

namespace revng::pypeline::piperuns {

EmitCAsSingleFile::EmitCAsSingleFile(const class Model &Model,
                                     llvm::StringRef Configuration,
                                     llvm::StringRef DynamicConfig,
                                     const PTMLCFunctionContainer &Input,
                                     PTMLCContainer &Output) :
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

  Tokens.emitIncludeDirective("types-and-globals.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitIncludeDirective("helpers.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitNewline();

  ptml::StreamEmitter RawEmitter(*Out);
  for (const auto &Object : Input.objects()) {
    auto Buffer = Input.getMemoryBuffer(Object);
    RawEmitter.emit(Buffer->getBuffer().str() + "\n");
  }
}

} // namespace revng::pypeline::piperuns
