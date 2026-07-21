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
  ptml::Tagging Tags = Configuration.DisableMarkup ? ptml::Tagging::Disabled :
                                                     ptml::Tagging::Enabled;

  std::string Header;
  {
    ptml::CTokenEmitter Tokens(Tags);

    Tokens.emitIncludeDirective("types-and-globals.h",
                                "",
                                ptml::CTokenEmitter::IncludeMode::Quote);
    Tokens.emitIncludeDirective("helpers.h",
                                "",
                                ptml::CTokenEmitter::IncludeMode::Quote);
    Tokens.emitNewline();

    Header = Tokens.extract();
  }

  // Wrap the includes and the function bodies (already emitted and reformatted
  // upstream) in a single root element, as a PTML document requires. The
  // wrapping tag is emitted by hand to avoid a whole CTokenEmitter just for it.
  std::unique_ptr<llvm::raw_pwrite_stream> Out = Output.getOStream(ObjectID());
  auto Mode = Tags == ptml::Tagging::Enabled ? ptml::EmissionMode::Tags :
                                               ptml::EmissionMode::PlainText;
  ptml::PTMLStreamEmitter Root(*Out, Mode);
  ptml::PTMLTagEmitter Document = Root.initializeOpenTag(ptml::tags::Div);
  Document.finalizeOpenTag();

  *Out << Header;
  for (const auto &Object : Input.objects()) {
    auto Buffer = Input.getMemoryBuffer(Object);
    *Out << Buffer->getBuffer() << "\n";
  }
}

} // namespace revng::pypeline::piperuns
