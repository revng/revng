//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Backend/EmitCAsSingleFile.h"
#include "revng/TypeNames/ModelCBuilder.h"

using namespace revng::pipes;

void printSingleCFile(ptml::ModelCBuilder &B,
                      const DecompileStringMap &Functions,
                      const std::set<MetaAddress> &Targets) {
  auto Scope = B.getScopeTag(ptml::tags::Div);
  // Print headers
  B.append(B.getIncludeQuote("types-and-globals.h")
           + B.getIncludeQuote("helpers.h") + "\n");

  if (Targets.empty()) {
    // If Targets is empty print all the Functions' bodies
    for (const auto &[MetaAddress, CFunction] : Functions)
      B.append(CFunction + '\n');
  } else {
    // Otherwise only print the bodies of the Targets
    auto End = Functions.end();
    for (const auto &MetaAddress : Targets)
      if (auto It = Functions.find(MetaAddress); It != End)
        B.append(It->second + '\n');
  }
}

namespace revng::pypeline::piperuns {

EmitCAsSingleFile::EmitCAsSingleFile(const class Model &Model,
                                     llvm::StringRef Config,
                                     llvm::StringRef DynamicConfig,
                                     const PTMLCFunctionBytesContainer &Input,
                                     PTMLCBytesContainer &Output) :
  Binary(*Model.get().get()), Input(Input), Output(Output) {
}

void EmitCAsSingleFile::run() {
  auto Out = Output.getOStream(ObjectID());
  ptml::ModelCBuilder B(*Out,
                        Binary,
                        /* EnableTaglessMode = */ false,
                        // Disable stack frame inlining because enabling it
                        // could break the property that we emit syntactically
                        // valid C code, due to the stack frame type definition
                        // being duplicated in the global header and
                        // in the function's body. In the single file artifact
                        // recompilability is still important.
                        { .EnableStackFrameInlining = false });

  auto Scope = B.getScopeTag(ptml::tags::Div);
  // Print headers
  B.append(B.getIncludeQuote("types-and-globals.h")
           + B.getIncludeQuote("helpers.h") + "\n");

  for (const auto &Object : Input.objects()) {
    auto Buffer = Input.getMemoryBuffer(Object);
    B.append(Buffer->getBuffer().str() + "\n");
  }
}

} // namespace revng::pypeline::piperuns
