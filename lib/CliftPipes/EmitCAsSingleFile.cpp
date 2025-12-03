//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompilePipe.h"
#include "revng/Backend/DecompileToSingleFilePipe.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

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
      ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Enabled);

      ptml::CTokenEmitter::Scope
        Scope = Tokens.enterScope(ptml::CTokenEmitter::ScopeKind::Basic, 0);

      // Print includes
      Tokens.emitIncludeDirective("types-and-globals.h",
                                  "",
                                  ptml::CTokenEmitter::IncludeMode::Quote);
      Tokens.emitIncludeDirective("helpers.h",
                                  "",
                                  ptml::CTokenEmitter::IncludeMode::Quote);
      Tokens.emitNewline();

      // Copy the functions one by one.
      for (const auto &[MetaAddress, CFunction] : DecompiledFunctions) {
        Tokens.emitRawContent(CFunction);
        Tokens.emitNewline();
      }
    }

    EC.commitUniqueTarget(OutCFile);
  }
};

static pipeline::RegisterPipe<EmitCAsSingleFile> Y;
