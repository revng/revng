/// \file AssemblyPlain.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/HTML.h"
#include "revng/Yield/Plain.h"

#include "Common.h"

int main(int ArgC, char *ArgV[]) {
  auto [IR, Model, Binary, Result, _] = parseCommandLineOptions(ArgC, ArgV);

  // Define the helper object to store the disassembly pipeline.
  // This allows it to only be created once.
  DissassemblyHelper Helper;

  for (const auto &LLVMFunction : FunctionTags::Isolated.functions(&IR)) {
    auto Metadata = extractFunctionMetadata(&LLVMFunction);
    auto ModelFunctionIterator = Model->Functions.find(Metadata->Entry);
    revng_assert(ModelFunctionIterator != Model->Functions.end());
    const auto &Function = *ModelFunctionIterator;

    auto Disassembled = Helper.disassemble(Function, *Metadata, Binary);

    auto Plain = yield::plain::assembly(Disassembled, *Metadata, *Model);
    Result.insert_or_assign(Function.Entry, std::move(Plain));
  }

  llvm::Error Error = Result.serialize(llvm::outs());
  revng_assert(Error);
  return EXIT_SUCCESS;
}
