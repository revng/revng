//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng-c/Backend/DecompileToSingleFile.h"

using namespace revng::pipes;

void printSingleCFile(llvm::raw_ostream &Out,
                      ptml::CBuilder &B,
                      const DecompileStringMap &Functions,
                      const std::set<MetaAddress> &Targets) {
  auto Scope = B.getTag(ptml::tags::Div).scope(Out);
  // Print headers
  Out << B.getIncludeQuote("types-and-globals.h")
      << B.getIncludeQuote("helpers.h") << "\n";

  if (Targets.empty()) {
    // If Targets is empty print all the Functions' bodies
    for (const auto &[MetaAddress, CFunction] : Functions)
      Out << CFunction << '\n';
  } else {
    // Otherwise only print the bodies of the Targets
    auto End = Functions.end();
    for (const auto &MetaAddress : Targets)
      if (auto It = Functions.find(MetaAddress); It != End)
        Out << It->second << '\n';
  }
}
