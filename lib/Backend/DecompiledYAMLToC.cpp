//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng-c/Backend/DecompiledYAMLToC.h"

void printSingleCFile(llvm::raw_ostream &Out,
                      const revng::pipes::FunctionStringMap &Functions,
                      const std::set<MetaAddress> &Targets) {
  // Print headers
  Out << "#include \"revng-model-declarations.h\"\n"
      << "#include \"revng-qemu-helpers-declarations.h\"\n\n";

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
