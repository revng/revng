//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Backend/DecompileToSingleFile.h"
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
