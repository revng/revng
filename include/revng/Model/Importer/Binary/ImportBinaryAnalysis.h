#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

class ImportBinaryAnalysis {
public:
  static constexpr auto Name = "ImportBinary";
  static constexpr auto Doc = "Import into the model all the available "
                              "metadata in one of the supported binary formats "
                              "(ELF, PE/COFF, Mach-O). Typically, this "
                              "includes loading directives, static and dynamic "
                              "symbols and debug information.";

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::Binary }
  };

public:
  llvm::Error
  run(pipeline::Context &Context, const BinaryFileContainer &SourceBinary);
};

} // namespace revng::pipes
