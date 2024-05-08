#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
  static constexpr auto Name = "import-binary";

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::Binary }
  };

public:
  llvm::Error run(pipeline::ExecutionContext &Context,
                  const BinaryFileContainer &SourceBinary);
};

} // namespace revng::pipes
