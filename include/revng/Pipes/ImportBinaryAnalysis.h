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

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::pipes::Binary }
  };

public:
  void run(pipeline::Context &Context, const FileContainer &SourceBinary);
};

} // namespace revng::pipes
