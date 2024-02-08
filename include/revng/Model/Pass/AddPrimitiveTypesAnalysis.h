#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Pass/AddPrimitiveTypes.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

class AddPrimitiveTypesAnalysis {
public:
  static constexpr const auto Name = "add-primitive-types";

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

public:
  void run(pipeline::ExecutionContext &Context);
};

} // namespace revng::pipes
