#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"

namespace revng::pipes {

class ImportBinaryPipe {
public:
  static constexpr auto Name = "ImportBinary";

public:
  std::array<pipeline::ContractGroup, 1> getContract() const { return {}; }

public:
  void run(pipeline::Context &Context, const FileContainer &SourceBinary);
};

} // namespace revng::pipes
