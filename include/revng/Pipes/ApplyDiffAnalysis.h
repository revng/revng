#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>
#include <vector>

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Option.h"

namespace revng::pipes {

struct ApplyDiffAnalysis {
  static constexpr auto Name = "Apply";

  constexpr static std::tuple Options = { pipeline::Option("diff-path", "") };

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  ErrorList run(pipeline::Context &Ctx, std::string DiffLocation);
};
} // namespace revng::pipes
