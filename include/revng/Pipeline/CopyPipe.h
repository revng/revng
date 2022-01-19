#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"

namespace pipeline {

/// A pipe that copies items from a container to another
template<typename Source, typename Destination = Source>
class CopyPipe {
public:
  static constexpr auto Name = "CopyPipe";

private:
  Kind *K;

public:
  CopyPipe(Kind &K) : K(&K) {}

public:
  std::array<ContractGroup, 1> getContract() const {
    return { ContractGroup(*K, Exactness::Exact, 0, *K, 1) };
  }

public:
  void run(const Context &, const Source &S, Destination &T) { T = S; }
};

} // namespace pipeline
