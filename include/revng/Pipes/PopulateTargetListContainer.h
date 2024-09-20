#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/TargetListContainer.h"

namespace revng::pipes {

/// This pipe works in tandem with a TargetListContainer (see
/// TargetListContainer.h). Its purpose is to be used to fill the container with
/// all targets that are returned from Kind::appendAllTargets.
template<typename ContainerType, const char *TheName>
class PopulateTargetListContainer {
public:
  static constexpr auto Name = TheName;

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(Binary,
                                      0,
                                      *ContainerType::Kind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &EC,
           const BinaryFileContainer &,
           ContainerType &Container) {
    Container.fill(EC.getContext());
  }
};

} // namespace revng::pipes
