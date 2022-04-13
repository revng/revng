#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

class ModelToHeaderPipe {
public:
  static constexpr auto Name = "ModelToHeader";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::pipes;

    return { ContractGroup({ Contract(Binary,
                                      Exactness::Exact,
                                      0,
                                      ModelHeader,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::Context &Ctx,
           const FileContainer &BinaryFile,
           FileContainer &HeaderFile);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
