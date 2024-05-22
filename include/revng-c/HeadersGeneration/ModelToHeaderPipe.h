#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

inline constexpr char ModelHeaderFileContainerMIMEType[] = "text/x.c+ptml";
inline constexpr char ModelHeaderFileContainerSuffix[] = ".h";
inline constexpr char ModelHeaderFileContainerName[] = "model-header";
using ModelHeaderFileContainer = FileContainer<&kinds::ModelHeader,
                                               ModelHeaderFileContainerName,
                                               ModelHeaderFileContainerMIMEType,
                                               ModelHeaderFileContainerSuffix>;

class ModelToHeader {
public:
  static constexpr auto Name = "model-to-header";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    Contract C1(Binary, 0, ModelHeader, 1, InputPreservation::Preserve);
    return { ContractGroup({ C1 }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           const BinaryFileContainer &BinaryFile,
           ModelHeaderFileContainer &HeaderFile);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
