#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

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

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
