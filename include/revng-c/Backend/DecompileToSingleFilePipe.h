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
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringBufferContainer.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompilePipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char DecompiledMIMEType[] = "text/x.c+ptml";
inline constexpr char DecompiledSuffix[] = ".c";
inline constexpr char DecompiledName[] = "decompiled-c-code";
using DecompiledFileContainer = StringBufferContainer<&kinds::DecompiledToC,
                                                      DecompiledName,
                                                      DecompiledMIMEType,
                                                      DecompiledSuffix>;

class DecompileToSingleFile {
public:
  static constexpr auto Name = "decompile-to-single-file";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(Decompiled,
                                      0,
                                      DecompiledToC,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           const DecompileStringMap &DecompiledFunctionsContainer,
           DecompiledFileContainer &OutCFile);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
