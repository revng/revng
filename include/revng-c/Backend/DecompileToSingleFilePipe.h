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
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringBufferContainer.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/CDecompilationPipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char DecompiledMIMEType[] = "text/x.c+ptml";
inline constexpr char DecompiledSuffix[] = ".c";
inline constexpr char DecompiledName[] = "DecompiledCCode";
using DecompiledFileContainer = StringBufferContainer<&kinds::DecompiledToC,
                                                      DecompiledName,
                                                      DecompiledMIMEType,
                                                      DecompiledSuffix>;

class DecompileToSingleFile {
public:
  static constexpr auto Name = "DecompileToSingleFile";

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
           const DecompiledCCodeInYAMLStringMap &DecompiledFunctionsContainer,
           DecompiledFileContainer &OutCFile);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
