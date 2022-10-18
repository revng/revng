#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char FunctionAssemblyYamlMIMEType[] = "text/x.yaml";
inline constexpr char FunctionAssemblyYamlName[] = "FunctionAssemblyInternal";
using FunctionAssemblyStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyInternal,
  FunctionAssemblyYamlName,
  FunctionAssemblyYamlMIMEType>;

inline constexpr char FunctionAssemblyPTMLMIMEType[] = "text/x.asm+ptml+yaml";
inline constexpr char FunctionAssemblyPTMLName[] = "FunctionAssemblyPTML";

using FunctionAssemblyPTMLStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyPTML,
  FunctionAssemblyPTMLName,
  FunctionAssemblyPTMLMIMEType>;

inline constexpr char FunctionControlFlowMIMEType[] = "image/svg";
inline constexpr char FunctionControlFlowName[] = "FunctionControlFlowGraphSVG";
using FunctionControlFlowStringMap = FunctionStringMap<
  &kinds::FunctionControlFlowGraphSVG,
  FunctionControlFlowName,
  FunctionControlFlowMIMEType>;

class YieldControlFlow {
public:
  static constexpr const auto Name = "YieldCFG";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::FunctionAssemblyInternal,
                                     0,
                                     kinds::FunctionControlFlowGraphSVG,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::Context &Context,
           const FunctionAssemblyStringMap &Input,
           FunctionControlFlowStringMap &Output);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
