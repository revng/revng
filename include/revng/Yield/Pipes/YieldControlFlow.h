#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pipes {

inline constexpr char FunctionAssemblyYamlMIMEType[] = "text/x.yaml";
inline constexpr char FunctionAssemblyYamlName[] = "function-assembly-internal";
inline constexpr char FunctionAssemblyYamlExtension[] = ".yml";
using FunctionAssemblyStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyInternal,
  FunctionAssemblyYamlName,
  FunctionAssemblyYamlMIMEType,
  FunctionAssemblyYamlExtension>;

inline constexpr char FunctionAssemblyPTMLMIMEType[] = "text/x.asm+ptml+tar+gz";
inline constexpr char FunctionAssemblyPTMLName[] = "function-assembly-ptml";
inline constexpr char FunctionAssemblyPTMLExtension[] = ".asm.tar.gz";

using FunctionAssemblyPTMLStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyPTML,
  FunctionAssemblyPTMLName,
  FunctionAssemblyPTMLMIMEType,
  FunctionAssemblyPTMLExtension>;

inline constexpr char FunctionControlFlowMIMEType[] = "image/svg";
#define NAME "function-control-flow-graph-svg"
inline constexpr char FunctionControlFlowName[] = NAME;
#undef NAME
inline constexpr char FunctionControlFlowExtension[] = ".svg";
using FunctionControlFlowStringMap = FunctionStringMap<
  &kinds::FunctionControlFlowGraphSVG,
  FunctionControlFlowName,
  FunctionControlFlowMIMEType,
  FunctionControlFlowExtension>;

class YieldControlFlow {
public:
  static constexpr const auto Name = "yield-cfg";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::FunctionAssemblyInternal,
                                     0,
                                     kinds::FunctionControlFlowGraphSVG,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const FunctionAssemblyStringMap &Input,
           FunctionControlFlowStringMap &Output);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
