#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
inline constexpr char FunctionAssemblyYamlName[] = "FunctionAssemblyInternal";
inline constexpr char FunctionAssemblyYamlExtension[] = ".yml";
using FunctionAssemblyStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyInternal,
  FunctionAssemblyYamlName,
  FunctionAssemblyYamlMIMEType,
  FunctionAssemblyYamlExtension>;

inline constexpr char FunctionAssemblyPTMLMIMEType[] = "text/x.asm+ptml+tar+gz";
inline constexpr char FunctionAssemblyPTMLName[] = "FunctionAssemblyPTML";
inline constexpr char FunctionAssemblyPTMLExtension[] = ".asm.tar.gz";

using FunctionAssemblyPTMLStringMap = FunctionStringMap<
  &kinds::FunctionAssemblyPTML,
  FunctionAssemblyPTMLName,
  FunctionAssemblyPTMLMIMEType,
  FunctionAssemblyPTMLExtension>;

inline constexpr char FunctionControlFlowMIMEType[] = "image/svg";
inline constexpr char FunctionControlFlowName[] = "FunctionControlFlowGraphSVG";
inline constexpr char FunctionControlFlowExtension[] = ".svg";
using FunctionControlFlowStringMap = FunctionStringMap<
  &kinds::FunctionControlFlowGraphSVG,
  FunctionControlFlowName,
  FunctionControlFlowMIMEType,
  FunctionControlFlowExtension>;

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
  void run(pipeline::ExecutionContext &Context,
           const FunctionAssemblyStringMap &Input,
           FunctionControlFlowStringMap &Output);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace revng::pipes
