#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/PassRegistry.h"

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

/// Pipe that allows running non-model-aware MLIR passes on a
/// CliftFunctionContainer. Just like the PureLLVMPassesPipe, the passes to be
/// run are set by the static configuration, e.g.:
/// ```yaml
/// passes:
///   - pass1
///   - pass2
/// ```
/// However MLIR passes can also be initialized with configuration, so in that
/// case you have to specify the parts separately:
/// ```yaml
/// passes:
///   - name: pass1
///     options: option1=value1 option2=value2
///   - pass2
/// ```
class PureMLIRPassesPipe {
public:
  static constexpr llvm::StringRef Name = "pure-mlir-passes-pipe";
  using Arguments = TypeList<
    PipeArgument<"Module", "MLIR Modules to apply the MLIR passes to">>;

private:
  std::string TaskName;

  struct PassInfo {
    const mlir::PassInfo *PassInfo;
    std::string Options;
  };
  std::vector<PassInfo> PassInfos;

public:
  struct Configuration;
  const std::string StaticConfiguration;
  PureMLIRPassesPipe(llvm::StringRef StaticConfiguration);

public:
  PipeOutput run(const Model &TheModel,
                 const revng::pypeline::Request &Incoming,
                 const revng::pypeline::Request &Outgoing,
                 llvm::StringRef Configuration,
                 CliftFunctionContainer &Container);
};

} // namespace revng::pypeline::pipes
