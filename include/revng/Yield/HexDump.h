#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

using HexDumpContainer = BytesContainer<"HexDumpContainer",
                                        "text/x.hexdump+ptml">;

namespace piperuns {

class HexDump {
public:
  static constexpr llvm::StringRef Name = "HexDump";
  using Arguments = TypeList<
    PipeArgument<"Binary", "The binaries to create the hexdump out of">,
    PipeArgument<"Module", "The LLVM Module(s) with lifted functions">,
    PipeArgument<"CFG", "The per-function CFG data">,
    PipeArgument<"Output", "The hexdump of the input binaries", Access::Write>>;

public:
  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  static void run(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  const BinariesContainer &BinaryContainer,
                  const LLVMFunctionContainer &ModuleContainer,
                  const CFGMap &CFG,
                  HexDumpContainer &Output);
};

} // namespace piperuns

} // namespace revng::pypeline
