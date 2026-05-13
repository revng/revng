#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

class HexDumpContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "HexDumpContainer";
  static constexpr llvm::StringRef MimeType = "text/x.hexdump+ptml";
};

namespace piperuns {

class HexDump {
private:
  const model::Binary &Binary;
  const BinariesContainer &BinaryContainer;
  const LLVMFunctionContainer &ModuleContainer;
  const CFGMap &CFG;
  HexDumpContainer &Output;

public:
  static constexpr llvm::StringRef Name = "hex-dump";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer,
                    "Binary",
                    "The binaries to create the hexdump out of">,
    PipeRunArgument<const LLVMFunctionContainer,
                    "Module",
                    "The LLVM Module(s) with lifted functions">,
    PipeRunArgument<const CFGMap, "CFG", "The per-function CFG data">,
    PipeRunArgument<HexDumpContainer,
                    "Output",
                    "The hexdump of the input binaries",
                    Access::Write>>;

public:
  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  HexDump(const class Model &Model,
          llvm::StringRef Config,
          llvm::StringRef DynamicConfig,
          const BinariesContainer &BinaryContainer,
          const LLVMFunctionContainer &ModuleContainer,
          const CFGMap &CFG,
          HexDumpContainer &Output);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
