#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/EarlyFunctionAnalysis/AnalyzeRegisterUsage.h"
#include "revng/Support/MetaAddress.h"

namespace revng {
class IRBuilder;
} // namespace revng

namespace efa {

class CallHandler {
public:
  virtual ~CallHandler() {}

  /// \note Implementers should not emit a terminator
  virtual void handleCall(MetaAddress CallerBlock,
                          revng::IRBuilder &Builder,
                          MetaAddress Callee,
                          const CSVSet &ClobberedRegisters,
                          const std::optional<int64_t> &MaybeFSO,
                          bool IsNoReturn,
                          bool IsTailCall,
                          llvm::Value *SymbolNamePointer) = 0;

  /// \note Implementers are responsible for terminator emissions
  virtual void handlePostNoReturn(revng::IRBuilder &Builder,
                                  const llvm::DebugLoc &DbgLocation) = 0;

  /// \note Implementers should not emit a terminator
  virtual void handleIndirectJump(revng::IRBuilder &Builder,
                                  MetaAddress Block,
                                  const CSVSet &ClobberedRegisters,
                                  llvm::Value *SymbolNamePointer) = 0;
};

} // namespace efa
