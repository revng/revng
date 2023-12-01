#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/IR/IRBuilder.h"

#include "revng/Support/MetaAddress.h"

namespace efa {

class CallHandler {
public:
  virtual ~CallHandler() {}

  /// \note Implementers should not emit a terminator
  virtual void
  handleCall(MetaAddress CallerBlock,
             llvm::IRBuilder<> &Builder,
             MetaAddress Callee,
             const std::set<llvm::GlobalVariable *> &ClobberedRegisters,
             const std::optional<int64_t> &MaybeFSO,
             bool IsNoReturn,
             bool IsTailCall,
             llvm::Value *SymbolNamePointer) = 0;

  /// \note Implementers are responsible for terminator emissions
  virtual void handlePostNoReturn(llvm::IRBuilder<> &Builder) = 0;

  /// \note Implementers should not emit a terminator
  virtual void
  handleIndirectJump(llvm::IRBuilder<> &Builder,
                     MetaAddress Block,
                     const std::set<llvm::GlobalVariable *> &ClobberedRegisters,
                     llvm::Value *SymbolNamePointer) = 0;
};

} // namespace efa
