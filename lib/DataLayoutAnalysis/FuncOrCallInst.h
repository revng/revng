#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

namespace dla {

// Some APIs to make Function and CallInst homogeneous
const inline llvm::Type *getRetType(const llvm::Function *F) {
  return F->getReturnType();
}
const inline llvm::Type *getRetType(const llvm::CallInst *C) {
  return C->getType();
}

inline auto getArgs(const llvm::Function *F) {
  return F->args();
}
inline auto getArgs(const llvm::CallInst *C) {
  return C->args();
}

inline unsigned long arg_size(const llvm::Function *F) {
  return F->arg_size();
}
inline unsigned long arg_size(const llvm::CallInst *C) {
  return C->arg_size();
}

/// Wrapper around a variant between Function and CallInst
class FuncOrCallInst {
private:
  std::variant<const llvm::Function *, const llvm::CallInst *> Val;

public:
  FuncOrCallInst() : Val((llvm::Function *) nullptr){};
  FuncOrCallInst(const llvm::Function *F) : Val(F){};
  FuncOrCallInst(const llvm::CallInst *F) : Val(F){};
  FuncOrCallInst(const FuncOrCallInst &F) : Val(F.Val){};

  ~FuncOrCallInst() = default;
  FuncOrCallInst(FuncOrCallInst &&) = default;
  FuncOrCallInst &operator=(const FuncOrCallInst &) = default;
  FuncOrCallInst &operator=(FuncOrCallInst &&) = default;

  bool isNull() const;

  const llvm::Value *getVal() const;
  const llvm::Type *getRetType() const;

  /// Get the \a Idx-th argument
  const llvm::Value *getArg(unsigned Idx) const;
  unsigned long arg_size() const;
};

} // namespace dla
