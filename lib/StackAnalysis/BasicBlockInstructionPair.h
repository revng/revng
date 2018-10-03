#ifndef BASICBLOCKINSTRUCTIONPAIR_H
#define BASICBLOCKINSTRUCTIONPAIR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

namespace llvm {
class BasicBlock;
class Instruction;
} // namespace llvm

namespace StackAnalysis {

/// \brief std::pair<BasicBlock *, Instruction *> on steroids
class BasicBlockInstructionPair {
public:
  BasicBlockInstructionPair() : BB(nullptr), I(nullptr) {}
  BasicBlockInstructionPair(llvm::BasicBlock *BB, llvm::Instruction *I) :
    BB(BB),
    I(I) {}

  bool isNull() const { return I == nullptr || BB == nullptr; }

  bool operator<(const BasicBlockInstructionPair &Other) const {
    return std::tie(BB, I) < std::tie(Other.BB, Other.I);
  }

  bool operator==(const BasicBlockInstructionPair &Other) const {
    return std::tie(BB, I) == std::tie(Other.BB, Other.I);
  }

  bool operator!=(const BasicBlockInstructionPair &Other) const {
    return !(*this == Other);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << getName(BB) << ":" << getName(I);
  }

protected:
  llvm::BasicBlock *BB;
  llvm::Instruction *I;
};

/// \brief Represent a call site within a function
///
/// \note caller() doesn't return callInstruction()->getParent(), but entry
///       basic block of the original function containing this call site.
class CallSite : public BasicBlockInstructionPair {
public:
  CallSite() : BasicBlockInstructionPair() {}
  CallSite(llvm::BasicBlock *BB, llvm::Instruction *I) :
    BasicBlockInstructionPair(BB, I) {}

  bool belongsTo(llvm::BasicBlock *OtherBB) const { return OtherBB == BB; }
  llvm::BasicBlock *caller() const { return BB; }
  llvm::Instruction *callInstruction() const { return I; }
};

class FunctionCall : public BasicBlockInstructionPair {
public:
  FunctionCall() : BasicBlockInstructionPair() {}
  FunctionCall(llvm::BasicBlock *BB, llvm::Instruction *I) :
    BasicBlockInstructionPair(BB, I) {}

  llvm::BasicBlock *callee() const { return BB; }
  llvm::Instruction *callInstruction() const { return I; }
};

/// \brief Represent a branch instruction within a function
///
/// \note caller() doesn't return callInstruction()->getParent(), but entry
///       basic block of the original function containing this branch.
class Branch : public BasicBlockInstructionPair {
public:
  Branch() : BasicBlockInstructionPair() {}
  Branch(llvm::BasicBlock *BB, llvm::Instruction *I) :
    BasicBlockInstructionPair(BB, I) {}

  bool belongsTo(llvm::BasicBlock *OtherBB) const { return OtherBB == BB; }
  llvm::Instruction *branch() const { return I; }
  llvm::BasicBlock *entry() const { return BB; }
};

inline void writeToLog(Logger<true> &This, const CallSite &Other, int) {
  Other.dump(This);
}

} // namespace StackAnalysis

#endif // BASICBLOCKINSTRUCTIONPAIR_H
