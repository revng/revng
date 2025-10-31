#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <set>
#include <sstream>
#include <type_traits>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Interval.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/MetaAddress.h"

class ProgramCounterHandler;

extern void dumpUsers(llvm::Value *V) debug_function;

namespace NewPCArguments {
enum {
  InstructionID,
  InstructionSize,
  IsJumpTarget,
  DissassembledInstruction,
  FirstLocalVariable
};
} // namespace NewPCArguments

inline llvm::Function *getCalledFunction(const llvm::CallBase *Call) {
  using namespace llvm;
  return dyn_cast<Function>(Call->getCalledOperand());
}

/// Given \p V, checks if there are uses left and then calls eraseFromParent.
/// In case of leftover uses, they are pretty printed.
///
/// \note The remaining use check is not performed on llvm::Functions since they
/// might have internal blockaddress self-references.
inline void eraseFromParent(llvm::Value *V) {
  using namespace llvm;

  if (not isa<Function>(V) and not V->use_empty()) {
    dbg << "Can't erase a Value still having uses.\n";
    dbg << "Value:\n  ";
    V->dump();
    dbg << "Users:\n";
    dumpUsers(V);
    revng_abort();
  } else {
    if (auto *I = dyn_cast<Instruction>(V))
      I->eraseFromParent();
    else if (auto *BB = dyn_cast<BasicBlock>(V))
      BB->eraseFromParent();
    else if (auto *G = dyn_cast<GlobalValue>(V))
      G->eraseFromParent();
    else
      revng_abort();
  }
}

constexpr const char *FunctionEntryMDName = "revng.function.entry";
constexpr const char *JTReasonMDName = "revng.jt.reasons";
constexpr const char *ControlFlowGraphMDName = "revng.function.metadata";
constexpr const char *ExplicitParenthesesMDName = "revng.explicit_parentheses";

template<typename T>
inline bool contains(T Range, typename T::value_type V) {
  return std::find(std::begin(Range), std::end(Range), V) != std::end(Range);
}

template<class T>
inline void freeContainer(T &Container) {
  T Empty;
  Empty.swap(Container);
}

/// Helper function to destroy an unconditional branch and, in case, the target
/// basic block, if it doesn't have any predecessors left.
inline void purgeBranch(llvm::BasicBlock::iterator I) {
  auto *DeadBranch = llvm::dyn_cast<llvm::BranchInst>(I);
  // We allow only a branch and nothing else
  revng_assert(DeadBranch != nullptr && ++I == DeadBranch->getParent()->end());

  std::set<llvm::BasicBlock *> Successors;
  for (unsigned C = 0; C < DeadBranch->getNumSuccessors(); C++)
    Successors.insert(DeadBranch->getSuccessor(C));

  // Destroy the dead branch
  eraseFromParent(DeadBranch);

  // Check if someone else was jumping there and then destroy
  for (llvm::BasicBlock *BB : Successors)
    if (BB->empty() && llvm::pred_empty(BB))
      eraseFromParent(BB);
}

inline llvm::ConstantInt *getConstValue(llvm::Constant *C,
                                        const llvm::DataLayout &DL) {
  while (auto *Expr = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    C = ConstantFoldConstant(Expr, DL);

    if (Expr->isCast())
      C = Expr->getOperand(0);
  }

  if (llvm::isa<llvm::ConstantPointerNull>(C)) {
    auto *Ptr = llvm::IntegerType::get(C->getType()->getContext(),
                                       DL.getPointerSizeInBits());
    return llvm::ConstantInt::get(Ptr, 0);
  }

  auto *Integer = llvm::cast<llvm::ConstantInt>(C);
  return Integer;
}

inline uint64_t getSExtValue(llvm::Constant *C, const llvm::DataLayout &DL) {
  return getConstValue(C, DL)->getSExtValue();
}

inline uint64_t getZExtValue(llvm::Constant *C, const llvm::DataLayout &DL) {
  return getConstValue(C, DL)->getZExtValue();
}

inline uint64_t
getExtValue(llvm::Constant *C, bool Sign, const llvm::DataLayout &DL) {
  if (Sign)
    return getSExtValue(C, DL);
  else
    return getZExtValue(C, DL);
}

inline uint64_t getLimitedValue(const llvm::Value *V) {
  return llvm::cast<llvm::ConstantInt>(V)->getLimitedValue();
}

inline uint64_t getSignedLimitedValue(const llvm::Value *V) {
  return llvm::cast<llvm::ConstantInt>(V)->getSExtValue();
}

template<typename T, unsigned I>
inline bool findOperand(llvm::Value *Op, T &Result) {
  return false;
}

template<typename T, unsigned I, typename Head, typename... Tail>
inline bool findOperand(llvm::Value *Op, T &Result) {
  using VT = typename std::remove_pointer<Head>::type;
  if (auto *Casted = llvm::dyn_cast<VT>(Op)) {
    std::get<I>(Result) = Casted;
    return true;
  } else {
    return findOperand<T, I + 1, Tail...>(Op, Result);
  }
}

/// Return a tuple of \p V's operands of the requested types
/// \return a tuple with the operands of the specified type in the specified
///         order, or, if not possible, a nullptr tuple.
template<typename... T>
inline std::tuple<T...> operandsByType(llvm::User *V) {
  std::tuple<T...> Result;
  unsigned OpCount = V->getNumOperands();
  revng_assert(OpCount == sizeof...(T));

  for (llvm::Value *Op : V->operands())
    if (!findOperand<std::tuple<T...>, 0, T...>(Op, Result))
      return std::tuple<T...>{};

  return Result;
}

/// Checks the instruction type and its operands
/// \return the instruction casted to I, or nullptr if not possible.
template<typename I, typename F, typename S>
inline I *isa_with_op(llvm::Instruction *Inst) {
  if (auto *Casted = llvm::dyn_cast<I>(Inst)) {
    revng_assert(Casted->getNumOperands() == 2);
    if (llvm::isa<F>(Casted->getOperand(0))
        && llvm::isa<S>(Casted->getOperand(1))) {
      return Casted;
    } else if (llvm::isa<F>(Casted->getOperand(0))
               && llvm::isa<S>(Casted->getOperand(1))) {
      revng_assert(Casted->isCommutative());
      Casted->swapOperands();
      return Casted;
    }
  }

  return nullptr;
}

template<typename C>
struct BlackListTraitBase {
  BlackListTraitBase(C Obj) : Obj(Obj) {}

protected:
  C Obj;
};

/// Trait to wrap an object of type C that can act as a blacklist for B
template<typename C, typename B>
struct BlackListTrait : BlackListTraitBase<C> {};

class NullBlackList {};

template<typename B>
struct BlackListTrait<const NullBlackList &, B>
  : BlackListTraitBase<const NullBlackList &> {
  using BlackListTraitBase<const NullBlackList &>::BlackListTraitBase;
  bool isBlacklisted(B Value) const { return false; }
};

template<typename C>
struct BlackListTrait<C, C> : BlackListTraitBase<C> {
  using BlackListTraitBase<C>::BlackListTraitBase;
  bool isBlacklisted(C Value) const { return Value == this->Obj; }
};

template<typename B>
struct BlackListTrait<const std::set<B> &, B>
  : BlackListTraitBase<const std::set<B> &> {
  using BlackListTraitBase<const std::set<B> &>::BlackListTraitBase;
  bool isBlacklisted(B Value) const { return this->Obj.contains(Value); }
};

template<typename B, typename C>
inline BlackListTrait<C, B> make_blacklist(C Obj) {
  return BlackListTrait<C, B>(Obj);
}

template<typename B>
inline BlackListTrait<const std::set<B> &, B>
make_blacklist(const std::set<B> &Obj) {
  return BlackListTrait<const std::set<B> &, B>(Obj);
}

/// Possible way to continue (or stop) exploration in a breadth-first visit
enum VisitAction {
  Continue, ///< Visit also the successor basic blocks
  NoSuccessors, ///< Do not visit the successors of this basic block
  ExhaustQueueAndStop, ///< Prevent adding visiting other basic blocks except
                       ///  those already pending
  StopNow ///< Interrupt immediately the visit
};

using BasicBlockRange = llvm::iterator_range<llvm::BasicBlock::iterator>;
using VisitorFunction = std::function<VisitAction(BasicBlockRange)>;

template<bool Forward>
struct IteratorDirection {};

template<>
struct IteratorDirection<true> {

  static llvm::BasicBlock::iterator iterator(llvm::Instruction *I) {
    return ++llvm::BasicBlock::iterator(I);
  }

  static llvm::BasicBlock::iterator begin(llvm::BasicBlock *BB) {
    return BB->begin();
  }

  static llvm::BasicBlock::iterator end(llvm::BasicBlock *BB) {
    return BB->end();
  }
};

template<>
struct IteratorDirection<false> {

  static llvm::BasicBlock::reverse_iterator iterator(llvm::Instruction *I) {
    return llvm::BasicBlock::reverse_iterator(++I->getReverseIterator());
  }

  static llvm::BasicBlock::reverse_iterator begin(llvm::BasicBlock *BB) {
    return BB->rbegin();
  }

  static llvm::BasicBlock::reverse_iterator end(llvm::BasicBlock *BB) {
    return BB->rend();
  }
};

template<bool Forward, typename Derived, typename SuccessorsRange>
struct BFSVisitorBase {
public:
  using BasicBlock = llvm::BasicBlock;
  using forward_iterator = BasicBlock::iterator;
  using backward_iterator = BasicBlock::reverse_iterator;
  template<bool C, typename A, typename B>
  using conditional_t = std::conditional_t<C, A, B>;
  using instruction_iterator = conditional_t<Forward,
                                             forward_iterator,
                                             backward_iterator>;
  using instruction_range = llvm::iterator_range<instruction_iterator>;

  void run(llvm::Instruction *I) {
    auto &ThisDerived = *static_cast<Derived *>(this);
    std::set<BasicBlock *> Visited;

    using ID = IteratorDirection<Forward>;
    instruction_iterator It = ID::iterator(I);

    if (not Forward)
      It--;

    struct WorkItem {
      WorkItem(BasicBlock *BB, instruction_iterator Start) :
        BB(BB), Range(make_range(Start, ID::end(BB))) {}

      WorkItem(BasicBlock *BB) :
        BB(BB), Range(make_range(ID::begin(BB), ID::end(BB))) {}

      BasicBlock *BB = nullptr;
      instruction_range Range;
    };

    std::queue<WorkItem> Queue;
    Queue.push(WorkItem(I->getParent(), It));

    bool ExhaustOnly = false;

    while (not Queue.empty()) {
      WorkItem Item = Queue.front();
      Queue.pop();

      switch (ThisDerived.visit(Item.Range)) {
      case Continue:
        if (not ExhaustOnly) {
          for (auto *Successor : ThisDerived.successors(Item.BB)) {
            if (!Visited.contains(Successor)) {
              Visited.insert(Successor);
              Queue.push(WorkItem(Successor));
            }
          }
        }
        break;
      case NoSuccessors:
        break;
      case ExhaustQueueAndStop:
        ExhaustOnly = true;
        break;
      case StopNow:
        return;
      default:
        revng_abort();
      }
    }
  }
};

template<typename Derived>
struct BackwardBFSVisitor
  : public BFSVisitorBase<false,
                          Derived,
                          llvm::iterator_range<llvm::pred_iterator>> {

  llvm::iterator_range<llvm::pred_iterator> successors(llvm::BasicBlock *BB) {
    return llvm::make_range(pred_begin(BB), pred_end(BB));
  }
};

template<typename Derived>
struct ForwardBFSVisitor
  : public BFSVisitorBase<true,
                          Derived,
                          llvm::iterator_range<llvm::succ_iterator>> {

  llvm::iterator_range<llvm::succ_iterator> successors(llvm::BasicBlock *BB) {
    return llvm::make_range(succ_begin(BB), succ_end(BB));
  }
};

inline std::string getName(const llvm::Value *V);

/// Return a string with the value of a given integer constant.
inline std::string getName(const llvm::ConstantInt *I) {
  return std::to_string(I->getValue().getZExtValue());
}

/// Return a sensible name for the given basic block
/// \return the name of the basic block, if available, its pointer value
///         otherwise.
inline std::string getName(const llvm::BasicBlock *BB) {
  if (BB == nullptr)
    return "(nullptr)";

  llvm::StringRef Result = BB->getName();
  if (!Result.empty()) {
    return Result.str();
  } else {
    std::stringstream SS;
    SS << "0x" << std::hex << intptr_t(BB);
    return SS.str();
  }
}

/// Return a sensible name for the given instruction
/// \return the name of the instruction, if available, a
///         [basic block name]:[instruction index] string otherwise.
inline std::string getName(const llvm::Instruction *I) {
  llvm::StringRef Result = I->getName();
  if (!Result.empty()) {
    return Result.str();
  } else if (const llvm::BasicBlock *Parent = I->getParent()) {
    return getName(Parent) + ":"
           + std::to_string(1
                            + std::distance(Parent->begin(), I->getIterator()));
  } else {
    std::stringstream SS;
    SS << "0x" << std::hex << intptr_t(I);
    return SS.str();
  }
}

/// Return a sensible name for the given function
/// \return the name of the function, if available, its pointer value otherwise.
inline std::string getName(const llvm::Function *F) {
  if (F == nullptr)
    return "(nullptr)";

  if (F->hasName())
    return F->getName().str();

  std::stringstream SS;
  SS << "0x" << std::hex << intptr_t(F);
  return SS.str();
}

/// Return a sensible name for the given argument
/// \return the name of the argument, if available, a
///         [function name]:[argument index] string otherwise.
inline std::string getName(const llvm::Argument *A) {
  if (nullptr == A)
    return "(nullptr)";

  llvm::StringRef Result = A->getName();
  if (not Result.empty()) {
    return Result.str();
  } else {
    const llvm::Function *F = A->getParent();
    return getName(F) + ":" + std::to_string(A->getArgNo());
  }
}

/// Return a sensible name for the given Value
/// \return if \p V is an Instruction, call the appropriate getName function,
///         otherwise return a pointer to \p V.
inline std::string getName(const llvm::Value *V) {
  if (V != nullptr) {
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
      return getName(I);
    if (auto *F = llvm::dyn_cast<llvm::Function>(V))
      return getName(F);
    if (auto *B = llvm::dyn_cast<llvm::BasicBlock>(V))
      return getName(B);
    if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      std::string Result;
      llvm::raw_string_ostream S(Result);
      C->print(S);
      S.flush();
      return Result;
    }
    if (auto *A = llvm::dyn_cast<llvm::Argument>(V))
      return getName(A);
  }
  std::stringstream SS;
  SS << "0x" << std::hex << intptr_t(V);
  return SS.str();
}

inline llvm::BasicBlock *blockByName(llvm::Function *F, const char *Name) {
  using namespace llvm;

  for (BasicBlock &BB : *F)
    if (BB.hasName() and BB.getName() == StringRef(Name))
      return &BB;

  return nullptr;
}

/// Specialization of writeToLog for llvm::Value-derived types
template<typename T>
  requires std::derived_from<llvm::Value, std::remove_const_t<T>>
inline void writeToLog(Logger &This, T *I, int) {
  if (I != nullptr)
    This << getName(I);
  else
    This << "nullptr";
}

inline llvm::LLVMContext &getContext(const llvm::Module *M) {
  return M->getContext();
}

inline llvm::LLVMContext &getContext(const llvm::GlobalObject *G) {
  return getContext(G->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::BasicBlock *BB) {
  return getContext(BB->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::Instruction *I) {
  return getContext(I->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::Value *V) {
  if (auto *I = llvm::dyn_cast<const llvm::Instruction>(V))
    return getContext(I);
  else if (auto *G = llvm::dyn_cast<const llvm::GlobalObject>(V))
    return getContext(G);
  else
    revng_abort();
}

inline const llvm::Module *getModule(const llvm::Function *F) {
  if (F == nullptr)
    return nullptr;
  return F->getParent();
}

inline const llvm::Module *getModule(const llvm::BasicBlock *BB) {
  if (BB == nullptr)
    return nullptr;
  return getModule(BB->getParent());
}

inline const llvm::Module *getModule(const llvm::Instruction *I) {
  if (I == nullptr)
    return nullptr;
  return getModule(I->getParent());
}

inline const llvm::Module *getModule(const llvm::Value *I) {
  if (I == nullptr)
    return nullptr;
  return getModule(llvm::cast<const llvm::Instruction>(I));
}

inline llvm::Module *getModule(llvm::Function *F) {
  if (F == nullptr)
    return nullptr;
  return F->getParent();
}

inline llvm::Module *getModule(llvm::BasicBlock *BB) {
  if (BB == nullptr)
    return nullptr;
  return getModule(BB->getParent());
}

inline llvm::Module *getModule(llvm::Instruction *I) {
  if (I == nullptr)
    return nullptr;
  return getModule(I->getParent());
}

inline llvm::Module *getModule(llvm::Value *I) {
  if (I == nullptr)
    return nullptr;
  return getModule(llvm::cast<llvm::Instruction>(I));
}

/// Helper class to easily create and use LLVM metadata
class QuickMetadata {
private:
  llvm::LLVMContext &C;
  llvm::IntegerType *Int1Ty;
  llvm::IntegerType *Int32Ty;
  llvm::IntegerType *Int64Ty;

public:
  QuickMetadata(llvm::LLVMContext &Context) :
    C(Context),
    Int1Ty(llvm::IntegerType::get(C, 1)),
    Int32Ty(llvm::IntegerType::get(C, 32)),
    Int64Ty(llvm::IntegerType::get(C, 64)) {}

  llvm::MDString *get(const char *String) {
    return llvm::MDString::get(C, String);
  }

  llvm::MDString *get(llvm::StringRef String) {
    return llvm::MDString::get(C, String);
  }

  llvm::ConstantAsMetadata *get(const llvm::APInt &N) {
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(C, N));
  }

  llvm::ConstantAsMetadata *get(llvm::Constant *C) {
    return llvm::ConstantAsMetadata::get(C);
  }

  llvm::ConstantAsMetadata *get(bool Boolean) {
    auto *Constant = llvm::ConstantInt::get(Int1Ty, Boolean);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::ConstantAsMetadata *get(uint32_t Integer) {
    auto *Constant = llvm::ConstantInt::get(Int32Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::ConstantAsMetadata *get(uint64_t Integer) {
    auto *Constant = llvm::ConstantInt::get(Int64Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::ConstantAsMetadata *get(int32_t Integer) {
    auto *Constant = llvm::ConstantInt::getSigned(Int32Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::ConstantAsMetadata *get(int64_t Integer) {
    auto *Constant = llvm::ConstantInt::getSigned(Int64Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::MDNode *get() { return llvm::MDNode::get(C, {}); }

  llvm::MDTuple *tuple(const char *String) { return tuple(get(String)); }

  llvm::MDTuple *tuple(llvm::StringRef String) { return tuple(get(String)); }

  llvm::MDTuple *tuple(uint32_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(uint64_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(int32_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(int64_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(llvm::ArrayRef<llvm::Metadata *> MDs) {
    return llvm::MDTuple::get(C, MDs);
  }

  llvm::MDTuple *tuple() { return llvm::MDTuple::get(C, {}); }

  template<typename T>
  T extract(const llvm::MDTuple *Tuple, unsigned Index) {
    return extract<T>(Tuple->getOperand(Index).get());
  }

  template<typename T>
  T extract(const llvm::Metadata *Tuple, unsigned Index) {
    return extract<T>(cast<llvm::MDTuple>(Tuple), Index);
  }

  template<typename T>
  T extract(const llvm::Metadata *MD) {
    revng_abort();
  }

  template<typename T>
  T extract(llvm::Metadata *MD) {
    revng_abort();
  }
};

template<>
inline llvm::MDTuple *
QuickMetadata::extract<llvm::MDTuple *>(llvm::Metadata *MD) {
  return llvm::cast<llvm::MDTuple>(MD);
}

template<>
inline llvm::Constant *
QuickMetadata::extract<llvm::Constant *>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return C->getValue();
}

template<>
inline llvm::Constant *
QuickMetadata::extract<llvm::Constant *>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return C->getValue();
}

template<>
inline llvm::ConstantInt *
QuickMetadata::extract<llvm::ConstantInt *>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return llvm::cast<llvm::ConstantInt>(C->getValue());
}

template<>
inline llvm::ConstantInt *
QuickMetadata::extract<llvm::ConstantInt *>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return llvm::cast<llvm::ConstantInt>(C->getValue());
}

template<>
inline bool QuickMetadata::extract<bool>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue()) != 0;
}

template<>
inline bool QuickMetadata::extract<bool>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue()) != 0;
}

template<>
inline uint32_t QuickMetadata::extract<uint32_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline uint32_t QuickMetadata::extract<uint32_t>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline uint64_t QuickMetadata::extract<uint64_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline uint64_t QuickMetadata::extract<uint64_t>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline int32_t QuickMetadata::extract<int32_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getSignedLimitedValue(C->getValue());
}

template<>
inline int32_t QuickMetadata::extract<int32_t>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getSignedLimitedValue(C->getValue());
}

template<>
inline int64_t QuickMetadata::extract<int64_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getSignedLimitedValue(C->getValue());
}

template<>
inline int64_t QuickMetadata::extract<int64_t>(llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getSignedLimitedValue(C->getValue());
}

template<>
inline llvm::StringRef
QuickMetadata::extract<llvm::StringRef>(const llvm::Metadata *MD) {
  return llvm::cast<llvm::MDString>(MD)->getString();
}

template<>
inline llvm::StringRef
QuickMetadata::extract<llvm::StringRef>(llvm::Metadata *MD) {
  return llvm::cast<llvm::MDString>(MD)->getString();
}

template<>
inline const llvm::MDString *
QuickMetadata::extract<const llvm::MDString *>(const llvm::Metadata *MD) {
  return llvm::cast<llvm::MDString>(MD);
}

template<>
inline llvm::MDString *
QuickMetadata::extract<llvm::MDString *>(llvm::Metadata *MD) {
  return llvm::cast<llvm::MDString>(MD);
}

/// Return the instruction coming before \p I, or nullptr if it's the first.
inline llvm::Instruction *getPrevious(llvm::Instruction *I) {
  llvm::BasicBlock::reverse_iterator It(++I->getReverseIterator());
  if (It == I->getParent()->rend())
    return nullptr;

  return &*It;
}

/// Return the instruction coming after \p I, or nullptr if it's the last.
inline llvm::Instruction *getNext(llvm::Instruction *I) {
  llvm::BasicBlock::iterator It(I);
  if (It == I->getParent()->end())
    return nullptr;

  It++;
  return &*It;
}

/// Check whether the instruction/basic block is the first in its container
template<typename T>
inline bool isFirst(T *I) {
  revng_assert(I != nullptr);
  return I == &*I->getParent()->begin();
}

inline std::array<unsigned, 3> CastOpcodes = {
  llvm::Instruction::BitCast,
  llvm::Instruction::PtrToInt,
  llvm::Instruction::IntToPtr,
};

// If \p V is a cast Instruction or a cast ConstantExpr, return its only operand
// (recursively)
inline const llvm::Value *skipCasts(const llvm::Value *V) {
  using namespace llvm;
  while (isa<CastInst>(V) or isa<IntToPtrInst>(V) or isa<PtrToIntInst>(V)
         or (isa<ConstantExpr>(V)
             and contains(CastOpcodes, cast<ConstantExpr>(V)->getOpcode())))
    V = cast<User>(V)->getOperand(0);
  return V;
}

// If \p V is a cast Instruction or a cast ConstantExpr, return its only operand
// (recursively)
inline llvm::Value *skipCasts(llvm::Value *V) {
  using namespace llvm;
  while (isa<CastInst>(V) or isa<IntToPtrInst>(V) or isa<PtrToIntInst>(V)
         or (isa<ConstantExpr>(V)
             and contains(CastOpcodes, cast<ConstantExpr>(V)->getOpcode())))
    V = cast<User>(V)->getOperand(0);
  return V;
}

inline bool isMemory(llvm::Value *V) {
  using namespace llvm;
  V = skipCasts(V);
  return not(isa<GlobalVariable>(V) or isa<AllocaInst>(V));
}

inline const llvm::Function *getCallee(const llvm::Instruction *I) {
  revng_assert(I != nullptr);

  using namespace llvm;
  if (auto *Call = dyn_cast<CallInst>(I))
    return llvm::dyn_cast<Function>(skipCasts(Call->getCalledOperand()));
  else
    return nullptr;
}

inline llvm::Function *getCallee(llvm::Instruction *I) {
  revng_assert(I != nullptr);

  using namespace llvm;
  if (auto *Call = dyn_cast<CallInst>(I))
    return llvm::dyn_cast<Function>(skipCasts(Call->getCalledOperand()));
  else
    return nullptr;
}

inline bool isCallTo(const llvm::Value *I, llvm::StringRef Name) {
  revng_assert(I != nullptr);

  auto *Call = llvm::dyn_cast<llvm::Instruction>(I);
  if (not Call)
    return false;

  const llvm::Function *Callee = getCallee(Call);
  return Callee != nullptr && Callee->getName() == Name;
}

inline bool isCallTo(const llvm::Value *I, const llvm::Function *F) {
  revng_assert(I != nullptr);

  auto *Call = llvm::dyn_cast<llvm::Instruction>(I);
  if (not Call)
    return false;

  const llvm::Function *Callee = getCallee(Call);
  return Callee != nullptr && Callee == F;
}

inline llvm::CallInst *getCallTo(llvm::Instruction *I, llvm::StringRef Name) {
  if (isCallTo(I, Name))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

inline llvm::CallInst *getCallTo(llvm::Instruction *I, llvm::Function *F) {
  if (isCallTo(I, F))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

inline const llvm::CallInst *getCallTo(const llvm::Instruction *I,
                                       llvm::StringRef Name) {
  if (isCallTo(I, Name))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

inline const llvm::CallInst *getCallTo(const llvm::Instruction *I,
                                       const llvm::Function *F) {
  if (isCallTo(I, F))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

inline bool CompareByName(const llvm::GlobalVariable *LHS,
                          const llvm::GlobalVariable *RHS) {
  revng_assert(LHS->hasName() and RHS->hasName());
  return LHS->getName() < RHS->getName();
};

template<RangeOf<llvm::GlobalVariable *> RangeOfPointersToGlobal>
inline llvm::SmallVector<llvm::GlobalVariable *>
toSortedByName(const RangeOfPointersToGlobal &Range) {
  llvm::SmallVector<llvm::GlobalVariable *> Result{ Range.begin(),
                                                    Range.end() };
  llvm::sort(Result, CompareByName);
  return Result;
}

class CSVsUsage {
public:
  void sortByName() {
    std::sort(Read.begin(), Read.end(), CompareByName);
    std::sort(Written.begin(), Written.end(), CompareByName);
  }

public:
  std::vector<llvm::GlobalVariable *> Read;
  std::vector<llvm::GlobalVariable *> Written;
};

inline MetaAddress getBasicBlockJumpTarget(llvm::BasicBlock *BB) {
  using namespace llvm;

  Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return MetaAddress::invalid();

  if (llvm::CallInst *Call = getCallTo(I, "newpc")) {
    if (getLimitedValue(Call->getOperand(2)) == 1) {
      return MetaAddress::fromValue(Call->getOperand(0));
    }
  }

  return MetaAddress::invalid();
}

template<typename V>
concept ValueLikePrintable = requires(V Val) {
  Val.print(std::declval<llvm::raw_ostream &>(), true);
};

template<typename ValueType>
concept ModuleSlotTrackerPrintable = requires(llvm::ModuleSlotTracker &Tracker,
                                              ValueType Value) {
  Value.print(std::declval<llvm::raw_ostream &>(), Tracker, true);
};

template<typename F>
concept ModFunLikePrintable = requires(F Fun) {
  Fun.print(std::declval<llvm::raw_ostream &>(), nullptr, false, true);
};

template<typename T>
concept LLVMRawOStreamPrintable = not ValueLikePrintable<T>
                                  and not ModFunLikePrintable<T>
                                  and requires(T V, llvm::raw_ostream &S) {
                                        V.print(S);
                                      };

// This is enabled only for references to types that inherit from llvm::Value
// but not from llvm::Function, since llvm::Function has a different prototype
// for the print() method
template<ValueLikePrintable ValueRef>
inline std::string dumpToString(ValueRef &V) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  V.print(Stream, true);
  Stream.flush();
  return Result;
}

// This is is similar to the overload above, but it allows taking advantage of
// a `llvm::ModuleSlotTracker` optimization.
template<ModuleSlotTrackerPrintable ValueRef>
inline std::string dumpToString(ValueRef &V, llvm::ModuleSlotTracker &Tracker) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  V.print(Stream, Tracker, true);
  Stream.flush();
  return Result;
}

// This is enabled only for references to types that inherit from llvm::Module
// or from llvm::Function, which share the same prototype for the print() method
template<ModFunLikePrintable ModOrFunRef>
inline std::string dumpToString(ModOrFunRef &M) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  M.print(Stream, nullptr, false, true);
  Stream.flush();
  return Result;
}

// This is enabled for all types with a print() method that prints to an
// llvm::raw_ostream
template<LLVMRawOStreamPrintable T>
inline std::string dumpToString(T &TheT) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  TheT.print(Stream);
  Stream.flush();
  return Result;
}

template<typename T>
concept LLVMRawOStreamDumpable = not ValueLikePrintable<T>
                                 and not ModFunLikePrintable<T>
                                 and requires(T TheT) {
                                       TheT.dump(std::declval<
                                                 llvm::raw_ostream &>());
                                     };

template<LLVMRawOStreamDumpable Dumpable>
inline void writeToLog(Logger &L, const Dumpable &P, int /* Ignore */) {
  if (L.isEnabled()) {
    llvm::SmallString<32> Buffer;
    {
      llvm::raw_svector_ostream Stream(Buffer);
      P.dump(Stream);
      L << Stream.str().str();
    }
  }
}

template<typename T>
  requires std::is_pointer_v<T>
inline std::string dumpToString(T TheT) {
  if (TheT == nullptr)
    return "nullptr";
  return dumpToString(*TheT);
}

void dumpModule(const llvm::Module *M, const char *Path) debug_function;

llvm::PointerType *getStringPtrType(llvm::LLVMContext &C);

llvm::GlobalVariable *
buildString(llvm::Module *M, llvm::StringRef String, const llvm::Twine &Name);

llvm::Constant *getUniqueString(llvm::Module *M,
                                llvm::StringRef String,
                                llvm::StringRef Namespace = "revng.const.");

llvm::StringRef extractFromConstantStringPtr(llvm::Value *V);

inline llvm::User *getUniqueUser(llvm::Value *V) {
  llvm::User *Result = nullptr;

  for (llvm::User *U : V->users()) {
    if (Result != nullptr)
      return nullptr;
    else
      Result = U;
  }

  return Result;
}

const llvm::BasicBlock *getJumpTargetBlock(const llvm::BasicBlock *BB);

inline BasicBlockID blockIDFromNewPC(const llvm::CallBase *Call) {
  revng_assert(isCallTo(Call, "newpc"));
  using namespace NewPCArguments;
  auto *Argument = Call->getArgOperand(InstructionID);
  return BasicBlockID::fromValue(Argument);
}

inline BasicBlockID blockIDFromNewPC(const llvm::Instruction *I) {
  return blockIDFromNewPC(llvm::cast<llvm::CallBase>(I));
}

inline MetaAddress addressFromNewPC(const llvm::CallBase *Call) {
  return blockIDFromNewPC(Call).notInlinedAddress();
}
inline MetaAddress addressFromNewPC(const llvm::Instruction *I) {
  return addressFromNewPC(llvm::cast<llvm::CallBase>(I));
}

inline BasicBlockID getBasicBlockID(const llvm::BasicBlock *BB) {
  using namespace llvm;

  revng_assert(BB != nullptr);

  const Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return BasicBlockID::invalid();

  if (const llvm::CallInst *Call = getCallTo(I, "newpc"))
    return blockIDFromNewPC(Call);

  return BasicBlockID::invalid();
}

inline MetaAddress getBasicBlockAddress(const llvm::BasicBlock *BB) {
  return getBasicBlockID(BB).notInlinedAddress();
}

/// Find the first call to NewPC starting from \p TheInstruction
///
llvm::CallInst *getLastNewPC(llvm::Instruction *TheInstruction);

/// Find the PC which lead to generated \p TheInstruction
///
/// \return a pair of integers: the first element represents the PC and the
///         second the size of the instruction.
std::pair<MetaAddress, uint64_t> getPC(llvm::Instruction *TheInstruction);

/// Replace all uses of \Old, with \New in \F.
///
/// \return true if it changes something, false otherwise.
inline bool replaceAllUsesInFunctionWith(llvm::Function *F,
                                         llvm::Value *Old,
                                         llvm::Value *New) {
  using namespace llvm;
  if (Old == New)
    return false;

  bool Changed = false;

  SmallPtrSet<ConstantExpr *, 8> OldUserConstExprs;
  auto UI = Old->use_begin();
  auto E = Old->use_end();
  while (UI != E) {
    Use &U = *UI;
    ++UI;

    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      if (I->getFunction() == F) {
        U.set(New);
        Changed = true;
      }
    } else if (auto *CE = dyn_cast<ConstantExpr>(U.getUser())) {
      // We can't convert ConstantExprs to Instructions while iterating on Old
      // uses. This would create new uses of Old (the new Instructions generated
      // by converting the ConstantExprs to Instructions) while iterating on Old
      // uses, so the trick with pre-incrementing the iterators used above would
      // not be enough to guard us from iterator invalidation.
      // We store ConstantExpr uses in a helper vector and process them later.
      if (CE->isCast())
        OldUserConstExprs.insert(CE);
    }
  }

  // Iterate on all ConstantExpr that use Old.
  for (ConstantExpr *OldUserCE : OldUserConstExprs) {
    // For each ConstantExpr that uses Old, we are interested in its uses in F,
    // so we iterate on all uses of OldUserCE, looking for uses in Instructions
    // that are in F.
    // When we find one, we cannot directly substitute the use of Old in
    // OldUserCE, because that is a constant expression that might be used
    // somewhere else, possibly outside of F.
    // What we do instead is to create an Instruction in F that is equivalent to
    // OldUserCE, and substitute Old with New only in that instruction.
    auto CEIt = OldUserCE->use_begin();
    auto CEEnd = OldUserCE->use_end();
    for (; CEIt != CEEnd;) {
      Use &CEUse = *CEIt;
      ++CEIt;
      auto *CEInstrUser = dyn_cast<Instruction>(CEUse.getUser());
      if (CEInstrUser and CEInstrUser->getFunction() == F) {
        Instruction *CastInst = OldUserCE->getAsInstruction();
        CastInst->replaceUsesOfWith(Old, New);
        CastInst->insertBefore(CEInstrUser);
        CEUse.set(CastInst);
        Changed = true;
      }
    }
  }
  return Changed;
}

inline llvm::Constant *toLLVMConstant(llvm::LLVMContext &Context,
                                      const llvm::APInt &Value) {
  using namespace llvm;
  llvm::Type *DestinationType = llvm::IntegerType::get(Context,
                                                       Value.getBitWidth());
  if (auto *IT = dyn_cast<IntegerType>(DestinationType)) {
    revng_assert(IT->getBitWidth() == Value.getBitWidth());
    return ConstantInt::get(DestinationType, Value);
  } else {
    revng_abort();
  }
}

template<typename T>
inline llvm::Type *cTypeToLLVMType(llvm::LLVMContext &C) {
  using namespace std;
  using namespace llvm;
  if constexpr (is_integral_v<T>) {
    return Type::getIntNTy(C, 8 * sizeof(T));
  } else if (is_pointer_v<T>) {
    return cTypeToLLVMType<remove_pointer_t<T>>(C)->getPointerTo();
  } else if (is_void_v<T>) {
    return Type::getVoidTy(C);
  } else {
    revng_abort();
  }
}

template<typename ReturnT, typename... Args>
inline llvm::FunctionType *
createFunctionType(llvm::LLVMContext &C, bool Variadic = false) {
  return llvm::FunctionType::get(cTypeToLLVMType<ReturnT>(C),
                                 { cTypeToLLVMType<Args>(C)... },
                                 Variadic);
}

inline cppcoro::generator<llvm::CallBase *> callers(llvm::Function *F) {
  using namespace llvm;
  SmallVector<Value *, 8> Queue;
  Queue.push_back(F);

  while (not Queue.empty()) {
    Value *V = Queue.back();
    Queue.pop_back();

    for (User *U : V->users()) {
      if (auto *Call = dyn_cast<CallBase>(U)) {
        co_yield Call;
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->isCast())
          Queue.push_back(CE);
      }
    }
  }
}

inline cppcoro::generator<const llvm::CallBase *>
callers(const llvm::Function *F) {
  using namespace llvm;
  SmallVector<const Value *, 8> Queue;
  Queue.push_back(F);

  while (not Queue.empty()) {
    const Value *V = Queue.back();
    Queue.pop_back();

    for (const User *U : V->users()) {
      if (const auto *Call = dyn_cast<CallBase>(U)) {
        co_yield Call;
      } else if (const auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->isCast())
          Queue.push_back(CE);
      }
    }
  }
}

inline cppcoro::generator<llvm::CallBase *>
callersIn(llvm::Function *F, llvm::Function *ContainingFunction) {
  for (llvm::CallBase *Call : callers(F))
    if (Call->getParent()->getParent() == ContainingFunction)
      co_yield Call;
}

template<typename T>
concept HasMetadata = requires(T &Value,
                               const T &ConstValue,
                               llvm::StringRef KindName,
                               unsigned KindID,
                               llvm::MDNode *MD) {
  Value.setMetadata(KindName, MD);
  Value.setMetadata(KindID, MD);
  { ConstValue.getMetadata(KindName) } -> std::same_as<llvm::MDNode *>;
  { ConstValue.getMetadata(KindID) } -> std::same_as<llvm::MDNode *>;
};

static_assert(HasMetadata<llvm::Instruction>);
static_assert(HasMetadata<llvm::Function>);
static_assert(HasMetadata<llvm::GlobalVariable>);
static_assert(not HasMetadata<llvm::Constant>);

template<HasMetadata T>
llvm::StringRef fromStringMetadata(const T *U, llvm::StringRef Name) {
  using namespace llvm;

  if (auto *MD = dyn_cast_or_null<MDTuple>(U->getMetadata(Name)))
    if (auto *String = dyn_cast<MDString>(MD->getOperand(0)))
      return String->getString();

  revng_abort();
}

template<HasMetadata T>
void setStringMetadata(T *U, llvm::StringRef Name, llvm::StringRef Value) {
  llvm::LLVMContext &C = getContext(U);
  U->setMetadata(Name,
                 llvm::MDTuple::get(C, { llvm::MDString::get(C, Value) }));
}

template<HasMetadata T>
MetaAddress getMetaAddressMetadata(const T *U, llvm::StringRef Name) {
  using namespace llvm;

  if (auto *MD = dyn_cast_or_null<MDTuple>(U->getMetadata(Name)))
    if (auto *String = dyn_cast<MDString>(MD->getOperand(0)))
      return MetaAddress::fromString(String->getString());

  return MetaAddress::invalid();
}

template<HasMetadata T>
void setMetaAddressMetadata(T *U, llvm::StringRef Name, const MetaAddress &MA) {
  using namespace llvm;
  auto &Context = getContext(U);
  auto *MD = MDTuple::get(Context, MDString::get(Context, MA.toString()));
  U->setMetadata(Name, MD);
}

template<typename T>
inline llvm::cl::opt<T> *
getOption(llvm::StringMap<llvm::cl::Option *> &Options, const char *Name) {
  return static_cast<llvm::cl::opt<T> *>(Options[Name]);
}

/// Extract MD text from MDString or GlobalVariable
llvm::StringRef getText(const llvm::Instruction *I, unsigned Kind);

inline void setInsertPointToFirstNonAlloca(revng::IRBuilder &Builder,
                                           llvm::Function &F) {
  using namespace llvm;

  BasicBlock &Entry = F.getEntryBlock();
  for (Instruction &I : Entry) {
    if (not isa<AllocaInst>(&I)) {
      Builder.SetInsertPoint(&I);
      return;
    }
  }
  revng_abort();
}

inline llvm::Value *getPointer(llvm::User *U) {
  using namespace llvm;

  if (auto *Load = dyn_cast<LoadInst>(U))
    return Load->getPointerOperand();
  else if (auto *Store = dyn_cast<StoreInst>(U))
    return Store->getPointerOperand();
  else
    return nullptr;
}

/// Steal the body of \p OldFunction and move it into \p NewFunction
void moveBlocksInto(llvm::Function &OldFunction, llvm::Function &NewFunction);

/// Create a new function similar to \p OldFunction and steals its name (but not
/// the body)
llvm::Function &recreateWithoutBody(llvm::Function &OldFunction,
                                    llvm::FunctionType &NewType);

/// Create a function identical to \p OldFunction but use \p NewType as type
llvm::Function &moveToNewFunctionType(llvm::Function &OldFunction,
                                      llvm::FunctionType &NewType);

/// Adds NewArguments and changes the return type of \p OldFunction
///
/// \param OldFunction the original function from which the body will be stolen.
/// \param NewReturnType the new return type. It can be: 1) nullptr to preserve
///        the old one, 2) the old type or 3), if the original type is void, a
///        new type.
/// \param NewArguments extra arguments to add on top of the existing ones.
///
/// \return the newly created Function.
///
/// \note \p OldFunction will not be deleted or RAUW'd.
llvm::Function *changeFunctionType(llvm::Function &OldFunction,
                                   llvm::Type *NewReturnType,
                                   llvm::ArrayRef<llvm::Type *> NewArguments);

inline llvm::SmallVector<llvm::Value *, 4> unpack(revng::IRBuilder &Builder,
                                                  llvm::Value *V) {
  using namespace llvm;
  Type *Type = V->getType();
  if (isa<IntegerType>(Type)) {
    return { V };
  } else if (auto *StructType = dyn_cast<llvm::StructType>(Type)) {
    llvm::SmallVector<llvm::Value *, 4> Result;
    for (unsigned I = 0; I < StructType->getNumElements(); ++I)
      Result.push_back(Builder.CreateExtractValue(V, { I }));
    return Result;
  } else {
    revng_abort("Cannot unpack the given type");
  }
}

inline llvm::Instruction *createLoad(revng::IRBuilder &Builder,
                                     llvm::GlobalVariable *GV) {
  return Builder.CreateLoad(GV->getValueType(), GV);
}

inline llvm::Instruction *createLoad(revng::IRBuilder &Builder,
                                     llvm::AllocaInst *Alloca) {
  return Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);
}

inline llvm::Instruction *createLoadVariable(revng::IRBuilder &Builder,
                                             llvm::Value *Variable) {
  if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Variable))
    return createLoad(Builder, Alloca);
  else if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Variable))
    return createLoad(Builder, GV);
  else
    revng_abort("Either GlobalVariable or AllocaInst expected");
}

inline llvm::Type *getVariableType(const llvm::Value *Variable) {
  if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Variable))
    return Alloca->getAllocatedType();
  else if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Variable))
    return GV->getValueType();
  else
    revng_abort("Either GlobalVariable or AllocaInst expected");
}

void pruneDICompileUnits(llvm::Module &M);

llvm::SmallSet<llvm::Value *, 2> findPhiTreeLeaves(llvm::Value *Root);

namespace revng {

/// If the verify logger is enabled, assert the module is valid
void verify(const llvm::Module *M);

/// If the verify logger is enabled, assert the function is valid
void verify(const llvm::Function *F);

/// Assert the module is valid
void forceVerify(const llvm::Module *M);

/// Assert the function is valid
void forceVerify(const llvm::Function *F);

} // namespace revng

void collectTypes(llvm::Type *Root, std::set<llvm::Type *> &Set);

namespace llvm {
class DominatorTree;
}

void pushInstructionALAP(llvm::DominatorTree &DT, llvm::Instruction *ToMove);

template<typename T>
std::optional<T> getConstantArg(llvm::CallInst *Call, unsigned Index) {
  using namespace llvm;

  if (auto *CI = dyn_cast<ConstantInt>(Call->getArgOperand(Index))) {
    return CI->getLimitedValue();
  } else {
    return {};
  }
}

inline std::optional<uint64_t> getUnsignedConstantArg(llvm::CallInst *Call,
                                                      unsigned Index) {
  return getConstantArg<uint64_t>(Call, Index);
}

inline std::optional<int64_t> getSignedConstantArg(llvm::CallInst *Call,
                                                   unsigned Index) {
  return getConstantArg<int64_t>(Call, Index);
}

/// Deletes the body of an llvm::Function, but preserving all the tags and
/// attributes (which llvm::Function::deleteBody() does not preserve).
/// Returns true if the body was cleared, false if it was already empty.
bool deleteOnlyBody(llvm::Function &F);

unsigned getMemoryAccessSize(llvm::Instruction *I);

class MetadataBackup {
private:
  llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 2> SavedMetadata;

public:
  MetadataBackup(llvm::Function *Source) {
    Source->getAllMetadata(SavedMetadata);
  }

public:
  void restoreIn(llvm::Function *Destination) const {
    // Restore metadata
    for (auto &MD : SavedMetadata) {
      // The !dbg attachment from the function definition cannot be attached
      // to its declaration.
      if (isa<llvm::DISubprogram>(MD.second))
        continue;

      Destination->addMetadata(MD.first, *MD.second);
    }
  }
};

void sortModule(llvm::Module &M);

std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Context,
                                      llvm::StringRef Path);

/// \p FinalLinkage final linkage for all the globals. Use std::nullopt to
///    preserve the original one.
void linkModules(std::unique_ptr<llvm::Module> &&Source,
                 llvm::Module &Destination,
                 std::optional<llvm::GlobalValue::LinkageTypes> FinalLinkage);

inline void linkModules(std::unique_ptr<llvm::Module> &&Source,
                        llvm::Module &Destination) {
  linkModules(std::move(Source), Destination, std::nullopt);
}
