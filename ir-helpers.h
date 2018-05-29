#ifndef _IRHELPERS_H
#define _IRHELPERS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>
#include <queue>
#include <sstream>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Interval.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

static inline void replaceInstruction(llvm::Instruction *Old,
                                      llvm::Instruction *New) {
  Old->replaceAllUsesWith(New);

  llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 2> Metadata;
  Old->getAllMetadata(Metadata);
  for (auto& MDPair : Metadata)
    New->setMetadata(MDPair.first, MDPair.second);

  Old->eraseFromParent();
}

/// Helper function to destroy an unconditional branch and, in case, the target
/// basic block, if it doesn't have any predecessors left.
static inline void purgeBranch(llvm::BasicBlock::iterator I) {
  auto *DeadBranch = llvm::dyn_cast<llvm::BranchInst>(I);
  // We allow only a branch and nothing else
  assert(DeadBranch != nullptr &&
         ++I == DeadBranch->getParent()->end());

  std::set<llvm::BasicBlock *> Successors;
  for (unsigned C = 0; C < DeadBranch->getNumSuccessors(); C++)
    Successors.insert(DeadBranch->getSuccessor(C));

  // Destroy the dead branch
  DeadBranch->eraseFromParent();

  // Check if someone else was jumping there and then destroy
  for (llvm::BasicBlock *BB : Successors)
    if (BB->empty() && llvm::pred_empty(BB))
      BB->eraseFromParent();
}

static inline llvm::ConstantInt *getConstValue(llvm::Constant *C,
                                               const llvm::DataLayout &DL) {
  while (auto *Expr = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    C = ConstantFoldConstantExpression(Expr, DL);

    if (Expr->getOpcode() == llvm::Instruction::IntToPtr
        || Expr->getOpcode() == llvm::Instruction::PtrToInt)
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

static inline uint64_t getSExtValue(llvm::Constant *C,
                                    const llvm::DataLayout &DL){
  return getConstValue(C, DL)->getSExtValue();
}

static inline uint64_t getZExtValue(llvm::Constant *C,
                                    const llvm::DataLayout &DL){
  return getConstValue(C, DL)->getZExtValue();
}

static inline uint64_t getExtValue(llvm::Constant *C,
                                   bool Sign,
                                   const llvm::DataLayout &DL){
  if (Sign)
    return getSExtValue(C, DL);
  else
    return getZExtValue(C, DL);
}

static inline uint64_t getLimitedValue(const llvm::Value *V) {
  return llvm::cast<llvm::ConstantInt>(V)->getLimitedValue();
}

static inline llvm::iterator_range<llvm::Interval::pred_iterator>
predecessors(llvm::Interval *BB) {
  return make_range(pred_begin(BB), pred_end(BB));
}

static inline llvm::iterator_range<llvm::Interval::succ_iterator>
successors(llvm::Interval *BB) {
  return make_range(succ_begin(BB), succ_end(BB));
}

template<typename T, unsigned I>
static inline bool findOperand(llvm::Value *Op, T &Result) {
  return false;
}

template<typename T, unsigned I, typename Head, typename... Tail>
static inline bool findOperand(llvm::Value *Op, T &Result) {
  using VT = typename std::remove_pointer<Head>::type;
  if (auto *Casted = llvm::dyn_cast<VT>(Op)) {
    std::get<I>(Result) = Casted;
    return true;
  } else {
    return findOperand<T, I + 1, Tail...>(Op, Result);
  }
}

/// \brief Return a tuple of \p V's operands of the requested types
/// \return a tuple with the operands of the specified type in the specified
///         order, or, if not possible, a nullptr tuple.
template<typename... T>
static inline std::tuple<T...> operandsByType(llvm::User *V) {
  std::tuple<T...> Result;
  unsigned OpCount = V->getNumOperands();
  assert(OpCount == sizeof...(T));

  for (llvm::Value *Op : V->operands())
    if (!findOperand<std::tuple<T...>, 0, T...>(Op, Result))
      return std::tuple<T...> { };

  return Result;
}

/// \brief Checks the instruction type and its operands
/// \return the instruction casted to I, or nullptr if not possible.
template<typename I, typename F, typename S>
static inline I *isa_with_op(llvm::Instruction *Inst) {
  if (auto *Casted = llvm::dyn_cast<I>(Inst)) {
    assert(Casted->getNumOperands() == 2);
    if (llvm::isa<F>(Casted->getOperand(0))
        && llvm::isa<S>(Casted->getOperand(1))) {
      return Casted;
    } else if (llvm::isa<F>(Casted->getOperand(0))
               && llvm::isa<S>(Casted->getOperand(1))) {
      assert(Casted->isCommutative());
      Casted->swapOperands();
      return Casted;
    }
  }

  return nullptr;
}

/// \brief Return an range iterating backward from the given instruction
static inline llvm::iterator_range<llvm::BasicBlock::reverse_iterator>
backward_range(llvm::Instruction *I) {
  return llvm::make_range(llvm::make_reverse_iterator(I->getIterator()),
                          I->getParent()->rend());
}

template<typename C>
struct BlackListTraitBase {
  BlackListTraitBase(C Obj) : Obj(Obj) { }
protected:
  C Obj;
};

/// \brief Trait to wrap an object of type C that can act as a blacklist for B
template<typename C, typename B>
struct BlackListTrait : BlackListTraitBase<C> {
};

template<typename C>
struct BlackListTrait<C, C> : BlackListTraitBase<C> {
  using BlackListTraitBase<C>::BlackListTraitBase;
  bool isBlacklisted(C Value) { return Value == this->Obj; }
};

template<typename B>
struct BlackListTrait<const std::set<B> &, B>
  : BlackListTraitBase<const std::set<B> &> {
  using BlackListTraitBase<const std::set<B> &>::BlackListTraitBase;
  bool isBlacklisted(B Value) { return this->Obj.count(Value) != 0; }
};

template<typename B, typename C>
static inline BlackListTrait<C, B> make_blacklist(C Obj) {
  return BlackListTrait<C, B>(Obj);
}

template<typename B>
static inline BlackListTrait<const std::set<B> &, B>
make_blacklist(const std::set<B> &Obj) {
  return BlackListTrait<const std::set<B> &, B>(Obj);
}

/// \brief Possible way to continue (or stop) exploration in a breadth-first
///        visit
enum VisitAction {
  Continue, ///< Visit also the successor basic blocks
  NoSuccessors, ///< Do not visit the successors of this basic block
  ExhaustQueueAndStop, ///< Prevent adding visiting other basic blocks except
                       ///  those already pending
  StopNow ///< Interrupt immediately the visit
};

using BasicBlockRange = llvm::iterator_range<llvm::BasicBlock::iterator>;
using VisitorFunction = std::function<VisitAction(BasicBlockRange)>;

/// Performs a breadth-first visit of the instructions after \p I and in the
/// successor basic blocks
///
/// \param I the instruction from where to the visit should start
/// \param BL a blacklist for basic blocks to ignore.
/// \param Visitor the visitor function, see VisitAction to understand what this
///        function should return
template<typename C>
static inline void visitSuccessors(llvm::Instruction *I,
                                   BlackListTrait<C, llvm::BasicBlock *> BL,
                                   VisitorFunction Visitor) {
  std::set<llvm::BasicBlock *> Visited;

  llvm::BasicBlock::iterator It(I);
  It++;

  std::queue<llvm::iterator_range<llvm::BasicBlock::iterator>> Queue;
  Queue.push(make_range(It, I->getParent()->end()));

  bool ExhaustOnly = false;

  while (!Queue.empty()) {
    auto Range = Queue.front();
    Queue.pop();

    switch (Visitor(Range)) {
    case Continue:
      if (!ExhaustOnly) {
        for (auto *Successor : successors(Range.begin()->getParent())) {
          if (Visited.count(Successor) == 0
              && !BL.isBlacklisted(Successor)) {
            Visited.insert(Successor);
            Queue.push(make_range(Successor->begin(), Successor->end()));
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
      assert(false);
    }
  }
}

using RBasicBlockRange =
  llvm::iterator_range<llvm::BasicBlock::reverse_iterator>;
using RVisitorFunction = std::function<VisitAction(RBasicBlockRange)>;

/// Performs a breadth-first visit of the instructions before \p I and in the
/// predecessor basic blocks
///
/// \param I the instruction from where to the visit should start
/// \param BL a blacklist for basic blocks to ignore.
/// \param Visitor the visitor function, see VisitAction to understand what this
///        function should return
template<typename C>
static inline void visitPredecessors(llvm::Instruction *I,
                                     RVisitorFunction Visitor,
                                     BlackListTrait<C, llvm::BasicBlock *> BL) {
  std::set<llvm::BasicBlock *> Visited;

  llvm::BasicBlock::reverse_iterator It(make_reverse_iterator(I));

  if (It == I->getParent()->rend())
    return;

  std::queue<RBasicBlockRange> Queue;
  Queue.push(llvm::make_range(It, I->getParent()->rend()));

  bool ExhaustOnly = false;

  while (!Queue.empty()) {
    RBasicBlockRange &Range = Queue.front();
    Queue.pop();

    switch (Visitor(Range)) {
    case Continue:
      if (!ExhaustOnly && Range.begin() != Range.end()) {
        for (auto *Predecessor : predecessors(Range.begin()->getParent())) {
          if (Visited.count(Predecessor) == 0
              && !BL.isBlacklisted(Predecessor)) {
            Visited.insert(Predecessor);
            Queue.push(make_range(Predecessor->rbegin(), Predecessor->rend()));
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
      assert(false);
    }
  }
}

/// \brief Return a sensible name for the given basic block
/// \return the name of the basic block, if available, its pointer value
///         otherwise.
static inline std::string getName(const llvm::BasicBlock *BB) {
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

/// \brief Return a sensible name for the given instruction
/// \return the name of the instruction, if available, a
///         [basic blockname]:[instruction index] string otherwise.
static inline std::string getName(const llvm::Instruction *I) {
  llvm::StringRef Result = I->getName();
  if (!Result.empty()) {
    return Result.str();
  } else {
    const llvm::BasicBlock *Parent = I->getParent();
    return getName(Parent) + ":"
      + std::to_string(1 + std::distance(Parent->begin(), I->getIterator()));
  }
}

/// \brief Return a sensible name for the given Value
/// \return if \p V is an Instruction, call the appropriate getName function,
///         otherwise return a pointer to \p V.
static inline std::string getName(const llvm::Value *V) {
  if (V != nullptr)
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
      return getName(I);
  std::stringstream SS;
  SS << "0x" << std::hex << intptr_t(V);
  return SS.str();
}

static inline llvm::LLVMContext &getContext(const llvm::Module *M) {
  return M->getContext();
}

static inline llvm::LLVMContext &getContext(const llvm::Function *F) {
  return getContext(F->getParent());
}

static inline llvm::LLVMContext &getContext(const llvm::BasicBlock *BB) {
  return getContext(BB->getParent());
}

static inline llvm::LLVMContext &getContext(const llvm::Instruction *I) {
  return getContext(I->getParent());
}

static inline llvm::LLVMContext &getContext(const llvm::Value *I) {
  return getContext(llvm::cast<const llvm::Instruction>(I));
}

static inline const llvm::Module *getModule(const llvm::Function *F) {
  return F->getParent();
}

static inline const llvm::Module *getModule(const llvm::BasicBlock *BB) {
  return getModule(BB->getParent());
}

static inline const llvm::Module *getModule(const llvm::Instruction *I) {
  return getModule(I->getParent());
}

static inline const llvm::Module *getModule(const llvm::Value *I) {
  return getModule(llvm::cast<const llvm::Instruction>(I));
}

/// \brief Helper class to easily create and use LLVM metadata
class QuickMetadata {
public:
  QuickMetadata(llvm::LLVMContext &Context) : C(Context),
    Int32Ty(llvm::IntegerType::get(C, 32)) { }

  llvm::MDString *get(const char *String) {
    return llvm::MDString::get(C, String);
  }

  llvm::MDString *get(llvm::StringRef String) {
    return llvm::MDString::get(C, String);
  }

  llvm::ConstantAsMetadata *get(uint32_t Integer) {
    auto *Constant = llvm::ConstantInt::get(Int32Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::MDTuple *tuple(const char *String) {
    return tuple(get(String));
  }

  llvm::MDTuple *tuple(llvm::StringRef String) {
    return tuple(get(String));
  }

  llvm::MDTuple *tuple(uint32_t Integer) {
    return tuple(get(Integer));
  }

  llvm::MDTuple *tuple(llvm::ArrayRef<llvm::Metadata *> MDs) {
    return llvm::MDTuple::get(C, MDs);
  }

  template<typename T>
  T extract(const llvm::MDTuple *Tuple, unsigned Index) {
    return extract<T>(Tuple->getOperand(Index).get());
  }

  template<typename T>
  T extract(const llvm::Metadata *MD) {
    abort();
  }

private:
  llvm::LLVMContext &C;
  llvm::IntegerType *Int32Ty;
};

template<>
inline uint32_t QuickMetadata::extract<uint32_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline uint64_t QuickMetadata::extract<uint64_t>(const llvm::Metadata *MD) {
  auto *C = llvm::cast<llvm::ConstantAsMetadata>(MD);
  return getLimitedValue(C->getValue());
}

template<>
inline llvm::StringRef
QuickMetadata::extract<llvm::StringRef>(const llvm::Metadata *MD) {
  return llvm::cast<llvm::MDString>(MD)->getString();
}

/// \brief Return the instruction coming before \p I, or nullptr if it's the
///        first.
static inline llvm::Instruction *getPrevious(llvm::Instruction *I) {
  llvm::BasicBlock::reverse_iterator It(make_reverse_iterator(I));
  if (It == I->getParent()->rend())
    return nullptr;

  return &*It;
}

/// \brief Return the instruction coming after \p I, or nullptr if it's the
///        last.
static inline llvm::Instruction *getNext(llvm::Instruction *I) {
  llvm::BasicBlock::iterator It(I);
  if (It == I->getParent()->end())
    return nullptr;

  It++;
  return &*It;
}

/// \brief Check whether the instruction/basic block is the first in its
///        container or not
template<typename T>
static inline bool isFirst(T *I) {
  assert(I != nullptr);
  return I == &*I->getParent()->begin();
}

/// \brief Check if among \p BB's predecessors there's \p Target
static inline bool hasPredecessor(llvm::BasicBlock *BB,
                                  llvm::BasicBlock *Target) {
  for (llvm::BasicBlock *Predecessor : predecessors(BB))
    if (Predecessor == Target)
      return true;
  return false;
}

// \brief If \p V is a cast Instruction or a cast ConstantExpr, return its only
//        operand (recursively)
static inline llvm::Value *skipCasts(llvm::Value *V) {
  using namespace llvm;
  while (isa<CastInst>(V)
         || (isa<ConstantExpr>(V)
             && cast<ConstantExpr>(V)->getOpcode() == Instruction::BitCast))
    V = cast<User>(V)->getOperand(0);
  return V;
}

static inline bool isCallTo(const llvm::Instruction *I, llvm::StringRef Name) {
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(I)) {
    llvm::Function *Callee = Call->getCalledFunction();
    if (Callee != nullptr && Callee->getName() == Name) {
      return true;
    }
  }

  return false;
}

static inline llvm::CallInst *getCallTo(llvm::Instruction *I,
                                        llvm::StringRef Name) {
  if (isCallTo(I, Name))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

// TODO: this function assumes 0 is not a valid PC
static inline uint64_t getBasicBlockPC(llvm::BasicBlock *BB) {
  auto It = BB->begin();
  assert(It != BB->end());
  if (llvm::CallInst *Call = getCallTo(&*It, "newpc")) {
    return getLimitedValue(Call->getOperand(0));
  }

  return 0;
}

template<typename C>
static inline auto skip(unsigned ToSkip, C &&Container)
  -> llvm::iterator_range<decltype(Container.begin())> {

  auto Begin = std::begin(Container);
  while (ToSkip --> 0)
    Begin++;
  return llvm::make_range(Begin, std::end(Container));
}

template<class Container, class UnaryPredicate>
static inline void erase_if(Container &C, UnaryPredicate P) {
  C.erase(std::remove_if(C.begin(), C.end(), P), C.end());
}

#endif // _IRHELPERS_H
