#ifndef IRHELPERS_H
#define IRHELPERS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <queue>
#include <set>
#include <sstream>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Interval.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

// Local libraries includes
#include "revng/Support/Debug.h"

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
  DeadBranch->eraseFromParent();

  // Check if someone else was jumping there and then destroy
  for (llvm::BasicBlock *BB : Successors)
    if (BB->empty() && llvm::pred_empty(BB))
      BB->eraseFromParent();
}

inline llvm::ConstantInt *
getConstValue(llvm::Constant *C, const llvm::DataLayout &DL) {
  while (auto *Expr = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    C = ConstantFoldConstant(Expr, DL);

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

inline llvm::iterator_range<llvm::Interval::pred_iterator>
predecessors(llvm::Interval *BB) {
  return make_range(pred_begin(BB), pred_end(BB));
}

inline llvm::iterator_range<llvm::Interval::succ_iterator>
successors(llvm::Interval *BB) {
  return make_range(succ_begin(BB), succ_end(BB));
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

/// \brief Return a tuple of \p V's operands of the requested types
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

/// \brief Checks the instruction type and its operands
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

/// \brief Trait to wrap an object of type C that can act as a blacklist for B
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
  bool isBlacklisted(B Value) const { return this->Obj.count(Value) != 0; }
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

    struct WorkItem {
      WorkItem(BasicBlock *BB, instruction_iterator Start) :
        BB(BB),
        Range(make_range(Start, ID::end(BB))) {}

      WorkItem(BasicBlock *BB) :
        BB(BB),
        Range(make_range(ID::begin(BB), ID::end(BB))) {}

      BasicBlock *BB;
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
            if (Visited.count(Successor) == 0) {
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

/// \brief Return a sensible name for the given basic block
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

/// \brief Return a sensible name for the given instruction
/// \return the name of the instruction, if available, a
///         [basic blockname]:[instruction index] string otherwise.
inline std::string getName(const llvm::Instruction *I) {
  llvm::StringRef Result = I->getName();
  if (!Result.empty()) {
    return Result.str();
  } else {
    const llvm::BasicBlock *Parent = I->getParent();
    return getName(Parent) + ":"
           + std::to_string(1
                            + std::distance(Parent->begin(), I->getIterator()));
  }
}

/// \brief Return a sensible name for the given Value
/// \return if \p V is an Instruction, call the appropriate getName function,
///         otherwise return a pointer to \p V.
inline std::string getName(const llvm::Value *V) {
  if (V != nullptr)
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
      return getName(I);
  std::stringstream SS;
  SS << "0x" << std::hex << intptr_t(V);
  return SS.str();
}

template<typename T>
using rc_t = typename std::remove_const<T>::type;
template<typename T>
using is_base_of_value = std::is_base_of<llvm::Value, rc_t<T>>;

/// \brief Specialization of writeToLog for llvm::Value-derived types
template<typename T,
         typename std::enable_if<is_base_of_value<T>::value, int>::type = 0>
inline void writeToLog(Logger<true> &This, T *I, int) {
  if (I != nullptr)
    This << getName(I);
  else
    This << "nullptr";
}

inline llvm::LLVMContext &getContext(const llvm::Module *M) {
  return M->getContext();
}

inline llvm::LLVMContext &getContext(const llvm::Function *F) {
  return getContext(F->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::BasicBlock *BB) {
  return getContext(BB->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::Instruction *I) {
  return getContext(I->getParent());
}

inline llvm::LLVMContext &getContext(const llvm::Value *I) {
  return getContext(llvm::cast<const llvm::Instruction>(I));
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

/// \brief Helper class to easily create and use LLVM metadata
class QuickMetadata {
public:
  QuickMetadata(llvm::LLVMContext &Context) :
    C(Context),
    Int32Ty(llvm::IntegerType::get(C, 32)),
    Int64Ty(llvm::IntegerType::get(C, 64)) {}

  llvm::MDString *get(const char *String) {
    return llvm::MDString::get(C, String);
  }

  llvm::MDString *get(llvm::StringRef String) {
    return llvm::MDString::get(C, String);
  }

  llvm::ConstantAsMetadata *get(llvm::Constant *C) {
    return llvm::ConstantAsMetadata::get(C);
  }

  llvm::ConstantAsMetadata *get(uint32_t Integer) {
    auto *Constant = llvm::ConstantInt::get(Int32Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::ConstantAsMetadata *get(uint64_t Integer) {
    auto *Constant = llvm::ConstantInt::get(Int64Ty, Integer);
    return llvm::ConstantAsMetadata::get(Constant);
  }

  llvm::MDNode *get() { return llvm::MDNode::get(C, {}); }

  llvm::MDTuple *tuple(const char *String) { return tuple(get(String)); }

  llvm::MDTuple *tuple(llvm::StringRef String) { return tuple(get(String)); }

  llvm::MDTuple *tuple(uint32_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(uint64_t Integer) { return tuple(get(Integer)); }

  llvm::MDTuple *tuple(llvm::ArrayRef<llvm::Metadata *> MDs) {
    return llvm::MDTuple::get(C, MDs);
  }

  template<typename T>
  T extract(const llvm::MDTuple *Tuple, unsigned Index) {
    return extract<T>(Tuple->getOperand(Index).get());
  }

  template<typename T>
  T extract(const llvm::Metadata *MD) {
    revng_abort();
  }

  template<typename T>
  T extract(llvm::Metadata *MD) {
    revng_abort();
  }

private:
  llvm::LLVMContext &C;
  llvm::IntegerType *Int32Ty;
  llvm::IntegerType *Int64Ty;
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

/// \brief Return the instruction coming before \p I, or nullptr if it's the
///        first.
inline llvm::Instruction *getPrevious(llvm::Instruction *I) {
  llvm::BasicBlock::reverse_iterator It(++I->getReverseIterator());
  if (It == I->getParent()->rend())
    return nullptr;

  return &*It;
}

/// \brief Return the instruction coming after \p I, or nullptr if it's the
///        last.
inline llvm::Instruction *getNext(llvm::Instruction *I) {
  llvm::BasicBlock::iterator It(I);
  if (It == I->getParent()->end())
    return nullptr;

  It++;
  return &*It;
}

/// \brief Check whether the instruction/basic block is the first in its
///        container or not
template<typename T>
inline bool isFirst(T *I) {
  revng_assert(I != nullptr);
  return I == &*I->getParent()->begin();
}

/// \brief Check if among \p BB's predecessors there's \p Target
inline bool hasPredecessor(llvm::BasicBlock *BB, llvm::BasicBlock *Target) {
  for (llvm::BasicBlock *Predecessor : predecessors(BB))
    if (Predecessor == Target)
      return true;
  return false;
}

static std::array<unsigned, 3> CastOpcodes = {
  llvm::Instruction::BitCast,
  llvm::Instruction::PtrToInt,
  llvm::Instruction::IntToPtr,
};

// \brief If \p V is a cast Instruction or a cast ConstantExpr, return its only
//        operand (recursively)
inline const llvm::Value *skipCasts(const llvm::Value *V) {
  using namespace llvm;
  while (isa<CastInst>(V) or isa<IntToPtrInst>(V) or isa<PtrToIntInst>(V)
         or (isa<ConstantExpr>(V)
             and contains(CastOpcodes, cast<ConstantExpr>(V)->getOpcode())))
    V = cast<User>(V)->getOperand(0);
  return V;
}

// \brief If \p V is a cast Instruction or a cast ConstantExpr, return its only
//        operand (recursively)
inline llvm::Value *skipCasts(llvm::Value *V) {
  using namespace llvm;
  while (isa<CastInst>(V) or isa<IntToPtrInst>(V) or isa<PtrToIntInst>(V)
         or (isa<ConstantExpr>(V)
             and contains(CastOpcodes, cast<ConstantExpr>(V)->getOpcode())))
    V = cast<User>(V)->getOperand(0);
  return V;
}

inline const llvm::Function *getCallee(const llvm::Instruction *I) {
  revng_assert(I != nullptr);

  using namespace llvm;
  if (auto *Call = dyn_cast<CallInst>(I))
    return llvm::dyn_cast<Function>(skipCasts(Call->getCalledValue()));
  else
    return nullptr;
}

inline llvm::Function *getCallee(llvm::Instruction *I) {
  revng_assert(I != nullptr);

  using namespace llvm;
  if (auto *Call = dyn_cast<CallInst>(I))
    return llvm::dyn_cast<Function>(skipCasts(Call->getCalledValue()));
  else
    return nullptr;
}

inline bool isCallTo(const llvm::Instruction *I, llvm::StringRef Name) {
  revng_assert(I != nullptr);
  const llvm::Function *Callee = getCallee(I);
  return Callee != nullptr && Callee->getName() == Name;
}

/// \brief Is \p I a call to an helper function?
inline bool isCallToHelper(const llvm::Instruction *I) {
  revng_assert(I != nullptr);
  const llvm::Function *Callee = getCallee(I);
  return Callee != nullptr && Callee->getName().startswith("helper_");
}

inline llvm::CallInst *getCallTo(llvm::Instruction *I, llvm::StringRef Name) {
  if (isCallTo(I, Name))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

// TODO: this function assumes 0 is not a valid PC
inline uint64_t getBasicBlockPC(llvm::BasicBlock *BB) {
  auto It = BB->begin();
  revng_assert(It != BB->end());
  if (llvm::CallInst *Call = getCallTo(&*It, "newpc")) {
    return getLimitedValue(Call->getOperand(0));
  }

  return 0;
}

template<typename C>
inline auto skip(unsigned ToSkip, C &&Container)
  -> llvm::iterator_range<decltype(Container.begin())> {

  auto Begin = std::begin(Container);
  while (ToSkip-- > 0)
    Begin++;
  return llvm::make_range(Begin, std::end(Container));
}

template<class Container, class UnaryPredicate>
inline void erase_if(Container &C, UnaryPredicate P) {
  C.erase(std::remove_if(C.begin(), C.end(), P), C.end());
}

inline std::string dumpToString(const llvm::Value *V) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  V->print(Stream, true);
  Stream.str();
  return Result;
}

inline std::string dumpToString(const llvm::Module *M) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  M->print(Stream, nullptr, false, true);
  Stream.str();
  return Result;
}

void dumpModule(const llvm::Module *M, const char *Path) debug_function;

llvm::GlobalVariable *
buildString(llvm::Module *M, llvm::StringRef String, const llvm::Twine &Name);

llvm::Constant *buildStringPtr(llvm::Module *M,
                               llvm::StringRef String,
                               const llvm::Twine &Name);

llvm::Constant *getUniqueString(llvm::Module *M,
                                llvm::StringRef Namespace,
                                llvm::StringRef String,
                                const llvm::Twine &Name = llvm::Twine());

#endif // IRHELPERS_H
