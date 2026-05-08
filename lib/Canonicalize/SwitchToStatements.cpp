//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <functional>
#include <iterator>
#include <map>
#include <memory_resource>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/TypeSize.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Canonicalize/SwitchToStatements.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DecompilationHelpers.h"
#include "revng/Support/IRBuilder.h"

static Logger Log{ "switch-to-statements" };

using namespace llvm;

//
// Templated using for types of local variables, copy and assign instructions.
// These are only necessary because we still have the legacy version.
// We can drop the template when we drop legacy mode, and just use StoreInst for
// AssignType, LoadInst for CopyType, and AllocaInst for LocalVarType.
//

template<bool IsLegacy>
using AssignType = std::conditional_t<IsLegacy, CallInst, StoreInst>;

template<bool IsLegacy>
using CopyType = std::conditional_t<IsLegacy, CallInst, LoadInst>;

template<bool IsLegacy>
using LocalVarType = std::conditional_t<IsLegacy, CallInst, AllocaInst>;

//
// Templated helpers for getting pointer and value operands for copy and assign
// instructions.
// These are only necessary because we still have the legacy version where a
// copy is not just a LoadInst but a call to an opaque function, and an
// assignment is not just a StoreInst but it's also a call to another opaque
// function.
// We can drop the template when we drop legacy mode, and just remove these
// helpers or at lease greatly simplify them.
//

template<bool IsLegacy>
static Use *getStorePointerOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(1);
  } else {
    if (auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(1);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getStorePointerOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(1);
  } else {
    if (const auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(1);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getStorePointerOperand(Instruction *I) {
  Use *U = getStorePointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getStorePointerOperand(const Instruction *I) {
  const Use *U = getStorePointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static bool isStorePointerOperand(const Use &U, const Instruction *I) {
  return getStorePointerOperandUse<IsLegacy>(I) == &U;
}

template<bool IsLegacy>
static Use *getStoreValueOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(0);
  } else {
    if (auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getStoreValueOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Assign))
      return &Assign->getArgOperandUse(0);
  } else {
    if (const auto *Assign = dyn_cast<StoreInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getStoreValueOperand(Instruction *I) {
  Use *U = getStoreValueOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getStoreValueOperand(const Instruction *I) {
  const Use *U = getStoreValueOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static bool isStoreValueOperand(const Use &U, const Instruction *I) {
  return getStoreValueOperandUse<IsLegacy>(I) == &U;
}

template<bool IsLegacy>
static Use *getLoadPointerOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (CallInst *Assign = getCallToTagged(I, FunctionTags::Copy))
      return &Assign->getArgOperandUse(0);
  } else {
    if (auto *Assign = dyn_cast<LoadInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static const Use *getLoadPointerOperandUse(const Instruction *I) {
  if (not I)
    return nullptr;

  if constexpr (IsLegacy) {
    if (const CallInst *Assign = getCallToTagged(I, FunctionTags::Copy))
      return &Assign->getArgOperandUse(0);
  } else {
    if (const auto *Assign = dyn_cast<LoadInst>(I))
      return &Assign->getOperandUse(0);
  }
  return nullptr;
}

template<bool IsLegacy>
static Value *getLoadPointerOperand(Instruction *I) {
  Use *U = getLoadPointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static const Value *getLoadPointerOperand(const Instruction *I) {
  const Use *U = getLoadPointerOperandUse<IsLegacy>(I);
  return U ? U->get() : nullptr;
}

template<bool IsLegacy>
static Value *getPointerOperand(Instruction *I) {
  if (auto *V = getLoadPointerOperand<IsLegacy>(I))
    return V;
  if (auto *V = getStorePointerOperand<IsLegacy>(I))
    return V;
  return nullptr;
}

template<bool IsLegacy>
static const Value *getPointerOperand(const Instruction *I) {
  if (const auto *V = getLoadPointerOperand<IsLegacy>(I))
    return V;
  if (const auto *V = getStorePointerOperand<IsLegacy>(I))
    return V;
  return nullptr;
}

static llvm::TypeSize getAccessedSize(const llvm::StoreInst *Store) {
  llvm::DataLayout TheDataLayout = Store->getModule()->getDataLayout();
  llvm::Type *ValueType = Store->getValueOperand()->getType();
  return TheDataLayout.getTypeStoreSize(ValueType);
}

//
// Helpers for statements and side effects.
//

static bool causesExponentialDataflowPaths(const Instruction *I) {
  // This forces all various kinds of instructions to get their value stored
  // into a local variable.
  // The reason for doing this is that these instructions often contribute to
  // the formation of pathological dataflows, causing exponential path explosion
  // when expanded into C expressions during decompilation.
  // Serializing their value into a dedicated local variable breaks such an
  // exponential path explosion.
  //
  // This is a temporary workaround, that we've already put in place for
  // SelectInst too, and that will be replaced in the future by a more
  // principled approach for splitting pathological dataflows that lead to
  // exponential path esplosion.

  if (isa<SelectInst>(I))
    return true;

  llvm::Value *Op0 = nullptr;
  llvm::Value *Op1 = nullptr;
  llvm::Value *Op2 = nullptr;

  using namespace llvm::PatternMatch;
  if (match(I, m_FShl(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;
  if (match(I, m_FShr(m_Value(Op0), m_Value(Op1), m_Value(Op2))))
    return true;

  return false;
}

static bool doesNotAccessMemory(const Instruction &I) {
  // We have to hardcode revng_call_stack_arguments and revng_stack_frame
  // because SegregateStackAccesses has to mark them as functions that read
  // inaccessible memory, in order to prevent some LLVM optimizations.
  // Same for OpaqueExtractValue.
  if (auto *Call = dyn_cast<CallInst>(&I)) {
    if (llvm::Function *Callee = getCalledFunction(Call)) {
      StringRef Name = Callee->getName();
      if (Name.startswith("revng_call_stack_arguments")
          or Name.startswith("revng_stack_frame")) {
        return true;
      }
    }
    if (getCallToTagged(&I, FunctionTags::OpaqueExtractValue))
      return true;
    if (getCallToTagged(&I, FunctionTags::StructInitializer))
      return true;
  }
  return false;
}

static bool mayHaveSideEffects(const Instruction *I) {
  // TODO: this is a workaround. This should be split up in a separate pass to
  // run before SwitchToStatements.
  if (causesExponentialDataflowPaths(I))
    return true;

  if (doesNotAccessMemory(*I))
    return false;

  return I->mayHaveSideEffects();
}

static bool mayWriteToMemory(const Instruction &I) {
  if (doesNotAccessMemory(I))
    return false;

  return I.mayWriteToMemory();
}

static bool mayReadMemory(const Instruction &I) {
  if (doesNotAccessMemory(I))
    return false;

  return I.mayReadFromMemory();
}

//
// Legacy mode helpers for traversing ModelGEPs and discovering the local
// variable accessed by an Instruction.
// They can be dropped when legacy mode goes away.
//

static RecursiveCoroutine<std::optional<const Value *>>
getAccessedLocalVariableFromModelGEP(const CallInst *ModelGEPRefCall) {
  revng_assert(isCallToTagged(ModelGEPRefCall, FunctionTags::ModelGEPRef));

  revng_assert(ModelGEPRefCall->arg_size() >= 2);

  // If the ModelGEPRefCall has more than 2 arguments, and some of them are not
  // constants, we cannot figure out all the list of potentially accessed local
  // variables, so we just return nullptr.
  for (const Use &GEPArg : llvm::drop_begin(ModelGEPRefCall->args(), 2)) {
    if (not isa<Constant>(GEPArg.get()))
      rc_return nullptr;
  }

  // If the Base argument of the ModelGEPRefCall isn't a LocalVariable, nor an
  // Argument, nor another ModelGEPRef, we just return nullopt, meaning that
  // this thing doesn't really access any local variable.
  auto *GEPBase = ModelGEPRefCall->getArgOperand(1);
  // If the GEPBase is directly an argument, we're done
  if (isa<Argument>(GEPBase))
    rc_return GEPBase;

  // If the GEPBase is directly a LocalVariable, we're done
  if (isCallToTagged(GEPBase, FunctionTags::AllocatesLocalVariable))
    rc_return GEPBase;

  // If the GEPBase is another ModelGEPRef we recur.
  // Notice that we don't recur on ModelGEP, only on ModelGEPRef, because simple
  // ModelGEP can have arbitrary base pointers, but they never access
  // LocalVariables.
  if (auto *NestedModelGEPRef = getCallToTagged(GEPBase,
                                                FunctionTags::ModelGEPRef))
    rc_return rc_recur getAccessedLocalVariableFromModelGEP(NestedModelGEPRef);

  // Everything else cannot access local variables, so we return nullopt.
  rc_return std::nullopt;
}

// Get the local variable accessed by I.
// If the returned optional is nullopt, it means that I doesn't accessy memory.
// If the returned optional is engaged:
// - if it holds a null pointer, it means that I accesses memory but we weren't
//   able to figure out where
// - if it holds a valid pointer, it must be either an Argument or the accessed
//   local variable
static std::optional<const Value *>
getAccessedLegacyLocal(const Instruction *I) {
  const CopyType<true> *Copy = getCallToTagged(I, FunctionTags::Copy);
  const AssignType<true> *Assign = getCallToTagged(I, FunctionTags::Assign);

  // If it's not a Copy not an Assign then it's not an access to a local
  // variable.
  // TODO: this doesn't take into consideration stuff like memcpy.
  if (not Copy and not Assign)
    return std::nullopt;

  const CallInst *AccessCall = Copy ? Copy : Assign;

  unsigned AccessArgumentNumber = Assign ? 1 : 0;
  const auto *Accessed = AccessCall->getArgOperand(AccessArgumentNumber);

  // If the accessed thing is directly an Argument or a LocalVariable we're
  // done.
  if (isa<Argument>(Accessed)
      or isCallToTagged(Accessed, FunctionTags::AllocatesLocalVariable)) {
    return Accessed;
  }

  // If the accessed thing is not a ModelGEPRef, then it's not an access to a
  // local variable.
  auto *ModelGEPRef = getCallToTagged(Accessed, FunctionTags::ModelGEPRef);
  if (not ModelGEPRef)
    return std::nullopt;

  return getAccessedLocalVariableFromModelGEP(ModelGEPRef);
}

/// A class that represents an available expression, along with the assignment
/// that writes its value somewhere, making it available.
template<bool IsLegacy>
struct AvailableExpression {
  using AssignType = AssignType<IsLegacy>;

  // The expression that is available
  Instruction *Expression = nullptr;

  // The Assign/Store that has assigned the Expression to some location.
  // It can be used to retrieve the address of the location itself.
  // nullptr means that we don't have a specific address but the Expression
  // itself can be computed at the given program point without breaking
  // semantics.
  AssignType *Assignment = nullptr;

  bool operator==(const AvailableExpression &) const = default;
  std::strong_ordering operator<=>(const AvailableExpression &) const = default;
};

template<bool IsLegacy>
using AvailableSet = std::set<AvailableExpression<IsLegacy>>;

template<bool IsLegacy>
static auto
findAvailableRange(const AvailableSet<IsLegacy> &Availables, Instruction *I) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  auto Begin = Availables.lower_bound(AvailableExpression{
    .Expression = I, .Assignment = nullptr });
  auto End = Availables.upper_bound(AvailableExpression{
    .Expression = std::next(I), .Assignment = nullptr });
  return llvm::make_range(Begin, End);
}

constexpr size_t SmallSize = 8;
using InstructionVector = SmallVector<Instruction *, SmallSize>;
using InstructionSetVector = SmallSetVector<Instruction *, SmallSize>;

struct ProgramPointData {
  Instruction *TheInstruction = nullptr;
  ProgramPointData(Instruction *I) : TheInstruction(I){};
};

using ProgramPointNode = BidirectionalNode<ProgramPointData>;
using ProgramPointsCFG = GenericGraph<ProgramPointNode>;

template<bool IsLegacy>
struct AvailableExpressionsMonotoneFramework;

template<bool IsLegacy>
struct AvailableExpressionsMonotoneFramework {
public:
  using GraphType = ProgramPointsCFG *;
  using LatticeElement = AvailableSet<IsLegacy>;
  using Label = ProgramPointNode *;

private:
  AliasAnalysis *AA;

public:
  AvailableExpressionsMonotoneFramework(AliasAnalysis *A) : AA(A) {}

public:
  LatticeElement combineValues(const LatticeElement &LHS,
                               const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::combineValues(LHS, RHS);
  }

  bool isLessOrEqual(const LatticeElement &LHS,
                     const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::isLessOrEqual(LHS, RHS);
  }

  LatticeElement applyTransferFunction(ProgramPointNode *L,
                                       const LatticeElement &E) const;

private:
  void applyTransferFunctionImpl(Instruction *I, LatticeElement &E) const;

  bool mayClobber(Instruction *Access, Instruction *Affected) const;
};

template<bool IsLegacy>
using AEMFP = AvailableExpressionsMonotoneFramework<IsLegacy>;

template<bool IsLegacy>
using LatticeElement = AEMFP<IsLegacy>::LatticeElement;

template<bool IsLegacy>
using AvailableExpressionsMap = MFP::MFIResultMap<AEMFP<IsLegacy>>;

static bool legacyLocalVariablesNoAlias(const Instruction *I,
                                        const Instruction *J) {

  // Copies from local variables never alias anyone else, except other
  // instructions that copy or assign the same local variable
  std::optional<const Value *> MayBeAccessedByI = getAccessedLegacyLocal(I);
  std::optional<const Value *> MayBeAccessedByJ = getAccessedLegacyLocal(J);

  // If either doesn't access a local variable, they are noAlias.
  if (not MayBeAccessedByI.has_value() or not MayBeAccessedByJ.has_value())
    return true;

  const Value *AccessedByI = *MayBeAccessedByI;
  const Value *AccessedByJ = *MayBeAccessedByJ;

  // If either is nullptr, there is at least one among I and J that access many
  // variables, and we just can't say with certainty that they are noAlias
  if (nullptr == AccessedByI or nullptr == AccessedByJ)
    return false;

  // If both are arguments, they may point overlapping memory, and we have no
  // way of knowing. So we return false, because we're not sure they don't
  // alias.
  if (isa<Argument>(AccessedByI) and isa<Argument>(AccessedByJ))
    return false;

  // For all the other cases they are noAlias only if the accessed
  // local variable is different.
  return AccessedByI != AccessedByJ;
}

template<bool IsLegacy>
bool AEMFP<IsLegacy>::mayClobber(Instruction *Access,
                                 Instruction *Affected) const {

  revng_log(Log, "mayClobber");
  LoggerIndent Indent{ Log };
  revng_log(Log, "Access: " << dumpToString(Access));
  revng_log(Log, "Affected: " << dumpToString(Affected));
  LoggerIndent MoreIndent{ Log };

  bool Result = true;
  if constexpr (IsLegacy) {

    // If either instruction doesn't access memory, they are noAlias for sure.
    if (not Access->mayReadOrWriteMemory()) {
      revng_log(Log, "Access->mayReadOrWriteMemory() == false");
      Result = false;
    } else if (not Affected->mayReadOrWriteMemory()) {
      revng_log(Log, "Affected->mayReadOrWriteMemory() == false");
      return false;
    } else {
      // Here both instructions access memory.

      // Just handle LocalVariables specifically.
      // TODO: this is a poor's man alias analysis, which only explicitly
      // handles stuff that is frequent and that we care about. In the future we
      // have plans to replace it with a full fledged AliasAnalysis from LLVM
      Result = not legacyLocalVariablesNoAlias(Access, Affected);
    }
  } else {
    Result = isModSet(AA->getModRefInfo(Access, Affected));
  }
  revng_log(Log,
            "Access may " << (Result ? std::string() : std::string("not "))
                          << "clobber Affected");
  return Result;
}

template<bool IsLegacy>
void AEMFP<IsLegacy>::applyTransferFunctionImpl(Instruction *I,
                                                LatticeElement &E) const {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  revng_log(Log, "applyTransferFunction on Instruction I: " << dumpToString(I));
  LoggerIndent Indent{ Log };

  if constexpr (IsLegacy) {
    revng_assert(not isa<LoadInst>(I) and not isa<StoreInst>(I));
  } else {
    revng_assert(not isCallToTagged(I, FunctionTags::Copy)
                 and not isCallToTagged(I, FunctionTags::Assign));
  }

  if (mayHaveSideEffects(I)) {
    revng_log(Log, "mayHaveSideEffects");
    LoggerIndent XX{ Log };
    for (const AvailableExpression &A : llvm::make_early_inc_range(E)) {
      const auto &[Available, Assign] = A;
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
      LoggerIndent XXX{ Log };
      if (mayClobber(I, Available)) {
        revng_log(Log, "I may clobber Available: " << dumpToString(Available));
        revng_log(Log, "erase Available");
        E.erase(A);
      } else if (Assign and mayClobber(I, Assign)) {
        revng_log(Log, "I may clobber Assign: " << dumpToString(Assign));
        revng_log(Log, "erase Available");
        E.erase(A);
      }
    }
  }

  auto *StoredOperand = getStoreValueOperand<IsLegacy>(I);
  auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
  if (AssignedInstruction) {
    revng_log(Log, "I is Assign");
    revng_log(Log, "insert Available: " << dumpToString(AssignedInstruction));
    revng_log(Log, "       Assign: " << dumpToString(I));

    E.insert(AvailableExpression{
      .Expression = AssignedInstruction,
      .Assignment = cast<AssignType>(I),
    });
  }

  if (mayReadMemory(*I)) {
    revng_log(Log, "mayReadMemory -> insert Available: I");
    E.insert(AvailableExpression{
      .Expression = I,
      .Assignment = nullptr,
    });
  }
}

template<bool IsLegacy>
AEMFP<IsLegacy>::LatticeElement
AEMFP<IsLegacy>::applyTransferFunction(ProgramPointNode *ProgramPoint,
                                       const AEMFP<IsLegacy>::LatticeElement &E)
  const {

  Instruction *I = ProgramPoint->TheInstruction;

  revng_log(Log, "applyTransferFunction on ProgramPoint: " << dumpToString(I));
  LoggerIndent Indent{ Log };

  LatticeElement Result = E;

  revng_log(Log, "initial set");
  if (Log.isEnabled()) {
    LoggerIndent ModeIndent{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  applyTransferFunctionImpl(I, Result);

  revng_log(Log, "final set");
  if (Log.isEnabled()) {
    LoggerIndent ModeIndent{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  return Result;
}

//
// Helpers for identifying program points that are relevant for available
// expressions.
//

template<bool IsLegacy>
static bool isProgramPoint(const Instruction *I) {

  const Instruction *UnexpectedInstruction = nullptr;
  if constexpr (IsLegacy) {
    // Legacy mode just assumes that we don't have Load/Store/Alloca at all.
    if (isa<LoadInst>(I) or isa<StoreInst>(I) or isa<AllocaInst>(I)
        or isa<PHINode>(I))
      UnexpectedInstruction = I;
  } else {
    // Non-legacy mode assumes that most custom opcode don't exist. Some of them
    // have been replaced by Load/Store/Alloca, and others have been dropped
    // because in the clift-based pipeline they will be only materialized in
    // Clift as regular operators, so we don't need them in LLVM anymore and we
    // want to make sure they disappear over time until we can actually drop
    // them.
    if (isCallToTagged(I, FunctionTags::AllocatesLocalVariable)
        or isCallToTagged(I, FunctionTags::LocalVariable)
        or isCallToTagged(I, FunctionTags::Copy)
        or isCallToTagged(I, FunctionTags::Assign)
        or isCallToTagged(I, FunctionTags::AddressOf)
        or isCallToTagged(I, FunctionTags::Marker)
        or isCallToTagged(I, FunctionTags::IsRef)
        or isCallToTagged(I, FunctionTags::StringLiteral)
        or isCallToTagged(I, FunctionTags::ModelCast)
        or isCallToTagged(I, FunctionTags::ModelGEP)
        or isCallToTagged(I, FunctionTags::ModelGEPRef)
        or isCallToTagged(I, FunctionTags::Parentheses)
        or isCallToTagged(I, FunctionTags::LiteralPrintDecorator)
        or isCallToTagged(I, FunctionTags::HexInteger)
        or isCallToTagged(I, FunctionTags::CharInteger)
        or isCallToTagged(I, FunctionTags::BoolInteger)
        or isCallToTagged(I, FunctionTags::NullPtr)
        or isCallToTagged(I, FunctionTags::SegmentRef)
        or isCallToTagged(I, FunctionTags::UnaryMinus)
        or isCallToTagged(I, FunctionTags::BinaryNot)
        or isCallToTagged(I, FunctionTags::BooleanNot)) {
      UnexpectedInstruction = I;
    }
    // We also don't expect PHINodes, since SwitchToStatements is designed to be
    // the last pass in the decompilation pipeline that injects local variables.
    if (isa<PHINode>(I))
      UnexpectedInstruction = I;
  }

  if (nullptr != UnexpectedInstruction) {
    I->dump();
    revng_abort("Unexpected Instruction");
  }

  return I == &I->getParent()->front() or mayHaveSideEffects(I)
         or mayReadMemory(*I);
}

template<bool IsLegacy>
static InstructionSetVector getProgramPoints(BasicBlock &B) {
  InstructionSetVector Results;
  for (Instruction &I : B)
    if (isProgramPoint<IsLegacy>(&I))
      Results.insert(&I);
  return Results;
}

using InstructionProgramPoint = std::unordered_map<const Instruction *,
                                                   ProgramPointNode *>;

// An extended version of ProgramPointsCFG, that holds a graph of statements
// points, along with a map from each Instruction to its previous statement.
template<bool IsLegacy>
class AvailableExpressionsResult {
public:
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using AvailableExpressionsMap = AvailableExpressionsMap<IsLegacy>;

public:
  ProgramPointsCFG ProgramPointsGraph;
  AvailableExpressionsMap AvailableExpressions;

private:
  // Map an Instruction to its associated program point in ProgramPointsGraph
  InstructionProgramPoint ProgramPoint;

  // Map an Instruction to its associated previous program point in
  // ProgramPointsGraph.
  InstructionProgramPoint PreviousProgramPointInBlock;

  // Map an Instruction to its associated next program point in
  // ProgramPointsGraph.
  InstructionProgramPoint NextProgramPointInBlock;

public:
  // Factory from llvm::Function
  static AvailableExpressionsResult makeFromFunction(Function &F) {

    SmallMap<BasicBlock *, std::pair<ProgramPointNode *, ProgramPointNode *>, 8>
      BlockToBeginEndNode;

    AvailableExpressionsResult Result;

    ProgramPointsCFG &TheCFG = Result.ProgramPointsGraph;
    InstructionProgramPoint &ProgramPoint = Result.ProgramPoint;
    InstructionProgramPoint
      &PreviousProgramPointInBlock = Result.PreviousProgramPointInBlock;
    InstructionProgramPoint
      &NextProgramPointInBlock = Result.NextProgramPointInBlock;

    const auto MakeCFGNode = [&TheCFG, &ProgramPoint](Instruction *I) {
      ProgramPointNode *NewNode = TheCFG.addNode(I);
      ProgramPoint[I] = NewNode;
      return NewNode;
    };

    for (BasicBlock &BB : F) {
      InstructionSetVector ProgramPoints = getProgramPoints<IsLegacy>(BB);

      // Reserve space for the new ProgramPoints. This is for performance but
      // also for stability of pointers while adding new nodes, which allows to
      // also save pointers to begin and end nodes of each block in a map, to
      // handle addition of inter-block edges. If we don't reserve the pointers
      // returned by addNode aren't stable and the trick for adding inter-block
      // edges doesn't work.
      TheCFG.reserve(TheCFG.size() + ProgramPoints.size());

      ProgramPointNode *FirstNode = MakeCFGNode(ProgramPoints.front());
      ProgramPointNode *LastNode = FirstNode;
      for (Instruction &I :
           llvm::make_range(BB.begin(), ProgramPoints.front()->getIterator()))
        NextProgramPointInBlock[&I] = LastNode;

      auto ProgramPointPairs = llvm::zip_equal(llvm::drop_end(ProgramPoints),
                                               llvm::drop_begin(ProgramPoints));
      for (const auto &[PreviousProgramPoint, NextProgramPoint] :
           ProgramPointPairs) {
        // Create a new node.
        ProgramPointNode *NewNode = MakeCFGNode(NextProgramPoint);
        // We can already add intra-block edges.
        LastNode->addSuccessor(NewNode);

        // Now we have to initialize PreviousProgramPointInBlock for all the
        // instructions that are not program points and that are among the
        // previous program point and the current new one.
        for (Instruction &I :
             llvm::make_range(std::next(PreviousProgramPoint->getIterator()),
                              NextProgramPoint->getIterator()))
          PreviousProgramPointInBlock[&I] = LastNode;

        // Finally we can update the LastNode.
        LastNode = NewNode;
      }
      for (Instruction &I :
           llvm::make_range(std::next(ProgramPoints.back()->getIterator()),
                            BB.end()))
        PreviousProgramPointInBlock[&I] = LastNode;

      BlockToBeginEndNode[&BB] = { FirstNode, LastNode };
    }

    // Now we add the inter-block edges.
    for (BasicBlock &BB : F)
      for (BasicBlock *Successor : llvm::successors(&BB))
        BlockToBeginEndNode.at(&BB)
          .second->addSuccessor(BlockToBeginEndNode.at(Successor).first);

    // And set the entry node, which makes the MFP later more efficient, because
    // it allows the algorithm to take the structure of the graph into account,
    // instead of iterating in sparse order.
    TheCFG.setEntryNode(BlockToBeginEndNode.at(&F.getEntryBlock()).first);

    return Result;
  }

public:
  auto getAvailableAt(Instruction *I, const Instruction *Where) const {

    revng_log(Log, "IsAvailableAt");
    revng_log(Log, "Available?: " << dumpToString(I));
    revng_log(Log, "Where: " << dumpToString(Where));

    auto ProgramPointIt = ProgramPoint.find(Where);
    if (ProgramPointIt != ProgramPoint.end()) {
      revng_log(Log, "is ProgramPoint");

      ProgramPointNode *UserProgramPoint = ProgramPointIt->second;
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    auto PreviousPointIt = PreviousProgramPointInBlock.find(Where);
    if (PreviousPointIt != PreviousProgramPointInBlock.end()) {
      revng_log(Log, "is NOT ProgramPoint");

      ProgramPointNode *UserProgramPoint = PreviousPointIt->second;
      revng_log(Log,
                "Previous ProgramPoint: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .OutValue;
      return findAvailableRange(Available, I);
    }

    auto NextProgramPointIt = NextProgramPointInBlock.find(Where);
    if (NextProgramPointIt != NextProgramPointInBlock.end()) {
      revng_log(Log, "is before first ProgramPoint in BasicBlock");

      ProgramPointNode *UserProgramPoint = NextProgramPointIt->second;
      revng_log(Log,
                "first ProgramPoint in BasicBlock: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    revng_abort();
  }

  auto getAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return getAvailableAt(I, UserInstruction);
  }

  bool isAvailableAt(Instruction *I, const Instruction *Where) const {
    bool Result = not getAvailableAt(I, Where).empty();
    revng_log(Log, "Result: " << Result);
    return Result;
  }

  bool isAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return isAvailableAt(I, UserInstruction);
  }
};

template<bool IsLegacy>
using AEResult = AvailableExpressionsResult<IsLegacy>;

template<bool IsLegacy>
static AEResult<IsLegacy>
getAvailableExpressions(Function &F, AliasAnalysis *AA) {
  revng_log(Log, "getAvailableExpressions: " << F.getName());

  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  auto Result = AEResult<IsLegacy>::makeFromFunction(F);

  AvailableSet Bottom;
  for (ProgramPointNode *N : llvm::nodes(&Result.ProgramPointsGraph)) {
    Instruction *I = N->TheInstruction;

    if (mayReadMemory(*I)) {
      Bottom.insert(AvailableExpression{
        .Expression = I,
        .Assignment = nullptr,
      });
    }

    auto *StoredOperand = getStoreValueOperand<IsLegacy>(I);
    auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
    if (AssignedInstruction) {
      Bottom.insert(AvailableExpression{
        .Expression = AssignedInstruction,
        .Assignment = cast<AssignType>(I),
      });
    }
  }

  AvailableSet Empty{};
  ProgramPointsCFG *Graph = &Result.ProgramPointsGraph;
  ProgramPointNode *Entry = Graph->getEntryNode();

  AEMFP<IsLegacy> AvailableExpressionsMF{ AA };
  // std::exchange here is only needed to make revng check-conventions happy.
  std::exchange(Result.AvailableExpressions,
                MFP::getMaximalFixedPoint<>(AvailableExpressionsMF,
                                            Graph,
                                            Bottom,
                                            Empty,
                                            { Entry }));
  return Result;
}

template<bool IsLegacy>
struct PickedInstructions {
  SetVector<Instruction *> ToSerialize = {};
  MapVector<Use *, AssignType<IsLegacy> *> ToReplaceWithAvailable = {};
  SmallPtrSet<AssignType<IsLegacy> *, 8> AssignToRemove = {};
};

template<bool IsLegacy>
class AvailableExpressionsAnalysis
  : public llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis<IsLegacy>> {

  friend llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis<IsLegacy>>;
  static llvm::AnalysisKey Key;

public:
  using Result = AvailableExpressionsResult<IsLegacy>;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    AliasAnalysis *AA = IsLegacy ? nullptr : &FAM.getResult<AAManager>(F);
    return getAvailableExpressions<IsLegacy>(F, AA);
  }
};

template<>
AnalysisKey AvailableExpressionsAnalysis<true>::Key = {};
template<>
AnalysisKey AvailableExpressionsAnalysis<false>::Key = {};

template<bool IsLegacy>
using AEA = AvailableExpressionsAnalysis<IsLegacy>;

template<bool IsLegacy>
class InstructionToSerializePicker
  : public AnalysisInfoMixin<InstructionToSerializePicker<IsLegacy>> {
  friend llvm::AnalysisInfoMixin<InstructionToSerializePicker<IsLegacy>>;
  static llvm::AnalysisKey Key;

public:
  using Result = PickedInstructions<IsLegacy>;
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

private:
  const AEResult<IsLegacy> *AvailableExpressions = nullptr;
  std::unordered_map<const Instruction *, size_t> ProgramOrdering = {};
  Result Picked;
  AliasAnalysis *AA;

public:
  InstructionToSerializePicker() :
    AvailableExpressions(nullptr), ProgramOrdering() {}

public:
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    if constexpr (not IsLegacy) {
      AA = &FAM.getResult<AAManager>(F);
    }
    AvailableExpressions = &FAM.getResult<AEA<IsLegacy>>(F);

    Picked = {};
    ProgramOrdering = {};

    return pick(F);
  }

private:
  bool isSerializable(const Instruction &I) const {
    const llvm::Type *T = I.getType();
    if constexpr (IsLegacy) {
      return not T->isVoidTy() and not T->isAggregateType()
             and not isCallToTagged(&I, FunctionTags::IsRef);
    } else {
      return not T->isVoidTy();
    }
  }

  void pick(llvm::Instruction *I) {
    LoggerIndent Indent{ Log };
    revng_log(Log, "pick(I), I: " << dumpToString(I));
    Picked.ToSerialize.insert(I);
  };

  bool isPicked(llvm::Instruction *I) const {
    return Picked.ToSerialize.contains(I);
  }

  Result pick(Function &F) {
    revng_log(Log, "pick: " << F.getName().str());
    LoggerIndent Indent{ Log };

    // Visit in RPO for determinism
    const auto RPO = llvm::ReversePostOrderTraversal(&F);
    size_t NextOrder = 0;
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        ProgramOrdering[&I] = NextOrder++;

    // First, pick all the statements amenable for serialization
    // Also compute the program order of instructions.
    if constexpr (IsLegacy) {
      revng_log(Log,
                "pick for serialization instructions with mayHaveSideEffects: "
                  << F.getName().str());
      LoggerIndent MoreIndent{ Log };

      for (BasicBlock *BB : RPO) {
        for (Instruction &I : *BB) {
          // If it's not a statement, don't pick it.
          if (not mayHaveSideEffects(&I))
            continue;

          // If it's a statement but it's not serializable, don't pick it.
          if (not isSerializable(I))
            continue;

          // If I is a call that reads and writes memory, but has only one use,
          // we may want to not serialize it and inline it in the use. This will
          // probably happen often. Any such call can be modeled as a write
          // followed by a read, such that the write aliases everything except
          // local variables that don't escape. Inlining it would have the
          // effect of postponing a write, which is something that we haven't
          // ever considered to do.
          pick(&I);
        }
      }
    }

    // Then, start from memory reads, and traverse the dataflow to pick other
    // instructions that need to be serialized.
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        if (mayReadMemory(I))
          pickInstructionForMemoryRead(&I, &I);

    return Picked;
  }

  bool assignSameLocation(const AssignType *X, const AssignType *Y)
    requires(not IsLegacy)
  {

    if (getAccessedSize(X) != getAccessedSize(Y))
      return false;

    const Value *XPointer = getStorePointerOperand<false>(X);
    const Value *YPointer = getStorePointerOperand<false>(Y);
    return AA->isMustAlias(XPointer, YPointer);
  }

  RecursiveCoroutine<void>
  pickInstructionForMemoryRead(Instruction *I, Instruction *MemoryRead) {
    revng_log(Log, "PickFrom I: " << dumpToString(I));
    revng_log(Log, "MemoryRead: " << dumpToString(MemoryRead));

    LoggerIndent Indent{ Log };

    // If I has already been picked we'de done.
    if (isPicked(I)) {
      revng_log(Log, "I isPicked");
      rc_return;
    }

    if constexpr (IsLegacy) {
      // If it has side effects, we must have already picked it, unless it's not
      // serializable. Just return.
      if (mayHaveSideEffects(I)) {
        revng_log(Log, "mayHaveSideEffects(I)");
        revng_assert(not isSerializable(*I));
        rc_return;
      }
      revng_log(Log, "not mayHaveSideEffects(I)");
    } else {
      // If MemoryRead may have side effects, it means that it's a call that
      // accesses memory in read+write fashion. If that happens, in principle we
      // don't have a strong case for picking I. However, if I has more than one
      // use, and we don't pick I, the MemoryRead (which is a call) will end up
      // duplicated in the full expression for I, which is not guaranteed to
      // break semantic. So, whenever MemoryRead may have side effects and I has
      // more than a single use, we have to pick I for preserving semantic.
      if (mayHaveSideEffects(MemoryRead) and I->getNumUses() > 1) {
        revng_log(Log, "I may have side effects, and has many uses");
        pick(I);
        rc_return;
      }
    }

    revng_log(Log, "Check users");
    LoggerIndent UserIndent{ Log };

    MapVector<Use *, AssignType *> ToReplaceWithAvailable;
    SmallPtrSet<AssignType *, 8> AssignToRemove;
    SmallVector<Use *> UsesToRecurOn;

    // If we're here, I hasn't been picked for serialization yet.
    // If possible we want to avoid picking it, because our broader goal is to
    // serialize as few instructions as possible.
    //
    // We can avoid serializing I if one of the following applies.
    //
    // 1.
    // MemoryRead is available directly in all uses of I. In this case I can be
    // emitted inline in each of its uses, and MemoryRead will be inlined along
    // with I in every use of I, without breaking semantics.
    // We can avoid serializing I, and recur.
    //
    // 2.
    // For each use U of I where MemoryRead is not available directly,
    // MemoryRead is available indirectly via some assignment S that stored it
    // somewhere.
    // In this case, for each of such use U, we want to make sure that if we
    // inline I in U we will not use MemoryRead directly, but replace the
    // transitive use of MemoryRead in I with a load from the same location as
    // S.
    // However, in practice we have a guarantee that this holds, thanks to
    // recursion. Indeed, if we're reached this place via recursion we must have
    // already processed the transitive use of MemoryRead in I, and we have
    // solved that part of the problem earlier. This means can be sure that
    // whenever we emit I, if MemoryRead is not available directly at I but only
    // via an assignment S, we have already set up things so that we will use a
    // load from S instead of MemoryRead directly to emit I.
    // So we can avoid serializing I, and recur.
    //
    // 3. If MemoryRead is not available at U we check if I itself is available
    // at U via some assignment S that assigned it to some location, where the
    // value of I is still available. If I it's not available via any assignment
    // we have to bail out and pick I for serialization.
    //
    // 4. For each use U of I where MemoryRead is not available, I happens to be
    // available via some assignment S that assigned it to some location, where
    // the value of I is still available.
    //
    // If we're looking at how I is used in U, it means that we've already
    // processed how MemoryRead is used in I, hence by construction the
    // expression that will be emitted in S to compute the value of I will
    // already use the correct value of MemoryRead, either directly or from
    // another assignment T that assigns before I and makes it available
    // indirectly at I.
    //
    // As a result, we just mark the use U to be replaced with the value of I
    // available at S.

    // For each U Use of I where MemoryRead is not available, check if the
    // whole expression represented by I is available at U. If so add it to
    // the ToReplaceWithAvailable.
    // Otherwise if we find even a single use of I where MemoryRead is not
    // available and such that I itself is not available, we have to require I
    // to be either be available somewhere else or be serialized in a new local
    // variable.
    for (Use &U : I->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      revng_log(Log,
                "UseNo: " << U.getOperandNo()
                          << " User: " << dumpToString(User));
      LoggerIndent MoreUserIndent{ Log };

      // Skip over the Assign operand representing target variables for
      // assignments, because we need to preserve them.
      if (isStorePointerOperand<IsLegacy>(U, User)) {
        revng_log(Log, "isStorePointerOperand(U, User)");
        // In principle we should do:
        // UsesToRecurOn.push_back(&U);
        // But that would just do nothing because Store instructions can't have
        // any Use. So we can just continue here.
        continue;
      }

      // Case 1. and 2. of the description above.
      // If the MemoryRead is available in U either directly or via an
      // assignment S we're fine and we start looking at the next use.
      if (AvailableExpressions->isAvailableAt(MemoryRead, U)) {
        revng_log(Log, "isAvailableAt(MemoryRead, User)");
        UsesToRecurOn.push_back(&U);
        continue;
      }
      revng_log(Log, "not isAvailableAt(MemoryRead, User)");

      // The same use U of I could have already been picked in a previous
      // iteration, from a different memory read.
      // Detect that case and quickly bail out, because in practice we have
      // already taken that decision and we will reuse that.
      if (Picked.ToReplaceWithAvailable.count(&U)) {
        revng_log(Log,
                  "I was already picked and it's available at User, reading "
                  "from: "
                    << dumpToString(Picked.ToReplaceWithAvailable.lookup(&U)));
        UsesToRecurOn.push_back(&U);
        continue;
      }

      revng_log(Log, "Find where I is available");

      // Case 3. If selectAssignment fails I is not available at U via any
      // assignment, so we pick I and bail out..
      AssignType *SelectedAssign = selectAssignment(I, U);
      if (not SelectedAssign) {
        revng_log(Log, "SelectedAssign: nullptr");
        revng_log(Log, "I is not available at User via other assignments");
        rc_return pick(I);
      }
      revng_log(Log, "SelectedAssign: " << dumpToString(SelectedAssign));

      // Case 4.
      // If the user is not writing to memory, it cannot interact in any way
      // with the memory written to by SelectedAssign.
      // Just mark U to be replaced from a read from the location assigned by
      // SelectedAssign.
      if (not mayWriteToMemory(*User)) {
        ToReplaceWithAvailable[&U] = SelectedAssign;
        revng_log(Log, "not mayWriteToMemory(User)");
        UsesToRecurOn.push_back(&U);
        continue;
      }
      revng_log(Log, "mayWriteToMemory(User)");

      // The following is a workaround to avoid emitting self-assigments in C,
      // which are perfectly fine semantically but ugly to see.
      if constexpr (IsLegacy) {

        // If User is an Assign, and it assigns the same local variable as the
        // SelectedAssign, replacing U with a Copy from the LocalVar assigned by
        // SelectedAssign from Load from the variable that assigns the same
        // local variable as the SelectedAssign, would turn User into a self
        // assignment.
        if (auto *UserAssign = getCallToTagged(User, FunctionTags::Assign);
            UserAssign
            and legacyLocalVariablesNoAlias(UserAssign, SelectedAssign)) {
          AssignToRemove.insert(UserAssign);
          // In principle we should recur on this but Assign is always
          // guaranteed to have zero uses.
        } else {
          // In all the other cases it's fine to replace U with a Copy from the
          // LocalVar assigned by SelectedAssign
          ToReplaceWithAvailable[&U] = SelectedAssign;
          UsesToRecurOn.push_back(&U);
        }

      } else {

        // If User is a store, and it assigns the same location as the
        // SelectedAssign, replacing U with a load from the alloca assigned by
        // SelectedAssign would turn User into a self assignment.
        auto *UserAssign = dyn_cast<StoreInst>(User);
        if (UserAssign and assignSameLocation(UserAssign, SelectedAssign)) {
          AssignToRemove.insert(UserAssign);
          // In principle we should recur on this but Assign is always
          // guaranteed to have zero uses.
        } else {
          // In all the other cases it's fine to replace U with a Copy from the
          // LocalVar assigned by SelectedAssign
          ToReplaceWithAvailable[&U] = SelectedAssign;
          UsesToRecurOn.push_back(&U);
        }
      }
    }

    // If we reach this point, it means that no user forced us to serialize I.
    // At this point we can commit ToReplaceWithAvailable into
    // Picked.ToReplaceWithAvailable.
    for (const auto &Element : ToReplaceWithAvailable)
      Picked.ToReplaceWithAvailable.insert(Element);

    // And we can also commit the fact that we want to remove the Assign.
    for (const auto &Assign : AssignToRemove)
      Picked.AssignToRemove.insert(Assign);

    // If we reach this point I is has not been picked for serialization, and
    // MemoryRead is available to all users of I, either directly of via some
    // other local variable where the whole I is available.
    // Hence, we can safely decide that I will not be picked.
    // We still have to check all users of I that are using MemoryRead directly
    // (transitively via I). Those may still be picked.
    revng_log(Log, "Recur on users of I");
    for (Use *U : UsesToRecurOn) {

      auto *User = cast<Instruction>(U->getUser());
      LoggerIndent UserIndent{ Log };
      revng_log(Log,
                "UseNo: " << U->getOperandNo()
                          << " User: " << dumpToString(User));
      LoggerIndent MoreUserIndent{ Log };
      if (auto *Call = dyn_cast<CallInst>(User)) {
        if (Call->isArgOperand(U)) {
          revng_log(Log, "User is CallInst and User->isArgOperand(U)");
          continue;
        }
      }
      rc_recur pickInstructionForMemoryRead(User, MemoryRead);
    }
    rc_return;
  }

  /// Returns an assignment where I is available at U
  AssignType *selectAssignment(Instruction *I, Use &U) {
    // If the MemoryRead is not available at U, it may still be the case that
    // I itself is available at U, because it was stored into a pre-existing
    // alloca.
    auto Available = AvailableExpressions->getAvailableAt(I, U);
    if (Available.empty()) {
      revng_log(Log, "I is not available at User");
      return nullptr;
    }
    revng_log(Log, "I is available at User");
    if (Log.isEnabled()) {
      LoggerIndent Indent{ Log };
      for (const AvailableExpression &AE : Available) {
        revng_log(Log, "AE.Available = " << dumpToString(AE.Expression));
        revng_log(Log, "AE.Assignment = " << dumpToString(AE.Assignment));
      }
    }

    // Here we have a bunch of places where I is available in U.
    // We want to pick one, so that we will end up replacing the use of I in U
    // with a load from there.

    SmallVector<AssignType *> AssignsWhereIIsAvailable;
    llvm::copy(Available
                 | std::views::transform([](const AvailableExpression &A) {
                     return A.Assignment;
                   })
                 | std::views::filter([](const AssignType *A) {
                     return A != nullptr;
                   }),
               std::back_inserter(AssignsWhereIIsAvailable));

    if (AssignsWhereIIsAvailable.empty())
      return nullptr;

    // Sort them in program order.
    // This is not strictly necessary, but it ensures determinism in picking the
    // candidate.
    // TODO: is this better or worse as an heuristic as opposed to reverse
    // program order, or possibly other heuristics?
    llvm::sort(AssignsWhereIIsAvailable,
               [&PO = ProgramOrdering](const AssignType *LHS,
                                       const AssignType *RHS) {
                 return PO.at(LHS) < PO.at(RHS);
               });

    AssignType *CandidateAssign = AssignsWhereIIsAvailable.front();
    revng_log(Log, "First CandidateAssign: " << dumpToString(CandidateAssign));
    return CandidateAssign;
  }
};

template<>
AnalysisKey InstructionToSerializePicker<true>::Key = {};
template<>
AnalysisKey InstructionToSerializePicker<false>::Key = {};

using TypeMap = std::map<const Value *, const model::UpcastableType>;

template<bool IsLegacy>
using LVB = LocalVariableBuilder<IsLegacy>;

template<bool IsLegacy>
LocalVariableBuilder<IsLegacy>
makeVariableBuilder(Function &F,
                    unsigned InputPointerByteSize,
                    const model::Binary &Model) {

  if constexpr (IsLegacy) {
    return LVB<IsLegacy>::makeLegacy(Model, &F);
  } else {
    VariableBuilderTypes Types = VariableBuilderTypes{ *F.getParent(),
                                                       InputPointerByteSize };

    return LVB<IsLegacy>::make(Types, &F);
  }
}

template<bool IsLegacy>
class VariableInserter {
public:
  using PickedInstructions = PickedInstructions<IsLegacy>;

private:
  // The Model is only used in legacy mode. Drop this when we drop legacy mode.
  const model::Binary &Model;
  // The TypeMap is only used in legacy mode. Drop this when we drop legacy
  // mode.
  const TypeMap TheTypeMap;

  Function &F;

  LocalVariableBuilder<IsLegacy> VariableBuilder;

public:
  VariableInserter(Function &TheF,
                   unsigned InputPointerByteSize,
                   // TODO: drop the next 2 arguments when we drop legacy mode
                   const model::Binary &TheModel,
                   TypeMap &&TMap) :
    Model(TheModel),
    TheTypeMap(std::move(TMap)),
    F(TheF),
    VariableBuilder(makeVariableBuilder<IsLegacy>(F,
                                                  InputPointerByteSize,
                                                  TheModel)) {}

public:
  bool run(const PickedInstructions &Picked) {
    bool Changed = false;

    for (const auto &[TheUse, TheAssign] : Picked.ToReplaceWithAvailable) {
      auto *Copy = VariableBuilder.createCopyFromAssignedOnUse(TheAssign,
                                                               *TheUse);
      TheUse->set(Copy);
    }

    for (Instruction *I : Picked.AssignToRemove) {
      Changed = true;
      I->eraseFromParent();
    }

    for (Instruction *I : Picked.ToSerialize) {
      // If ToSerialize contains something with 0 uses we don't add the
      // local variable.
      Changed |= serializeToLocalVariable(I);
    }

    return Changed;
  }

private:
  bool serializeToLocalVariable(Instruction *I);

  bool shouldReplaceUseWithCopies(const Instruction *I, const Use &U) const;
};

template<bool IsLegacy>
using VI = VariableInserter<IsLegacy>;

template<bool IsLegacy>
bool VI<IsLegacy>::shouldReplaceUseWithCopies(const Instruction *I,
                                              const Use &U) const {
  auto *Call = getCallToIsolatedFunction(I);
  if (not Call)
    return true;

  if constexpr (IsLegacy) {

    const auto *ProtoT = getCallSitePrototype(Model, cast<CallInst>(I));
    abi::FunctionType::Layout Layout = abi::FunctionType::Layout::make(*ProtoT);

    // If the Isolated function doesn't return an aggregate, we have to
    // inject copies from local variables.
    using namespace abi::FunctionType;
    if (Layout.returnMethod() != ReturnMethod::ModelAggregate)
      return true;

    // SPTAR return aggregates also need copies from local variables,
    // because they are emitted as scalar pointer variables in C.
    if (Layout.hasSPTAR())
      return true;

    // Non-SPTAR return aggregates, in legacy mode, are special in many ways:
    // 1. they basically imply a LocalVariable;
    // 2. their only expected use is supposed to be in custom opcodes that
    // expect references
    // For these reasons it would be wrong to inject a Copy.
    if (isCallToTagged(U.getUser(), FunctionTags::Assign)) {
      // If it's an assignment, and the operand number is 0, the value of I is
      // being written somewhere (in the location referenced by operand 1).
      // Hence we have to inject a copy.
      return U.getOperandNo() == 0;
    }
    revng_assert(1 == U.getOperandNo());
    revng_assert(isCallToTagged(U.getUser(), FunctionTags::AddressOf)
                 or isCallToTagged(U.getUser(), FunctionTags::ModelGEPRef));
    return false;
  }
  return true;
}

template<bool IsLegacy>
bool VI<IsLegacy>::serializeToLocalVariable(Instruction *I) {
  // We can't serialize instructions with reference semantics into local
  // variables because C doesn't have references.
  revng_assert(not isCallToTagged(I, FunctionTags::IsRef));

  // First, we have to declare the LocalVariable, always at the entry block.
  // Create instruction that allocates a LocalVariable
  LocalVarType<IsLegacy> *LocalVariable = nullptr;

  if constexpr (IsLegacy) {
    revng_assert(I->getType()->isIntOrPtrTy());
    const model::UpcastableType &VariableType = TheTypeMap.at(I);
    LocalVariable = VariableBuilder.createLocalVariable(*VariableType);
    LocalVariable->setDebugLoc(I->getDebugLoc());
  } else {
    revng::NonDebugInfoCheckingIRBuilder B(F.getContext());
    B.SetInsertPointPastAllocas(&F, I->getDebugLoc());
    LocalVariable = B.CreateAlloca(I->getType());
  }

  // Then, we have to replace all the uses of I so that they make a Copy
  // from the LocalVariable, unless it's a call to an IsolatedFunction that
  // already returns a local variable, in which case we don't have to do
  // anything with uses.
  for (Use &U : llvm::make_early_inc_range(I->uses())) {
    revng_assert(isa<Instruction>(U.getUser()));

    llvm::Instruction *ValueToUse = LocalVariable;
    if (not IsLegacy or shouldReplaceUseWithCopies(I, U)) {
      ValueToUse = VariableBuilder.createCopyOnUse(LocalVariable, U);
    }
    U.set(ValueToUse);
  }

  VariableBuilder.createAssignmentBefore(LocalVariable,
                                         I,
                                         I->getNextNonDebugInstruction());

  return true;
}

template<bool IsLegacy>
void registerCommonAnalyses(FunctionAnalysisManager &FAM) {

  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);

  if constexpr (not IsLegacy) {
    FAM.registerPass([] { return llvm::registerAAAnalyses(); });
  }
  FAM.registerPass([] { return AvailableExpressionsAnalysis<IsLegacy>(); });
  FAM.registerPass([] { return InstructionToSerializePicker<IsLegacy>(); });
}

template<bool IsLegacy>
class SwitchToStatements
  : public llvm::PassInfoMixin<SwitchToStatements<IsLegacy>> {

private:
  // This is only used in legacy mode for operating the LocalVariableBuilder
  // inside the VariableInserter.
  // TODO: drop when we drop legacy mode.
  const model::Binary *Model;

  /// The size in bytes of a pointer in the Binary we're decompiling.
  /// Necessary for initializing a LocalVariableBuilder without the Model, in
  /// non-legacy mode.
  unsigned InputPointerByteSize;

public:
  // This is meant to be used only in non-legacy mode.
  SwitchToStatements(unsigned InputPointerByteSize) :
    Model(nullptr), InputPointerByteSize(InputPointerByteSize) {
    revng_assert(not IsLegacy);
  }

private:
  // This is meant to be used only in legacy mode.
  // This is why it's private, so we it can only be invoked via the factory,
  // which is only available when IsLegacy is true.
  //
  // TODO: drop this when we drop legacy mode.
  SwitchToStatements(const model::Binary &TheModel) :
    Model(&TheModel),
    InputPointerByteSize(getPointerSize(Model->Architecture())) {
    revng_assert(IsLegacy);
  }

public:
  // Factory meant to be called only for legacy mode.
  //
  // TODO: drop this when we drop legacy mode.
  static SwitchToStatements makeLegacy(const model::Binary &Model)
    requires IsLegacy
  {
    return SwitchToStatements(Model);
  }

public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {

    TypeMap InstructionTypes = {};
    if constexpr (IsLegacy) {
      auto ModelFunction = llvmToModelFunction(*Model, F);
      revng_assert(ModelFunction != nullptr);

      InstructionTypes = initModelTypesConsideringUses(F,
                                                       ModelFunction,
                                                       *Model,
                                                       /* PointersOnly */
                                                       false);
    }
    VariableInserter<IsLegacy> VarInserter{
      F, InputPointerByteSize, *Model, std::move(InstructionTypes)
    };

    const auto
      &Picked = FAM.getResult<InstructionToSerializePicker<IsLegacy>>(F);
    bool Changed = VarInserter.run(Picked);

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static void registerCallbacks(PassBuilder &PB)
    requires(not IsLegacy)
  {
    using PipelineElementArray = ArrayRef<PassBuilder::PipelineElement>;
    PB.registerAnalysisRegistrationCallback(registerCommonAnalyses<false>);
    PB.registerPipelineParsingCallback([](StringRef Name,
                                          FunctionPassManager &FPM,
                                          PipelineElementArray) {
      if (Name == "switch-to-statements-test") {
        FPM.addPass(SwitchToStatements<false>{ /*InputPointerSize*/ 8 });
        return true;
      }
      return false;
    });
  }
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return { LLVM_PLUGIN_API_VERSION,
           "SwitchToStatementsTests",
           "1.0",
           SwitchToStatements<false>::registerCallbacks };
}

// The template is only for legacy mode.
// TODO: drop when we drop legacy mode.
template<bool IsLegacy>
bool switchToStatements(const model::Binary *Model, llvm::Function &F);

// This specialization is only for legacy mode.
// TODO: drop when we drop legacy mode.
template<>
bool switchToStatements<true>(const model::Binary *Model, llvm::Function &F) {
  revng_log(Log, "switchToStatements (legacy): " << F.getName());

  FunctionAnalysisManager FAM;
  registerCommonAnalyses<true>(FAM);

  FunctionPassManager FPM;
  FPM.addPass(SwitchToStatements<true>::makeLegacy(*Model));
  llvm::PreservedAnalyses Preserved = FPM.run(F, FAM);

  return Preserved.areAllPreserved() ? false : true;
}

template<>
bool switchToStatements<false>(const model::Binary *Model, llvm::Function &F) {

  revng_log(Log, "switchToStatements (non-legacy): " << F.getName());

  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;

  // Cross register passes, because function-level alias analyses fall back to
  // querying GlobalsAA in some cases. If the analysis manager aren't
  // cross-registered, that fallback query just crashes.
  // We don't activate GlobalsAA, but GlobalsAA still needs to be registered, in
  // order for local AA to query it and see it doesn't have cached results
  // because it did not run.
  FAM.registerPass([&MAM] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  MAM.registerPass([&FAM] { return FunctionAnalysisManagerModuleProxy(FAM); });

  // Register module-level analyses, because alias analysis looks up GlobalsAA,
  // which is a module-level pass.
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);

  registerCommonAnalyses<false>(FAM);

  FunctionPassManager FPM;
  FPM.addPass(SwitchToStatements<false>(getPointerSize(Model->Architecture())));
  llvm::PreservedAnalyses Preserved = FPM.run(F, FAM);

  return Preserved.areAllPreserved() ? false : true;
}

template<bool IsLegacy>
class SwitchToStatementsPass : public FunctionPass {
public:
  static char ID;

  SwitchToStatementsPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

template<>
char SwitchToStatementsPass<false>::ID = 0;
template class SwitchToStatementsPass<false>;

template<>
char SwitchToStatementsPass<true>::ID = 0;
template class SwitchToStatementsPass<true>;

template<bool IsLegacy>
bool SwitchToStatementsPass<IsLegacy>::runOnFunction(llvm::Function &F) {
  auto
    *Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();
  return switchToStatements<IsLegacy>(Model, F);
}

using RegisterLegacy = RegisterPass<SwitchToStatementsPass<true>>;
static RegisterLegacy
  X("legacy-switch-to-statements", "LegacySwitchToStatements", false, false);

using Register = RegisterPass<SwitchToStatementsPass<false>>;
static Register
  Y("switch-to-statements", "SwitchToStatementsPass", false, false);

namespace revng::pypeline::piperuns {

// TODO: inline switchToStatements once we dismiss the old pipeline
void SwitchToStatements::runOnLLVMFunction(const model::Function &Function,
                                           llvm::Function &LLVMFunction) {
  switchToStatements<false>(Model.get(), LLVMFunction);
}

} // namespace revng::pypeline::piperuns
