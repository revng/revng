//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/Queue.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Support/Assert.h"
#include "revng/Support/GlobalToLocalPromoter.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

/// All the uses of a single global variable within one function
struct GlobalUses {
  GlobalVariable *Global = nullptr;
  SmallVector<Use *, 8> Uses;
  bool Replaced = false;
};

template<>
struct KeyedObjectTraits<GlobalUses> {
  static GlobalVariable *key(const GlobalUses &Element) {
    return Element.Global;
  }

  static GlobalUses fromKey(GlobalVariable *Global) {
    return GlobalUses{ .Global = Global };
  }
};

/// Orders global variables alphabetically
struct CompareGlobalsByName {
  bool operator()(const GlobalVariable *LHS, const GlobalVariable *RHS) const {
    return CompareByName(LHS, RHS);
  }
};

/// Replaces the uses of global variables within one function with allocas
///
/// The constructor records every use, in the function, of every global
/// `ShouldPromote` accepts. `globals()` returns those globals in alphabetical
/// order.
///
/// Only a global used directly, or through a chain of cast expressions, can
/// be replaced: an equivalent cast of the alloca takes the place of the whole
/// chain. Referring to a global to promote in any other way aborts.
///
/// The globals in `DefaultStringNamespace` are never promoted, whatever
/// `ShouldPromote` says.
class GlobalToLocalPromoter {
public:
  /// Whether to promote a global. An empty predicate promotes all of them.
  using ShouldPromote = std::function<bool(const GlobalVariable &)>;

  using SortedVector = SortedVector<GlobalUses, CompareGlobalsByName>;

private:
  Function *TheFunction = nullptr;

  /// The uses of each global to promote, in alphabetical order
  SortedVector UsedGlobals;

public:
  GlobalToLocalPromoter(const ShouldPromote &ShouldPromote, Function &F);

  /// Replaces with an AllocaInst all uses of the globals selected with
  /// ShouldPromote. \return a vector mapping each replaced GlobalVariable to
  /// its associated AllocaInst, sorted by name of the globals.
  SmallVector<std::pair<GlobalVariable *, AllocaInst *>> promote();

private:
  /// Replace with \p Alloca all the uses of \p Global within the function
  ///
  /// \return true if at least one use has been replaced
  bool replaceWithAlloca(GlobalVariable *Global, AllocaInst *Alloca);

  static GlobalVariable *getGlobal(const GlobalUses &Element) {
    return Element.Global;
  }
};

using Promoter = GlobalToLocalPromoter;

/// \return true if \p Global is one of the string constants rev.ng emits
static bool isEmittedStringConstant(const GlobalVariable &Global) {
  return Global.getName().starts_with(DefaultStringNamespace);
}

/// Collects the uses of the globals to promote in one walk of a function
class UseCollector {
private:
  const Promoter::ShouldPromote &ShouldPromote;
  Promoter::SortedVector &UsedGlobals;

  /// What `shouldPromote` answered for each global it was asked about
  DenseMap<const GlobalVariable *, bool> Answers;

public:
  UseCollector(const Promoter::ShouldPromote &ShouldPromote,
               Promoter::SortedVector &UsedGlobals) :
    ShouldPromote(ShouldPromote), UsedGlobals(UsedGlobals) {}

public:
  /// Record every use of a global to promote in \p F
  void run(Function &F);

private:
  /// \return true if \p Global is one of the globals to promote
  bool shouldPromote(const GlobalVariable &Global);

  /// \return the global to promote \p Root is a use of, if any
  GlobalVariable *getUsedGlobal(Constant *Root);

  /// \return true if \p Root refers to a global to promote
  bool refersToGlobalToPromote(const Constant *Root);
};

void UseCollector::run(Function &F) {
  // The uses are grouped by global on the way, keyed by pointer. Ordering the
  // globals by name once, at the end, is what keeps the walk from comparing
  // names once per use.
  DenseMap<GlobalVariable *, SmallVector<Use *, 8>> Grouped;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (Use &TheUse : I.operands()) {
        Value *Operand = TheUse.get();

        // A global reaches a non-constant only through the operands of the
        // instruction that produced it, which this walk visits in their own
        // right.
        auto *C = dyn_cast<Constant>(Operand);
        if (not C)
          continue;

        if (auto *Global = getUsedGlobal(C))
          Grouped[Global].push_back(&TheUse);
      }
    }
  }

  // `Grouped` is keyed by pointer, so it is iterated in no particular order.
  // The order of what comes out of here does not depend on that: the batch
  // inserter sorts by name on commit, and the uses of one global keep the
  // order the walk found them in.
  auto Inserter = UsedGlobals.batch_insert();
  for (auto &[Global, Uses] : Grouped)
    Inserter.emplace(GlobalUses{ .Global = Global, .Uses = std::move(Uses) });
}

bool UseCollector::shouldPromote(const GlobalVariable &Global) {
  auto It = Answers.find(&Global);
  if (It != Answers.end())
    return It->second;

  bool Promote = not isEmittedStringConstant(Global)
                 and (not ShouldPromote or ShouldPromote(Global));

  // A global to promote ends up in a container that orders by name.
  revng_assert(not Promote or Global.hasName());

  Answers[&Global] = Promote;
  return Promote;
}

GlobalVariable *UseCollector::getUsedGlobal(Constant *Root) {
  // Casts are the only expressions that can be rebuilt out of an alloca.
  auto *Stripped = cast<Constant>(skipCasts(Root));

  if (auto *Global = dyn_cast<GlobalVariable>(Stripped))
    return shouldPromote(*Global) ? Global : nullptr;

  // Constant data has no operands to look into.
  if (Stripped->getNumOperands() == 0)
    return nullptr;

  revng_assert(not refersToGlobalToPromote(Stripped),
               "A global variable to replace is used by a constant "
               "expression that is not a cast");

  return nullptr;
}

bool UseCollector::refersToGlobalToPromote(const Constant *Root) {
  OnceQueue<const Constant *> Queue;
  Queue.insert(Root);

  while (not Queue.empty()) {
    const Constant *Current = Queue.pop();

    // The initializer of a global is not a use within the function.
    if (auto *Global = dyn_cast<GlobalVariable>(Current)) {
      if (shouldPromote(*Global))
        return true;
      continue;
    }

    if (isa<BlockAddress>(Current))
      continue;

    for (const Use &Operand : Current->operands())
      Queue.insert(cast<Constant>(Operand.get()));
  }

  return false;
}

Promoter::GlobalToLocalPromoter(const ShouldPromote &ShouldPromote,
                                Function &F) :
  TheFunction(&F) {
  UseCollector Collector(ShouldPromote, UsedGlobals);
  Collector.run(F);
}

SmallVector<std::pair<GlobalVariable *, AllocaInst *>> Promoter::promote() {

  SmallVector<std::pair<GlobalVariable *, AllocaInst *>> Promoted;

  revng::IRBuilder AllocaBuilder(TheFunction->getContext());
  AllocaBuilder.SetInsertPointPastAllocas(TheFunction);

  for (GlobalVariable *CSV : map_range(UsedGlobals, getGlobal)) {
    Type *CSVType = CSV->getValueType();
    auto *Alloca = AllocaBuilder.CreateAlloca(CSVType, nullptr, CSV->getName());
    replaceWithAlloca(CSV, Alloca);
    Promoted.emplace_back(CSV, Alloca);
  }
  return Promoted;
}

bool Promoter::replaceWithAlloca(GlobalVariable *Global, AllocaInst *Alloca) {
  revng_assert(Global != nullptr);
  revng_assert(Alloca != nullptr);
  revng_assert(Alloca->getFunction() == TheFunction);

  auto It = UsedGlobals.find(Global);
  if (It == UsedGlobals.end())
    return false;

  GlobalUses &Element = *It;
  revng_assert(not Element.Replaced,
               "A global variable can only be replaced once");
  Element.Replaced = true;

  // A use of a cast expression of the global takes the same cast of the
  // alloca. The casts go next to the allocas, where they dominate every use,
  // and one of them serves every use of the same type.
  revng::IRBuilder Builder(Alloca->getContext());
  Builder.SetInsertPointPastAllocas(Alloca->getFunction());
  SmallDenseMap<Type *, Value *, 2> Replacements;

  for (Use *TheUse : Element.Uses) {
    Type *UseType = TheUse->get()->getType();
    Value *&Replacement = Replacements[UseType];

    // For a use of the global itself this creates nothing and gives back
    // the alloca.
    if (Replacement == nullptr)
      Replacement = Builder.CreateBitOrPointerCast(Alloca, UseType);

    TheUse->set(Replacement);
  }

  return not Element.Uses.empty();
}

SmallVector<std::pair<GlobalVariable *, AllocaInst *>>
promoteGlobalsToLocals(Function &F,
                       const std::function<bool(const GlobalVariable &)>
                         &ShouldPromote) {
  return GlobalToLocalPromoter(ShouldPromote, F).promote();
}
