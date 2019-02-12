#ifndef REACHINGDEFINITIONSANALYSISIMPL_H
#define REACHINGDEFINITIONSANALYSISIMPL_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/ADT/SmallVector.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/MemoryAccess.h"
#include "revng/Support/MonotoneFramework.h"

struct MemoryInstruction {
  MemoryInstruction() : I(nullptr), MA() {}
  MemoryInstruction(llvm::StoreInst *I, const llvm::DataLayout &DL) :
    I(I),
    MA(I, DL) {}
  MemoryInstruction(llvm::LoadInst *I, const llvm::DataLayout &DL) :
    I(I),
    MA(I, DL) {}

  template<typename T>
  static MemoryInstruction
  create(llvm::StoreInst *I, const llvm::DataLayout &DL, const T &Container) {
    MemoryInstruction Result(I, DL);
    for (int32_t Color : Container)
      Result.Colors.push_back(Color);
    return Result;
  }

  template<typename T>
  static MemoryInstruction
  create(llvm::LoadInst *I, const llvm::DataLayout &DL, const T &Container) {
    MemoryInstruction Result(I, DL);
    for (int32_t Color : Container)
      Result.Colors.push_back(Color);
    return Result;
  }

  bool operator<(const MemoryInstruction Other) const { return I < Other.I; }
  bool operator>(const MemoryInstruction Other) const { return I > Other.I; }
  bool operator==(const MemoryInstruction Other) const { return I == Other.I; }

  llvm::Instruction *I;
  MemoryAccess MA;
  llvm::SmallVector<int32_t, 4> Colors;
};

/// \brief Normalize the graph: indirect branch successors must have only one
///        predecessor
inline std::set<llvm::BasicBlock *> highlightConditionEdges(llvm::Function &F) {
  using namespace llvm;

  LLVMContext &C = getContext(&F);

  std::set<BasicBlock *> ToDelete;
  for (BasicBlock &BB : F) {
    auto *T = dyn_cast<BranchInst>(BB.getTerminator());
    if (T == nullptr or T->isUnconditional())
      continue;

    std::array<Use *, 2> SuccessorsUses{ &T->getOperandUse(1),
                                         &T->getOperandUse(2) };
    for (Use *SuccessorUse : SuccessorsUses) {
      BasicBlock *Successor = cast<BasicBlock>(SuccessorUse->get());

      // Check if the successor has more than one predecessor
      if (Successor->getSinglePredecessor() == &BB)
        continue;

      // Create a new basic block, set it as successor of T
      auto *NewBB = BasicBlock::Create(C, "", &F);
      ToDelete.insert(NewBB);
      SuccessorUse->set(NewBB);

      // Add NewBB -> Successor branch
      BranchInst::Create(Successor, NewBB);
    }
  }

  return ToDelete;
}

namespace RDA {

using ColorsList = llvm::SmallVector<int32_t, 4>;
using MISet = MonotoneFrameworkSet<MemoryInstruction>;

class Interrupt {
private:
  enum InterruptType { Regular, Summary, NoReturn };

private:
  InterruptType Type;
  MISet E;

private:
  Interrupt(InterruptType Type) : Type(Type) { revng_assert(Type != Regular); }

  Interrupt(InterruptType Type, MISet E) : Type(Type), E(E) {
    revng_assert(Type == Regular);
  }

public:
  static Interrupt createRegular(MISet E) { return Interrupt(Regular, E); }

  static Interrupt createNoReturn() { return Interrupt(NoReturn); }

  static Interrupt createSummary() { return Interrupt(Summary); }

  bool requiresInterproceduralHandling() {
    switch (Type) {
    case Regular:
      return false;
    case Summary:
    case NoReturn:
      return true;
    }

    revng_abort();
  }

  MISet &&extractResult() { return std::move(E); }
  bool isReturn() const { return false; }
};

template<typename T>
struct ColorsProviderTraits {};

class NullColorsProvider {};

extern ColorsList EmptyColorsList;
extern llvm::SmallVector<int32_t, 4> EmptyResetColorsList;

template<>
struct ColorsProviderTraits<NullColorsProvider> {

  static const ColorsList &
  getBlockColors(const NullColorsProvider &CP, llvm::BasicBlock *BB) {
    return EmptyColorsList;
  }

  static int32_t getEdgeColor(const NullColorsProvider &CP,
                              llvm::BasicBlock *Source,
                              llvm::BasicBlock *Destination) {
    return 0;
  }

  static const llvm::SmallVector<int32_t, 4> &
  getResetColors(const NullColorsProvider &CP, llvm::BasicBlock *BB) {
    return EmptyResetColorsList;
  }
};

extern llvm::SmallVector<llvm::Instruction *, 4> EmtpyReachersList;

template<typename T>
inline const GeneratedCodeBasicInfo *getGCBIOrNull(const T &Obj) {
  return nullptr;
}

template<>
inline const GeneratedCodeBasicInfo *
getGCBIOrNull<GeneratedCodeBasicInfo>(const GeneratedCodeBasicInfo &GCBI) {
  return &GCBI;
}

template<typename ColorsProvider = NullColorsProvider,
         typename BlackList = NullBlackList>
class Analysis
  : public MonotoneFramework<llvm::BasicBlock *,
                             MISet,
                             Interrupt,
                             Analysis<ColorsProvider, BlackList>,
                             llvm::SmallVector<llvm::BasicBlock *, 2>,
                             ReversePostOrder> {
public:
  using SuccessorsList = llvm::SmallVector<llvm::BasicBlock *, 2>;

private:
  using Base = MonotoneFramework<llvm::BasicBlock *,
                                 MISet,
                                 Interrupt,
                                 Analysis<ColorsProvider, BlackList>,
                                 SuccessorsList,
                                 ReversePostOrder>;

private:
  /// The function to analyze
  llvm::Function *F;

  /// Map for the results: records all the reaching definitions
  using InstructionVector = llvm::SmallVector<llvm::Instruction *, 4>;
  std::map<llvm::LoadInst *, InstructionVector> ReachedBy;

  /// Map of colors associated to a basic block
  const ColorsProvider &TheColorsProvider;

  /// Trait to query a blacklist
  BlackListTrait<const BlackList &, llvm::BasicBlock *> TheBlackList;

  const GeneratedCodeBasicInfo *GCBI;
  const FunctionCallIdentification *FCI;
  const StackAnalysis::StackAnalysis<false> *SA;

public:
  Analysis(llvm::Function *F,
           const ColorsProvider &TheColorsProvider,
           const BlackList &TheBlackList,
           const FunctionCallIdentification *FCI,
           const StackAnalysis::StackAnalysis<false> *SA) :
    Base(&F->getEntryBlock()),
    F(F),
    TheColorsProvider(TheColorsProvider),
    TheBlackList(TheBlackList),
    FCI(FCI),
    SA(SA) {

    GCBI = getGCBIOrNull(TheBlackList);
  }

  std::map<llvm::LoadInst *, llvm::SmallVector<llvm::Instruction *, 4>> &&
  extractResults() {
    return std::move(ReachedBy);
  }

  void initialize() {
    ReachedBy.clear();
    Base::initialize();
  }

  void assertLowerThanOrEqual(const MISet &A, const MISet &B) const {}

  void dumpFinalState() const {}

  SuccessorsList successors(llvm::BasicBlock *BB, Interrupt &) const {
    SuccessorsList Result;

    if (FCI != nullptr) {
      const CustomCFG &FilteredCFG = FCI->cfg();
      revng_assert(FilteredCFG.hasNode(BB));
      for (CustomCFGNode *Node : FilteredCFG.getNode(BB)->successors())
        Result.push_back(Node->block());
    } else {
      for (llvm::BasicBlock *Successor : make_range(succ_begin(BB),
                                                    succ_end(BB)))
        Result.push_back(Successor);
    }

    return Result;
  }

  size_t successor_size(llvm::BasicBlock *BB, Interrupt &I) const {
    // TODO: not nice
    return successors(BB, I).size();
  }

  Interrupt createSummaryInterrupt() { return Interrupt::createSummary(); }

  Interrupt createNoReturnInterrupt() const {
    return Interrupt::createNoReturn();
  }

  MISet extremalValue(llvm::BasicBlock *) const { return MISet(); }

  typename Base::LabelRange extremalLabels() const {
    return { &F->getEntryBlock() };
  }

  const ColorsList &getBlockColors(llvm::BasicBlock *BB) const {
    using CP = ColorsProviderTraits<ColorsProvider>;
    return CP::getBlockColors(TheColorsProvider, BB);
  }

  int32_t
  getEdgeColor(llvm::BasicBlock *Source, llvm::BasicBlock *Destination) const {
    return ColorsProviderTraits<ColorsProvider>::getEdgeColor(TheColorsProvider,
                                                              Source,
                                                              Destination);
  }

  const llvm::SmallVector<int32_t, 4> &
  getResetColors(llvm::BasicBlock *BB) const {
    using CP = ColorsProviderTraits<ColorsProvider>;
    return CP::getResetColors(TheColorsProvider, BB);
  }

  llvm::Optional<MISet> handleEdge(const MISet &Original,
                                   llvm::BasicBlock *Source,
                                   llvm::BasicBlock *Destination) const {
    using namespace llvm;

    // Is the destination blacklisted?
    if (TheBlackList.isBlacklisted(Destination))
      return { MISet() };

    // Is the destination painted with a color that is opposite to one of those
    // where it has been defined?
    int32_t EdgeColor = getEdgeColor(Source, Destination);
    if (EdgeColor == 0)
      return Optional<MISet>();

    MISet Filtered = Original;

    using MI = MemoryInstruction;
    auto HasOppositeColors = [EdgeColor](const MI &Other) {
      for (int32_t DefiningColor : Other.Colors)
        if (DefiningColor == -EdgeColor)
          return true;
      return false;
    };
    Filtered.erase_if(HasOppositeColors);
    bool Changed = Filtered.size() != Original.size();

    llvm::SmallVector<MemoryInstruction, 4> ToInsert;
    for (auto It = Filtered.begin(); It != Filtered.end();) {

      // TODO: this is suboptimal
      // Check if this MI has EdgeColor
      auto ColorIt = std::find(It->Colors.begin(), It->Colors.end(), EdgeColor);
      if (ColorIt == It->Colors.end()) {
        // Prepare new entry adding EdgeColor
        MemoryInstruction Clone = *It;
        Clone.Colors.push_back(EdgeColor);
        ToInsert.push_back(Clone);

        // Delete old entry
        It = Filtered.erase(It);

        Changed = true;
      } else {
        It++;
      }
    }

    for (MemoryInstruction &MI : ToInsert)
      Filtered.insert(MI);

    // If something changed, return the updated version
    if (Changed)
      return { Filtered };

    // Returning an empty optional means Original will be used as is
    return Optional<MISet>();
  }

  Interrupt transfer(llvm::BasicBlock *BB) {
    using namespace llvm;

    MISet AliveMIs = this->State[BB];

    //
    // Remove the colors that need to be reset in this basic block
    //
    const SmallVector<int32_t, 4> &ResetColors = getResetColors(BB);

    SmallVector<MemoryInstruction, 4> ToInsert;
    for (auto It = AliveMIs.begin(); It != AliveMIs.end();) {

      // Clone and drop all reset colors
      MemoryInstruction Clone = *It;
      auto IsResetColor = [&ResetColors](int32_t Color) {
        auto It = std::find(ResetColors.begin(), ResetColors.end(), Color);
        return It != ResetColors.end();
      };
      Clone.Colors.erase(std::remove_if(Clone.Colors.begin(),
                                        Clone.Colors.end(),
                                        IsResetColor),
                         Clone.Colors.end());

      // Did we remove at least one color?
      if (Clone.Colors.size() != It->Colors.size()) {
        // Remove and register for insertion
        It = AliveMIs.erase(It);
        ToInsert.push_back(Clone);
      } else {
        // Proceed
        It++;
      }
    }

    for (MemoryInstruction &MI : ToInsert)
      AliveMIs.insert(MI);

    //
    // Apply the transfer function instruction by instruction
    //
    const DataLayout &DL = getModule(BB)->getDataLayout();

    for (Instruction &I : *BB) {
      MemoryInstruction MI;
      auto *Load = dyn_cast<LoadInst>(&I);
      auto *Store = dyn_cast<StoreInst>(&I);
      auto MayAlias = [&MI](const MemoryInstruction &Other) {
        return MI.MA.mayAlias(Other.MA);
      };

      if (Load != nullptr) {
        MI = MemoryInstruction::create(Load, DL, getBlockColors(BB));

        if (not MI.MA.isValid())
          continue;

        // Register all the reachers
        SmallVector<Instruction *, 4> &Reachers = ReachedBy[Load];
        Reachers.clear();

        for (const MemoryInstruction &AliveMI : AliveMIs)
          if (AliveMI.I != MI.I and AliveMI.MA == MI.MA)
            Reachers.push_back(AliveMI.I);

        if (Reachers.size() == 0)
          AliveMIs.insert(MI);

      } else if (Store != nullptr) {
        Value *Pointer = Store->getPointerOperand();
        if ((not isa<GlobalVariable>(Pointer) and not isa<AllocaInst>(Pointer))
            or Pointer->getName() == "env")
          continue;

        MI = MemoryInstruction::create(Store, DL, getBlockColors(BB));

        // Erase all the instruction clobbered by this store
        AliveMIs.erase_if(MayAlias);

        // Insert the store instruction among the alive instructions
        AliveMIs.insert(MI);
      }
    }

    //
    // Drop MIs clobbered by callee, if function call
    //
    if (FCI != nullptr and SA != nullptr and GCBI != nullptr
        and FCI->isCall(BB)) {
      // Filter definitions according to stack analysis
      BasicBlock *Callee = getFunctionCallCallee(BB);
      const auto &Clobbered = SA->getClobbered(Callee);
      GlobalVariable *StackPointer = GCBI->spReg();
      auto IsClobbered = [&Clobbered,
                          StackPointer](const MemoryInstruction &Other) {
        Value *CSVValue = Other.MA.globalVariable();
        if (auto *CSV = dyn_cast_or_null<GlobalVariable>(CSVValue))
          return Clobbered.count(CSV) != 0;
        else if (const Value *Base = Other.MA.base())
          return Base != StackPointer;
        else
          return false;
      };

      AliveMIs.erase_if(IsClobbered);
    }

    //
    // Prevent excessive propagation
    //
    // TODO: is this still necessary? This was an issue with return instructions
    unsigned SuccessorsCount = succ_end(BB) - succ_begin(BB);
    unsigned Size = AliveMIs.size();
    if (Size * SuccessorsCount > 5000)
      return Interrupt::createRegular(MISet());

    return Interrupt::createRegular(std::move(AliveMIs));
  }

public:
  const llvm::SmallVector<llvm::Instruction *, 4> &
  getReachers(llvm::LoadInst *I) const {
    auto It = ReachedBy.find(I);
    if (It == ReachedBy.end())
      return EmtpyReachersList;
    else
      return It->second;
  }
};

} // namespace RDA

#endif // REACHINGDEFINITIONSANALYSISIMPL_H
