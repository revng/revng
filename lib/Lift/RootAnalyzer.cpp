/// \file RootAnalyzer.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/BasicAnalyses/ShrinkInstructionOperandsPass.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueRegisterUser.h"
#include "revng/Support/Statistics.h"
#include "revng/TypeShrinking/BitLiveness.h"
#include "revng/TypeShrinking/TypeShrinking.h"
#include "revng/ValueMaterializer/DataFlowGraph.h"

#include "CPUStateAccessAnalysisPass.h"
#include "JumpTargetManager.h"
#include "RootAnalyzer.h"
#include "ValueMaterializerPass.h"

using namespace llvm;

RunningStatistics BlocksAnalyzedByValueMaterializer("blocks-analyzed-by-avi");
RunningStatistics WrittenInPCStatistics("written-in-pc");
RunningStatistics DetectedEdgesStatistics("detected-edges");
RunningStatistics StoredInMemoryStatistics("stored-in-memory");
RunningStatistics LoadAddressStatistics("load-address");

Logger<> NewEdgesLog("new-edges");
static Logger<> Log("root-analyzer");

// NOTE: Setting this to 1 gives us performance improvement. We have tested and
// realized that there is an impact on performance if setting it to 2.
constexpr unsigned InstCombineMaxIterations = 1;

/// Drop all the call to marker functions
class DropMarkerCalls : public PassInfoMixin<DropMarkerCalls> {
private:
  SmallVector<StringRef, 4> NoReturns;

public:
  DropMarkerCalls(SmallVector<StringRef, 4> NoReturns) : NoReturns(NoReturns) {}

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    Module *M = F.getParent();
    std::vector<CallBase *> ToErase;

    for (Function &Marker : FunctionTags::Marker.functions(M)) {
      StringRef MarkerName = Marker.getName();
      if (llvm::count(NoReturns, MarkerName) != 0) {
        // Preserve but mark as noreturn
        Marker.setDoesNotReturn();
      } else {
        for (CallBase *Call : callersIn(&Marker, &F)) {
          // Register the call to be erased
          ToErase.push_back(Call);
        }
      }
    }

    //
    // Actually drop the calls
    //
    for (CallBase *Call : ToErase)
      eraseFromParent(Call);

    return PreservedAnalyses::none();
  }
};

/// Simple pass to drop `range` metadata, which is sometimes detrimental
class DropRangeMetadataPass : public PassInfoMixin<DropRangeMetadataPass> {

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        I.setMetadata("range", nullptr);
    return PreservedAnalyses::all();
  }
};

/// Turn load instructions from constant addresses into constants
class ConstantLoadsFolderPass : public PassInfoMixin<ConstantLoadsFolderPass> {
private:
  StaticDataMemoryOracle &MO;

public:
  ConstantLoadsFolderPass(StaticDataMemoryOracle &MO) : MO(MO) {}

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    for (Instruction &I : llvm::instructions(F)) {
      auto *Load = dyn_cast<LoadInst>(&I);
      if (Load == nullptr)
        continue;

      Type *LoadType = Load->getType();
      if (not LoadType->isIntegerTy())
        continue;

      ConstantInt *Address = nullptr;
      using namespace PatternMatch;
      if (not match(Load, m_Load(m_IntToPtr(m_ConstantInt(Address)))))
        continue;

      uint64_t LoadAddress = Address->getLimitedValue();
      unsigned LoadSize = LoadType->getIntegerBitWidth() / 8;
      MaterializedValue Loaded = MO.load(LoadAddress, LoadSize);
      if (not Loaded.isValid() or Loaded.hasSymbol())
        continue;

      Load->replaceAllUsesWith(ConstantInt::get(LoadType, Loaded.value()));
    }
    return PreservedAnalyses::none();
  }
};

namespace TrackedInstructionType {

enum Values {
  Invalid,
  WrittenInPC,
  StoredInMemory,
  StoreTarget,
  LoadTarget
};

inline const char *getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case WrittenInPC:
    return "WrittenInPC";
  case StoredInMemory:
    return "StoredInMemory";
  case StoreTarget:
    return "StoreTarget";
  case LoadTarget:
    return "LoadTarget";
  default:
    revng_abort();
  }
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid")
    return Invalid;
  else if (Name == "WrittenInPC")
    return WrittenInPC;
  else if (Name == "StoredInMemory")
    return StoredInMemory;
  else if (Name == "StoreTarget")
    return StoreTarget;
  else if (Name == "LoadTarget")
    return LoadTarget;
  else
    revng_abort();
}

} // namespace TrackedInstructionType

class AnalysisRegistry {
public:
  using TrackedValueType = TrackedInstructionType::Values;

  struct TrackedValue {
    MetaAddress Address;
    TrackedValueType Type;
    Instruction *I = nullptr;
  };

private:
  std::vector<TrackedValue> TrackedValues;
  QuickMetadata QMD;
  llvm::Function *ValueMaterializerMarker;
  IRBuilder<> Builder;

public:
  AnalysisRegistry(Module *M) : QMD(getContext(M)), Builder(getContext(M)) {
    ValueMaterializerMarker = ValueMaterializerPass::createMarker(M);
  }

  llvm::Function *aviMarker() const { return ValueMaterializerMarker; }

  void registerValue(MetaAddress Address,
                     Value *OriginalValue,
                     Value *ValueToTrack,
                     TrackedValueType Type) {
    revng_assert(Address.isValid());

    Instruction *InstructionToTrack = dyn_cast<Instruction>(ValueToTrack);
    if (InstructionToTrack == nullptr)
      return;

    revng_assert(InstructionToTrack != nullptr);

    unsigned MaxPhiLike = DataFlowGraph::Limits::Max;
    unsigned MaxLoad = DataFlowGraph::Limits::Max;
    Oracle::Values Oracle = Oracle::AdvancedValueInfo;

    // Configure ValueMaterializer
    switch (Type) {
    case TrackedInstructionType::WrittenInPC:
      // We do not think it's beneficial to traverse more than one load
      MaxLoad = 1;
      break;
    case TrackedInstructionType::LoadTarget:
    case TrackedInstructionType::StoreTarget:
      // The main goal of tracking load/store addresses is to detect constant
      // pools between functions. We deem it hard that the address of a constant
      // pool is loaded from memory.
      MaxLoad = 0;
      break;
    case TrackedInstructionType::StoredInMemory:
      // We do not think it's beneficial to traverse more than one load
      MaxLoad = 1;

      // Tracking this values is a version of collecting simple literals on
      // steroids. Basically we want to just track constant merged through a
      // phi. No need for LazyValueInfo or AdvancedValueInfo.
      Oracle = Oracle::None;
      break;
    default:
      revng_abort();
    }

    // Create the marker call and attach as last argument a unique
    // identifier. This is necessary since the instruction itself could be
    // deleted, duplicated and what not. Later on, we will use TrackedValues
    // to now the values that have been identified to which value in the
    // original function did they belong to
    llvm::Module *M = ValueMaterializerMarker->getParent();
    uint32_t ValueMaterializerID = TrackedValues.size();
    Builder.SetInsertPoint(InstructionToTrack->getNextNode());
    Builder.CreateCall(ValueMaterializerMarker,
                       { InstructionToTrack,
                         Builder.getInt32(MaxPhiLike),
                         Builder.getInt32(MaxLoad),
                         Builder.getInt32(Oracle),
                         getUniqueString(M, Address.toString()),
                         Builder.getInt32(ValueMaterializerID) });
    TrackedValue NewTV{ Address,
                        Type,
                        cast_or_null<Instruction>(OriginalValue) };
    TrackedValues.push_back(NewTV);
  }

  const TrackedValue &rootInstructionById(uint32_t ID) const {
    return TrackedValues.at(ID);
  }
};

RootAnalyzer::RootAnalyzer(JumpTargetManager &JTM) :
  JTM(JTM), TheModule(JTM.module()), Model(JTM.model()) {
}

RootAnalyzer::MetaAddressSet RootAnalyzer::inflateValueMaterializerWhitelist() {
  MetaAddressSet Result;

  // We start from all the new basic blocks (i.e., those in
  // ValueMaterializerPCWhiteList) and proceed backward in the CFG in order to
  // whitelist all the jump targets we meet. We stop when we meet the dispatcher
  // or a function call.

  // Prepare the backward visit
  df_iterator_default_set<BasicBlock *> VisitSet;

  // Stop at the dispatcher
  VisitSet.insert(JTM.dispatcher());

  // TODO: OriginalInstructionAddresses is not reliable, we should drop it
  for (User *NewPCUser : TheModule.getFunction("newpc")->users()) {
    auto *I = cast<Instruction>(NewPCUser);
    auto WhitelistedMA = addressFromNewPC(I);
    if (WhitelistedMA.isValid()) {
      if (JTM.isInValueMaterializerPCWhitelist(WhitelistedMA)) {
        BasicBlock *BB = I->getParent();
        auto VisitRange = inverse_depth_first_ext(BB, VisitSet);
        for (const BasicBlock *Reachable : VisitRange) {
          auto MA = getBasicBlockAddress(Reachable);
          if (MA.isValid() and JTM.isJumpTarget(MA)) {
            Result.insert(MA);
          }
        }
      }
    }
  }

  return Result;
}

// Update CPUStateAccessAnalysisPass
void RootAnalyzer::updateCSAA() {
  legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(ModelWrapper::createConst(Model)));
  PM.add(JTM.createCSAA());
  PM.add(new FunctionCallIdentification);
  PM.run(TheModule);
}

static llvm::SmallSet<model::Register::Values, 16>
getPreservedRegisters(const model::TypeDefinition &Prototype) {
  llvm::SmallSet<model::Register::Values, 16> Result;
  namespace FT = abi::FunctionType;
  for (model::Register::Values Register : FT::calleeSavedRegisters(Prototype))
    Result.insert(Register);
  return Result;
}

// Clone the root function.
Function *RootAnalyzer::createTemporaryRoot(Function *TheFunction,
                                            ValueToValueMapTy &OldToNew) {
  Function *OptimizedFunction = nullptr;
  Module *M = TheFunction->getParent();
  // Break all the call edges. We want to ignore those for CFG recovery
  // purposes.
  llvm::DenseSet<BasicBlock *> Callees;
  llvm::DenseMap<Use *, BasicBlock *> Undo;
  auto *FunctionCall = TheModule.getFunction("function_call");
  revng_assert(FunctionCall != nullptr);
  for (CallBase *Call : callers(FunctionCall)) {
    auto *T = Call->getParent()->getTerminator();

    Callees.insert(getFunctionCallCallee(Call->getParent()));

    if (auto *Branch = dyn_cast<BranchInst>(T)) {
      revng_assert(Branch->isUnconditional());
      BasicBlock *Target = Branch->getSuccessor(0);
      Use *U = &Branch->getOperandUse(0);

      // We're after a function call: pretend we're jumping to anypc
      U->set(JTM.anyPC());

      // Record Use for later undoing
      Undo[U] = Target;
    }
  }

  // Compute ValueMaterializerJumpTargetWhitelist
  auto
    ValueMaterializerJumpTargetWhitelist = inflateValueMaterializerWhitelist();

  // Prune the dispatcher
  JTM.setCFGForm(CFGForm::NoFunctionCalls,
                 &ValueMaterializerJumpTargetWhitelist);

  // Detach all the unreachable basic blocks, so they don't get copied
  llvm::DenseSet<BasicBlock *> UnreachableBBs = JTM.computeUnreachable();
  for (BasicBlock *UnreachableBB : UnreachableBBs)
    UnreachableBB->removeFromParent();

  // Clone the function
  OptimizedFunction = CloneFunction(TheFunction, OldToNew);

  // Restore callees after function_call
  for (auto [U, BB] : Undo)
    U->set(BB);

  // Force canonical register values at the beginning of each callee
  Callees.erase(nullptr);
  llvm::IRBuilder<> Builder(TheModule.getContext());
  for (BasicBlock *BB : Callees) {
    if (OldToNew.count(BB) == 0)
      continue;
    BB = cast<BasicBlock>(OldToNew[BB]);
    revng_assert(BB->getTerminator() != nullptr);
    Builder.SetInsertPoint(BB->getFirstNonPHI());

    for (const model::Segment &Segment : Model->Segments()) {
      if (Segment.contains(getBasicBlockAddress(BB))) {
        for (const auto &CanonicalValue : Segment.CanonicalRegisterValues()) {
          auto Name = model::Register::getCSVName(CanonicalValue.Register());
          if (auto *CSV = M->getGlobalVariable(Name)) {
            auto *Type = getCSVType(CSV);
            Builder.CreateStore(ConstantInt::get(Type, CanonicalValue.Value()),
                                CSV);
          }
        }
        break;
      }
    }
  }

  //
  // Turn function_call into clobbering non-callee-saved registers
  //
  {
    // Compute preserved registers using the default prototype
    using RegisterSet = llvm::SmallSet<model::Register::Values, 16>;
    RegisterSet PreservedRegisters;
    model::ABI::Values ABI = Model->DefaultABI();
    if (const auto *DefaultPrototype = Model->defaultPrototype()) {
      // TODO: don't forget to simplify the logic here if we decide to make
      //       default prototypes always available (after merging
      //       `abi::Definition` back into the model).
      PreservedRegisters = getPreservedRegisters(*DefaultPrototype);
    } else if (ABI != model::ABI::Invalid) {
      auto &CSRs = abi::Definition::get(ABI).CalleeSavedRegisters();
      PreservedRegisters.insert(CSRs.begin(), CSRs.end());
    } else {
      // TODO: this must be a preliminary check
      revng_abort("Either DefaultABI or DefaultPrototype needs to be "
                  "specified");
    }

    OpaqueRegisterUser Clobberer(M);
    SmallVector<CallBase *, 16> FunctionCallCalls;
    llvm::copy(callersIn(FunctionCall, OptimizedFunction),
               std::back_inserter(FunctionCallCalls));
    for (CallBase *Call : FunctionCallCalls) {
      Builder.SetInsertPoint(Call);

      // Clobber registers that are not preserved
      for (model::Register::Values Register :
           model::Architecture::registers(Model->Architecture())) {
        if (not PreservedRegisters.contains(Register))
          Clobberer.clobber(Builder, Register);
      }
    }

    for (CallBase *Call : FunctionCallCalls)
      Call->eraseFromParent();
  }

  // Record the size of OptimizedFunction
  size_t BlocksCount = OptimizedFunction->size();
  BlocksAnalyzedByValueMaterializer.push(BlocksCount);

  // Reattach the unreachable basic blocks to the original root function
  for (BasicBlock *UnreachableBB : UnreachableBBs)
    UnreachableBB->insertInto(TheFunction);

  // Restore the dispatcher in the original function
  JTM.setCFGForm(CFGForm::SemanticPreserving);
  revng_assert(JTM.computeUnreachable().size() == 0);

  // Clear the whitelist
  JTM.clearValueMaterializerPCWhitelist();

  return OptimizedFunction;
}

// Helper to intrinsic promotion
void RootAnalyzer::promoteHelpersToIntrinsics(Function *OptimizedFunction,
                                              IRBuilder<> &Builder) {
  using MapperFunction = std::function<Instruction *(CallInst *)>;
  std::pair<std::vector<StringRef>, MapperFunction> Mapping[] = {
    { { "helper_clz", "helper_clz32", "helper_clz64", "helper_dclz" },
      [&Builder](CallInst *Call) {
        return Builder.CreateBinaryIntrinsic(Intrinsic::ctlz,
                                             Call->getArgOperand(0),
                                             Builder.getFalse());
      } }
  };

  for (auto &[HelperNames, Mapper] : Mapping) {
    for (StringRef HelperName : HelperNames) {
      if (Function *Original = TheModule.getFunction(HelperName)) {

        SmallVector<std::pair<Instruction *, Instruction *>, 16> Replacements;
        for (User *U : Original->users()) {
          if (auto *Call = dyn_cast<CallInst>(U)) {
            if (Call->getParent()->getParent() == OptimizedFunction) {
              Builder.SetInsertPoint(Call);
              Instruction *NewI = Mapper(Call);
              NewI->copyMetadata(*Call);
              Replacements.emplace_back(Call, NewI);
            }
          }
        }

        // Apply replacements
        for (auto &P : Replacements) {
          P.first->replaceAllUsesWith(P.second);
          eraseFromParent(P.first);
        }
      }
    }
  }
}

RootAnalyzer::GlobalToAllocaTy
RootAnalyzer::promoteCSVsToAlloca(Function *OptimizedFunction) {
  GlobalToAllocaTy CSVMap;

  // Collect all the non-PC affecting CSVs
  DenseSet<GlobalVariable *> NonPCCSVs;
  for (GlobalVariable &CSV : FunctionTags::CSV.globals(&TheModule))
    if (not JTM.programCounterHandler()->affectsPC(&CSV))
      NonPCCSVs.insert(&CSV);

  // Create and initialize an alloca per CSV (except for the PC-affecting ones)
  BasicBlock *EntryBB = &OptimizedFunction->getEntryBlock();
  IRBuilder<> AllocaBuilder(&*EntryBB->begin());
  IRBuilder<> InitializeBuilder(EntryBB->getTerminator());

  for (GlobalVariable *CSV : toSortedByName(NonPCCSVs)) {
    Type *CSVType = CSV->getValueType();
    auto *Alloca = AllocaBuilder.CreateAlloca(CSVType, nullptr, CSV->getName());
    CSVMap[CSV] = Alloca;

    // Replace all uses of the CSV within OptimizedFunction with the alloca
    replaceAllUsesInFunctionWith(OptimizedFunction, CSV, Alloca);

    // Initialize the alloca
    InitializeBuilder.CreateStore(createLoad(InitializeBuilder, CSV), Alloca);
  }

  return CSVMap;
}

SummaryCallsBuilder RootAnalyzer::optimize(llvm::Function *OptimizedFunction,
                                           const Features &CommonFeatures) {
  using namespace model::Architecture;
  using namespace model::Register;

  Module *M = OptimizedFunction->getParent();

  auto CSVMap = promoteCSVsToAlloca(OptimizedFunction);

  SummaryCallsBuilder SCB(CSVMap);

  // Put together information about syscalls
  StringRef SyscallHelperName = getSyscallHelper(Model->Architecture());
  Function *SyscallHelper = M->getFunction(SyscallHelperName);
  auto SyscallIDRegister = getSyscallNumberRegister(Model->Architecture());
  StringRef SyscallIDCSVName = getName(SyscallIDRegister);
  GlobalVariable *SyscallIDCSV = M->getGlobalVariable(SyscallIDCSVName);

  // Remove PC initialization from entry block: this is required otherwise the
  // dispatcher will be constant-propagated away
  {
    BasicBlock &Entry = OptimizedFunction->getEntryBlock();
    std::vector<Instruction *> ToDelete;
    for (Instruction &I : Entry)
      if (auto *Store = dyn_cast<StoreInst>(&I))
        if (isa<Constant>(Store->getValueOperand())
            and JTM.programCounterHandler()->affectsPC(Store))
          ToDelete.push_back(&I);

    for (Instruction *I : ToDelete)
      eraseFromParent(I);
  }

  // The StaticDataMemoryOracle provide the contents of memory areas that are
  // mapped statically (i.e., in segments). This is critical to capture, e.g.,
  // virtual tables
  StaticDataMemoryOracle MO(TheModule.getDataLayout(), JTM, CommonFeatures);

  {
    // Note: it is important to let the pass manager go out of scope ASAP:
    //       LazyValueInfo registers a lot of callbacks to get notified when a
    //       Value is destroyed, slowing down OptimizedFunction->eraseFromParent
    //       enormously.

    // The order of the passes, when and how many times they are run are
    // inspired by the -O2 pipeline. You can see it in action as follows:
    //
    //       clang test.c -emit-llvm -o- -Xclang -disable-O0-optnone | \
    //         opt -O2 -S -debug-pass-manager

    FunctionPassManager FPM;

    // Drop all markers except exitTB
    FPM.addPass(DropMarkerCalls({ "exitTB" }));

    // Summarize calls to helpers
    FPM.addPass(DropHelperCallsPass(SyscallHelper, SyscallIDCSV, SCB));

    // TODO: do we still need this?
    FPM.addPass(ShrinkInstructionOperandsPass());

    // Canonicalization
    FPM.addPass(PromotePass());
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(InstCombinePass(InstCombineMaxIterations));

    // This ensures we have in the IR values from constant pools, which will
    // then get collected by collectValuesStoredIntoMemory
    FPM.addPass(ConstantLoadsFolderPass(MO));

    // Running JumpThreading is important to merge multiple instructions with
    // the same predicate in a single "if" (in particular in ARM) and obtain
    // more accurate constraints.
    FPM.addPass(JumpThreadingPass());

    // Shrink instructions

    // InstCombine should not run after TypeShrinking since it undoes its work.
    // Specifically it turns icmp that have been shrank to 32-bit by
    // TypeShrinking back to 64-bits.
    FPM.addPass(TypeShrinking::TypeShrinkingPass());

    // It is important to run EarlyCSE *after* JumpThreading. This has the
    // side effect of invalidating LazyValueInfo (which would otherwise be
    // shared between JumpThreadingPass and ValueMaterializerPass).
    // If we don't run it we get failures on ARM.
    // It is also important to run EarlyCSE after TypeShrinking to factor trunc
    // instructions and have more accurate constraints.
    FPM.addPass(EarlyCSEPass(true));

    // Drop range metadata
    FPM.addPass(DropRangeMetadataPass());

    // Run ValueMaterializer!
    FPM.addPass(ValueMaterializerPass(MO));

    FunctionAnalysisManager FAM;
    FAM.registerPass([]() { return TypeShrinking::BitLivenessPass(); });
    FAM.registerPass([] {
      AAManager AA;
      AA.registerFunctionAnalysis<BasicAA>();
      AA.registerFunctionAnalysis<ScopedNoAliasAA>();

      return AA;
    });

    ModuleAnalysisManager MAM;
    auto MAMFunactionProxyFactory = [&MAM] {
      return ModuleAnalysisManagerFunctionProxy(MAM);
    };
    FAM.registerPass(MAMFunactionProxyFactory);

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    FPM.run(*OptimizedFunction, FAM);
  }

  return SCB;
}

void RootAnalyzer::collectMaterializedValues(AnalysisRegistry &AR) {
  // Iterate over all the ValueMaterializer markers
  Function *ValueMaterializerMarker = AR.aviMarker();
  for (CallBase *Call : callers(ValueMaterializerMarker)) {
    revng_log(Log, "collectMaterializedValues on " << getName(Call));
    LoggerIndent<> Indent(Log);

    // Get the ID from the marker, and then the original instruction and marker
    // type
    Value *LastArgument = Call->getArgOperand(Call->arg_size() - 1);
    uint32_t ValueMaterializerID = getLimitedValue(LastArgument);
    auto TV = AR.rootInstructionById(ValueMaterializerID);
    auto TIT = TV.Type;
    Instruction *I = TV.I;

    revng_log(Log, TrackedInstructionType::getName(TIT));

    // Did ValueMaterializer produce any info?
    auto *T = dyn_cast_or_null<MDTuple>(Call->getMetadata("revng.avi"));
    if (T == nullptr)
      continue;

    // Is this a direct write to PC?
    bool IsComposedIntegerPC = (TIT == TrackedInstructionType::WrittenInPC);

    // We want to register the results only if *all* of them are good
    bool AllValid = true;
    bool AllPCs = true;

    SmallVector<MetaAddress, 16> Targets;
    QuickMetadata QMD(TheModule.getContext());

    // Iterate over all the generated values
    for (const MDOperand &Operand : cast<MDTuple>(T)->operands()) {
      // Extract the value
      auto *Tuple = QMD.extract<MDTuple *>(Operand.get());
      auto SymbolName = QMD.extract<StringRef>(Tuple->getOperand(0).get());
      auto *Value = QMD.extract<ConstantInt *>(Tuple->getOperand(1).get());

      bool HasDynamicSymbol = SymbolName.size() != 0;
      if (not HasDynamicSymbol) {
        // Deserialize value into a MetaAddress, depending on the tracked
        // instruction type
        auto MA = (IsComposedIntegerPC ?
                     MetaAddress::decomposeIntegerPC(Value) :
                     MetaAddress::fromPC(TV.Address, getLimitedValue(Value)));

        if (MA.isInvalid()) {
          AllValid = false;
        } else {
          if (not JTM.isPC(MA))
            AllPCs = false;

          Targets.push_back(MA);
        }
      }
    }

    if (Log.isEnabled()) {
      Log << "Targets:\n";
      for (const MetaAddress &Target : Targets)
        Log << "  " << Target << "\n";
      Log << DoLog;
    }

    // Proceed only if all the results are valid
    if (not AllValid) {
      revng_log(Log, "Not all targets are valid, ignoring.");
      continue;
    }

    // If it's supposed to be a PC, all of them have to be a PC
    bool ShouldBePC = (TIT == TrackedInstructionType::WrittenInPC
                       or TIT == TrackedInstructionType::StoredInMemory);
    if (ShouldBePC and not AllPCs) {
      revng_log(Log,
                "All targets were expected to point to code, but some don't.");
      continue;
    }

    // Register the resulting addresses
    unsigned RegisteredAddresses = 0;

    auto RegisterJT = [this, &RegisteredAddresses](MetaAddress Address,
                                                   JTReason::Values Reason) {
      bool IsNew = not JTM.hasJT(Address);
      if (JTM.registerJT(Address, Reason) != nullptr and IsNew) {
        ++RegisteredAddresses;
      }
    };

    switch (TIT) {
    case TrackedInstructionType::WrittenInPC:
      for (const MetaAddress &MA : Targets)
        RegisterJT(MA, JTReason::PCStore);
      WrittenInPCStatistics.push(RegisteredAddresses);
      break;

    case TrackedInstructionType::StoredInMemory:
      for (const MetaAddress &MA : Targets)
        RegisterJT(MA, JTReason::MemoryStore);
      StoredInMemoryStatistics.push(RegisteredAddresses);
      break;

    case TrackedInstructionType::StoreTarget:
    case TrackedInstructionType::LoadTarget:
      for (const MetaAddress &MA : Targets)
        if (JTM.markJT(MA, JTReason::LoadAddress))
          ++RegisteredAddresses;
      LoadAddressStatistics.push(RegisteredAddresses);
      break;

    case TrackedInstructionType::Invalid:
      revng_abort();
    }

    if (TIT == TrackedInstructionType::WrittenInPC) {
      // This is a call to `exit_tb`, transfer the revng.avi metadata on the
      // call as revng.targets for later processing
      revng_assert(TV.I != nullptr);
      TV.I->setMetadata("revng.targets", T);
      DetectedEdgesStatistics.push(Targets.size());
      revng_log(NewEdgesLog,
                Targets.size() << " targets from " << getName(Call));
    }
  }
}

using JTM2 = RootAnalyzer;

void JTM2::collectValuesStoredIntoMemory(Function *F,
                                         const Features &CommonFeatures) {
  for (Instruction &I : llvm::instructions(F)) {
    if (auto *Store = dyn_cast<StoreInst>(&I)) {
      auto *Pointer = Store->getPointerOperand();
      auto *Address = dyn_cast<ConstantInt>(Store->getValueOperand());
      if (isMemory(Pointer) and Address != nullptr
          and JTM.programCounterHandler()->isPCSizedType(Address->getType())) {
        auto MA = MetaAddress::fromPC(Address->getLimitedValue(),
                                      CommonFeatures);
        if (MA.isValid()) {
          JTM.registerJT(MA, JTReason::MemoryStore);
        }
      }
    }
  }
}

static MetaAddress::Features findCommonFeatures(Function *F) {
  bool First = true;
  MetaAddress::Features Result;
  for (CallBase *NewPCCall :
       callersIn(F->getParent()->getFunction("newpc"), F)) {
    MetaAddress Address = addressFromNewPC(NewPCCall);

    if (First) {
      Result = Address.features();
    } else {
      // TODO: once we switch to multi-binary run ValueMaterializer once per
      //       each feature set
      revng_assert(Result == Address.features());
    }
  }

  return Result;
}

void RootAnalyzer::cloneOptimizeAndHarvest(Function *TheFunction) {
  updateCSAA();

  ValueToValueMapTy OldToNew;
  Function *OptimizedFunction = createTemporaryRoot(TheFunction, OldToNew);

  MetaAddress::Features CommonFeatures = findCommonFeatures(OptimizedFunction);

  AnalysisRegistry AR(&TheModule);

  // Register for analysis the value written in the PC before each exit_tb call
  IRBuilder<> Builder(TheModule.getContext());
  for (CallBase *Call : callersIn(JTM.exitTB(), TheFunction)) {
    BasicBlock *BB = Call->getParent();
    auto It = OldToNew.find(Call);
    if (It == OldToNew.end())
      continue;
    Builder.SetInsertPoint(cast<CallInst>(&*It->second));
    ProgramCounterHandler *PCH = JTM.programCounterHandler();
    Instruction *ComposedIntegerPC = PCH->composeIntegerPC(Builder);
    AR.registerValue(getPC(Call).first,
                     Call,
                     ComposedIntegerPC,
                     TrackedInstructionType::WrittenInPC);
  }

  promoteHelpersToIntrinsics(OptimizedFunction, Builder);

  // Replace calls to newpc with stores to the PC
  SmallVector<CallBase *, 16> ToErase;
  for (CallBase *Call :
       callersIn(TheModule.getFunction("newpc"), OptimizedFunction)) {
    JTM.programCounterHandler()->expandNewPC(Call);
    ToErase.push_back(Call);
  }

  for (CallBase *Call : ToErase)
    eraseFromParent(Call);

  // Optimize the hell out of it and collect the possible values of indirect
  // branches.
  auto SCB = optimize(OptimizedFunction, CommonFeatures);

  revng::verify(OptimizedFunction);

  // Collect the results
  collectMaterializedValues(AR);

  // Collect pointer-sized values being stored in memory
  collectValuesStoredIntoMemory(OptimizedFunction, CommonFeatures);

  // Drop the optimized function
  eraseFromParent(OptimizedFunction);

  // Drop temporary functions
  SCB.cleanup();
}
