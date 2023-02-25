/// \file CFGAnalyzer.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/BasicAnalyses/RemoveNewPCCalls.h"
#include "revng/EarlyFunctionAnalysis/AAWriterPass.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/IndirectBranchInfoPrinterPass.h"
#include "revng/EarlyFunctionAnalysis/PromoteGlobalToLocalVars.h"
#include "revng/EarlyFunctionAnalysis/SegregateDirectStackAccesses.h"
#include "revng/Support/TemporaryLLVMOption.h"

using namespace llvm;
using namespace llvm::cl;

static opt<std::string> AAWriterPath("aa-writer",
                                     desc("Dump to disk the outlined functions "
                                          "with annotated alias info."),
                                     value_desc("filename"));

static opt<std::string> IndirectBranchInfoSummaryPath("indirect-branch-info-"
                                                      "summary",
                                                      desc("Write the results "
                                                           "of SA2 on disk."),
                                                      value_desc("filename"));

static Logger<> Log("cfg-analyzer");

static MetaAddress getFinalAddressOfBasicBlock(llvm::BasicBlock *BB) {
  auto [End, Size] = getPC(BB->getTerminator());
  return End + Size;
}

static UpcastablePointer<efa::FunctionEdgeBase>
makeCall(MetaAddress Destination) {
  using ReturnType = UpcastablePointer<efa::FunctionEdgeBase>;
  return ReturnType::make<efa::CallEdge>(BasicBlockID(Destination),
                                         efa::FunctionEdgeType::FunctionCall);
};

static UpcastablePointer<efa::FunctionEdgeBase>
makeEdge(BasicBlockID Destination, efa::FunctionEdgeType::Values Type) {
  revng_assert(Type != efa::FunctionEdgeType::FunctionCall);
  efa::FunctionEdge *Result = nullptr;
  using ReturnType = UpcastablePointer<efa::FunctionEdgeBase>;
  return ReturnType::make<efa::FunctionEdge>(Destination, Type);
};

static UpcastablePointer<efa::FunctionEdgeBase>
makeIndirectEdge(efa::FunctionEdgeType::Values Type) {
  return makeEdge(BasicBlockID::invalid(), Type);
};

namespace efa {

/// Indexes for arguments of indirect_branch_info
enum {
  CallerBlockIDIndex,
  CalledSymbolIndex,
  JumpsToReturnAddressIndex,
  StackPointerOffsetIndex,
  ReturnValuePreservedIndex,
  PreservedRegistersIndex
};

static efa::BasicBlock &
blockFromIndirectBranchInfo(CallBase *CI, SortedVector<efa::BasicBlock> &CFG) {
  auto *BlockIDArgument = CI->getArgOperand(CallerBlockIDIndex);
  return CFG.at(BasicBlockID::fromValue(BlockIDArgument));
}

static std::unique_ptr<llvm::raw_ostream>
streamFromOption(const opt<std::string> &Option) {
  if (Option.getNumOccurrences() == 1) {
    std::ifstream File(Option.c_str());
    if (File.is_open()) {
      int Status = std::remove(Option.c_str());
      revng_assert(Status == 0);
    }

    std::error_code EC;
    auto Result = std::make_unique<raw_fd_ostream>(Option,
                                                   EC,
                                                   llvm::sys::fs::OF_Append);
    revng_assert(!EC);
    return Result;
  } else {
    return std::make_unique<llvm::raw_null_ostream>();
  }
}

CFGAnalyzer::CFGAnalyzer(llvm::Module &M,
                         GeneratedCodeBasicInfo &GCBI,
                         const TupleTree<model::Binary> &Binary,
                         const FunctionSummaryOracle &Oracle) :
  M(M),
  GCBI(GCBI),
  PCH(GCBI.programCounterHandler()),
  Oracle(Oracle),
  Binary(Binary),
  PreCallHook(createCallMarkerType(M), "precall_hook", &M),
  PostCallHook(PreCallHook.get()->getFunctionType(), "postcall_hook", &M),
  RetHook(createRetMarkerType(M), "retcall_hook", &M),
  Summarizer(&M,
             PreCallHook.get(),
             PostCallHook.get(),
             RetHook.get(),
             GCBI.spReg()),
  Outliner(M, GCBI, Oracle),
  OpaqueBranchConditionsPool(&M, false),
  OutputAAWriter(streamFromOption(AAWriterPath)),
  OutputIBI(streamFromOption(IndirectBranchInfoSummaryPath)) {

  // Collect all ABI CSVs except for the stack pointer
  for (GlobalVariable *CSV : GCBI.abiRegisters())
    if (CSV != nullptr and not GCBI.isSPReg(CSV))
      ABICSVs.emplace_back(CSV);

  // Prepare header for debugging information about indirect branch infos
  *OutputIBI << "name,ra,fso,address";
  for (const auto &Reg : ABICSVs)
    *OutputIBI << "," << Reg->getName();
  *OutputIBI << "\n";
}

OutlinedFunction CFGAnalyzer::outline(llvm::BasicBlock *Entry) {
  OutlinedFunction Result = Outliner.outline(Entry, &Summarizer);

  // Make sure we start a new block before a PreCallHook
  auto IsFirst = [](llvm::Instruction *I) {
    return I->getParent()->getFirstNonPHI() == I;
  };
  for (llvm::CallBase *Call : callers(PreCallHook.get()))
    if (not IsFirst(Call))
      Call->getParent()->splitBasicBlock(Call);

  // Make sure we start a new block for each jump target
  auto IsJumpTarget = [](llvm::CallBase *Call) {
    auto IsJumpTarget = NewPCArguments::IsJumpTarget;
    return getLimitedValue(&*Call->getArgOperand(IsJumpTarget)) == 1;
  };
  for (llvm::CallBase *Call : callers(M.getFunction("newpc")))
    if (IsJumpTarget(Call) and not IsFirst(Call))
      Call->getParent()->splitBasicBlock(Call);

  return Result;
}

llvm::FunctionType *CFGAnalyzer::createCallMarkerType(llvm::Module &M) {
  auto &Context = M.getContext();
  Type *I8Ptr = Type::getInt8PtrTy(Context);
  Type *BoolType = Type::getInt1Ty(Context);
  Type *Void = Type::getVoidTy(Context);
  return llvm::FunctionType::get(Void,
                                 { I8Ptr, I8Ptr, I8Ptr, BoolType },
                                 false);
}

llvm::FunctionType *CFGAnalyzer::createRetMarkerType(llvm::Module &M) {
  auto &Context = M.getContext();
  Type *Void = Type::getVoidTy(Context);
  Type *I8Ptr = Type::getInt8PtrTy(Context);
  return llvm::FunctionType::get(Void, { I8Ptr }, false);
}

std::optional<UpcastablePointer<efa::FunctionEdgeBase>>
CFGAnalyzer::handleCall(llvm::CallInst *PreCallHookCall) {
  using namespace llvm;

  //
  // Extract parameters
  //

  // Is this a direct call to a function defined in the model?
  Value *CalleePC = PreCallHookCall->getArgOperand(1);
  MetaAddress CalleeAddress = MetaAddress::invalid();
  bool IsDirectCall = false;
  auto Address = BasicBlockID::fromValue(CalleePC).notInlinedAddress();
  if (Address.isValid()) {
    IsDirectCall = Binary->Functions().count(Address) != 0;
    if (IsDirectCall)
      CalleeAddress = Address;
  }

  // Symbol name
  StringRef SymbolName;
  auto *SymbolArgument = PreCallHookCall->getArgOperand(2);
  bool IsDynamicCall = not isa<ConstantPointerNull>(SymbolArgument);
  if (IsDynamicCall)
    SymbolName = extractFromConstantStringPtr(SymbolArgument);

  // Tail call?
  bool IsTailCall = (getLimitedValue(PreCallHookCall->getArgOperand(3)) == 1);

  // Here we handle only direct, non-tail calls
  if (not IsDirectCall and IsTailCall)
    return std::nullopt;

  if (IsDynamicCall)
    revng_assert(not IsDirectCall);

  //
  // Construct call edge
  //
  UpcastablePointer<efa::FunctionEdgeBase> Edge = makeCall(CalleeAddress);

  auto *CE = cast<efa::CallEdge>(Edge.get());
  CE->IsTailCall() = IsTailCall;
  if (IsDynamicCall)
    CE->DynamicFunction() = SymbolName.str();

  return Edge;
}

SortedVector<efa::BasicBlock>
CFGAnalyzer::collectDirectCFG(OutlinedFunction *OF) {
  using namespace llvm;
  using llvm::BasicBlock;

  revng_log(Log, "collectDirectCFG(" << OF->Function->getName().str() << ")");
  LoggerIndent<> Indent(Log);

  SortedVector<efa::BasicBlock> CFG;

  for (BasicBlock &BB : *OF->Function) {
    if (isJumpTarget(&BB)) {
      // Create a efa::BasicBlock for each jump target
      BasicBlockID ID = getBasicBlockID(&BB);
      MetaAddress End = getFinalAddressOfBasicBlock(&BB);
      revng_assert(End.isValid());

      efa::BasicBlock Block{ ID };
      Block.End() = End;
      Block.InlinedFrom() = OF->InlinedFunctionsByIndex.at(ID.inliningIndex());
      bool ReachesUnexpectedPC = false;

      // Initialize the end address of the basic block, we'll extend it later on
      revng_log(Log,
                "Creating block starting at " << ID.toString()
                                              << " (preliminary ending is "
                                              << Block.End().toString() << ")");
      LoggerIndent<> Indent(Log);

      OnceQueue<BasicBlock *> Queue;
      Queue.insert(&BB);

      bool ReachesUnreachable = false;

      while (!Queue.empty()) {
        BasicBlock *Current = Queue.pop();
        revng_log(Log, "Analyzing block " << getName(Current));

        // If this block belongs to a single `newpc`, record its address
        MetaAddress CurrentBlockEnd = getFinalAddressOfBasicBlock(Current);
        if (CurrentBlockEnd.isValid() and CurrentBlockEnd > Block.End()) {
          revng_log(Log,
                    "Extending block end to " << CurrentBlockEnd.toString());
          Block.End() = CurrentBlockEnd;
        }

        if (isa<UnreachableInst>(Current->getTerminator())) {
          revng_log(Log, "Reaches unreachable");
          ReachesUnreachable = true;
        } else {
          revng_assert(succ_size(Current) > 0);
        }

        revng_log(Log, "Considering successors");
        LoggerIndent<> Indent2(Log);

        for (BasicBlock *Succ : successors(Current)) {
          revng_log(Log, "Considering successor " << getName(Succ));
          if (isa<ReturnInst>(Succ->getTerminator())) {
            revng_log(Log, "It's a ret");
            // Did we meet the end of the cloned function? Do nothing
            revng_assert(Succ->size() == 1);
          } else if (auto *Call = getCallTo(&*Succ->begin(),
                                            PreCallHook.get())) {
            // Handle edge for regular function calls
            if (auto MaybeEdge = handleCall(Call); MaybeEdge) {
              revng_log(Log, "It's a direct call, emitting a CallEdge");
              Block.Successors().insert(*MaybeEdge);
            }
          } else if (isJumpTarget(Succ)) {
            // TODO: handle situation in which it's a *direct* tail call.
            //       We might need an IBI here to know if the stack position is
            //       compatible with a tail call.
            BasicBlockID Destination = getBasicBlockID(Succ);
            revng_log(Log,
                      "It's a jump target: emitting a DirectBranch to "
                        << Destination.toString());
            auto Edge = makeEdge(Destination,
                                 efa::FunctionEdgeType::DirectBranch);
            Block.Successors().insert(Edge);
          } else if (Succ == OF->UnexpectedPCCloned) {
            revng_log(Log, "Reaches UnexpectedPC");
            ReachesUnexpectedPC = true;
          } else {

            revng_log(Log, "Nothing special, enqueue");
            // Not one of the cases above? Enqueue the successor basic block.
            Queue.insert(Succ);
          }
        }
      }

      bool HasNoSuccessor = Block.Successors().size() == 0;
      if (HasNoSuccessor) {
        if (ReachesUnreachable) {
          // If we reach any unreachable instruction, add a single unreachable
          // edge
          revng_log(Log, "Reaches unreachable, add to successors");
          revng_assert(Block.Successors().empty());
          using namespace efa::FunctionEdgeType;
          auto NewEdge = makeEdge(BasicBlockID::invalid(), Unreachable);
          Block.Successors().insert(NewEdge);
        } else if (ReachesUnexpectedPC) {
          // successor of the current basic block.
          revng_log(Log,
                    "No other successors other than UnexpectedPC, emitting "
                    "LongJmp");
          auto Edge = makeEdge(BasicBlockID::invalid(),
                               efa::FunctionEdgeType::LongJmp);
          Block.Successors().insert(Edge);
        }
      }

      // Commit the newly created block to the CFG
      CFG.insert(Block);
    }
  }

  return CFG;
}

CFGAnalyzer::State CFGAnalyzer::loadState(llvm::IRBuilder<> &Builder) const {
  using namespace llvm;
  LLVMContext &Context = M.getContext();

  // Load the stack pointer
  auto *SP0 = createLoad(Builder, GCBI.spReg());

  // Load the return address
  Value *ReturnAddress = nullptr;
  if (GlobalVariable *Register = GCBI.raReg()) {
    ReturnAddress = createLoad(Builder, Register);
  } else {
    auto *OpaquePointer = PointerType::get(Context, 0);
    auto *StackPointerPointer = Builder.CreateIntToPtr(SP0, OpaquePointer);
    ReturnAddress = Builder.CreateLoad(GCBI.pcReg()->getValueType(),
                                       StackPointerPointer);
  }

  // Load the PC
  auto LLVMArchitecture = toLLVMArchitecture(Binary->Architecture());
  auto DissectedPC = PCH->dissectJumpablePC(Builder,
                                            ReturnAddress,
                                            LLVMArchitecture);
  Value *IntegerPC = MetaAddress::composeIntegerPC(Builder,
                                                   DissectedPC[0],
                                                   DissectedPC[1],
                                                   DissectedPC[2],
                                                   DissectedPC[3]);

  // Load all CSVs
  SmallVector<Value *, 16> CSVs;
  Type *IsRetTy = Type::getInt128Ty(Context);
  for (auto *CSR : ABICSVs) {
    auto *V = createLoad(Builder, CSR);
    CSVs.emplace_back(V);
  }

  return { SP0, IntegerPC, CSVs };
}

void CFGAnalyzer::createIBIMarker(OutlinedFunction *OutlinedFunction) {
  using namespace llvm;
  using llvm::BasicBlock;

  IRBuilder<> Builder(&OutlinedFunction->Function->getEntryBlock().front());

  State Initial = loadState(Builder);

  //
  // Create IBI for this function
  //
  LLVMContext &Context = M.getContext();
  auto *IntTy = GCBI.spReg()->getValueType();
  Type *I8Ptr = Type::getInt8PtrTy(Context);
  SmallVector<Type *, 16> ArgTypes;
  ArgTypes.resize(PreservedRegistersIndex);
  ArgTypes[CallerBlockIDIndex] = I8Ptr;
  ArgTypes[CalledSymbolIndex] = I8Ptr;
  ArgTypes[JumpsToReturnAddressIndex] = Initial.ReturnPC->getType();
  ArgTypes[StackPointerOffsetIndex] = Initial.StackPointer->getType();
  ArgTypes[ReturnValuePreservedIndex] = Initial.ReturnPC->getType();
  for (auto *CSV : ABICSVs)
    ArgTypes.emplace_back(CSV->getValueType());

  auto *FTy = llvm::FunctionType::get(IntTy, ArgTypes, false);
  auto *IBI = Function::Create(FTy,
                               GlobalValue::ExternalLinkage,
                               "indirect_branch_info",
                               M);
  OutlinedFunction->IndirectBranchInfoMarker = UniqueValuePtr<Function>(IBI);
  OutlinedFunction->IndirectBranchInfoMarker->addFnAttr(Attribute::NoUnwind);
  OutlinedFunction->IndirectBranchInfoMarker->addFnAttr(Attribute::NoReturn);

  // When an indirect jump is encountered we load the state at that point and
  // compare it against the initial state

  // Initalize markers for ABI analyses and set up the branches on which
  // `indirect_branch_info` will be installed.
  SmallVector<Instruction *, 16> BranchesToAnyPC;
  if (OutlinedFunction->AnyPCCloned == nullptr)
    return;

  SmallVector<BasicBlock *, 16> Predecessors;
  for (BasicBlock *Predecessor : predecessors(OutlinedFunction->AnyPCCloned))
    Predecessors.push_back(Predecessor);

  for (BasicBlock *Predecessor : Predecessors) {
    Instruction *Term = Predecessor->getTerminator();

    // Create a block and jump there
    auto *IBIBlock = BasicBlock::Create(Context,
                                        Predecessor->getName()
                                          + Twine("_indirect_branch_info"),
                                        OutlinedFunction->Function.get(),
                                        nullptr);

    Term->replaceUsesOfWith(OutlinedFunction->AnyPCCloned, IBIBlock);

    Builder.SetInsertPoint(IBIBlock);

    // Load the state
    State Final = loadState(Builder);

    // Prepare the arguments for the indirect_branch_info probe call
    SmallVector<Value *, 16> ArgValues;
    ArgValues.resize(PreservedRegistersIndex);

    // Record the MetaAddress of the caller
    auto NewPCID = getBasicBlockID(getJumpTargetBlock(Term->getParent()));
    revng_assert(NewPCID.isValid());
    ArgValues[CallerBlockIDIndex] = NewPCID.toValue(getModule(Term));

    // Record the name of the symbol, if any
    using CPN = ConstantPointerNull;
    Value *SymbolName = CPN::get(Type::getInt8PtrTy(Context));
    for (Instruction &I : reverse(*Term->getParent())) {
      if (auto *Call = getCallTo(&I, PreCallHook.get())) {
        SymbolName = Call->getArgOperand(2);
        break;
      }
    }
    ArgValues[CalledSymbolIndex] = SymbolName;

    // Compute the difference between the final PC at this program point and
    // the expected return address
    auto *FinalPC = PCH->composeIntegerPC(Builder);
    ArgValues[JumpsToReturnAddressIndex] = Builder.CreateSub(FinalPC,
                                                             Initial.ReturnPC);

    // Compute the difference between the initial stack pointer and the stack
    // pointer at this program point
    Value *Difference = Builder.CreateSub(Final.StackPointer,
                                          Initial.StackPointer);
    ArgValues[StackPointerOffsetIndex] = Difference;

    // Check if the expected ReturnPC has been preserved
    ArgValues[ReturnValuePreservedIndex] = Builder.CreateSub(Final.ReturnPC,
                                                             Initial.ReturnPC);

    // Compute the difference between the initial and final values of the CSVs
    for (const auto &[Initial, End] : zip(Initial.CSVs, Final.CSVs)) {
      auto *ABIRegistersDifference = Builder.CreateSub(Initial, End);
      ArgValues.emplace_back(ABIRegistersDifference);
    }

    // Emit the `indirect_branch_info` call
    Builder.CreateCall(OutlinedFunction->IndirectBranchInfoMarker.get(),
                       ArgValues);
    Builder.CreateUnreachable();
  }
}

void CFGAnalyzer::opaqueBranchConditions(llvm::Function *F,
                                         llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  for (auto &BB : *F) {
    auto *Term = BB.getTerminator();
    if ((isa<BranchInst>(Term) && cast<BranchInst>(Term)->isConditional())
        || isa<SwitchInst>(Term)) {
      Value *Condition = isa<BranchInst>(Term) ?
                           cast<BranchInst>(Term)->getCondition() :
                           cast<SwitchInst>(Term)->getCondition();

      OpaqueBranchConditionsPool.addFnAttribute(Attribute::NoUnwind);
      auto MemoryEffects = MemoryEffects::inaccessibleMemOnly();
      OpaqueBranchConditionsPool.setMemoryEffects(MemoryEffects);
      OpaqueBranchConditionsPool.addFnAttribute(Attribute::WillReturn);

      auto *FTy = llvm::FunctionType::get(Condition->getType(),
                                          { Condition->getType() },
                                          false);

      auto *OpaqueTrueCallee = OpaqueBranchConditionsPool.get(FTy,
                                                              FTy,
                                                              "opaque_true");

      IRB.SetInsertPoint(Term);
      auto *RetVal = IRB.CreateCall(OpaqueTrueCallee, { Condition });

      if (isa<BranchInst>(Term))
        cast<BranchInst>(Term)->setCondition(RetVal);
      else
        cast<SwitchInst>(Term)->setCondition(RetVal);
    }
  }
}

void CFGAnalyzer::materializePCValues(llvm::Function *F,
                                      llvm::IRBuilder<> &IRB) {
  using namespace llvm;

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *Call = getCallTo(&I, "newpc")) {
        MetaAddress NewPC = blockIDFromNewPC(Call).start();
        IRB.SetInsertPoint(Call);
        PCH->setPC(IRB, NewPC);
      }
    }
  }
}

void CFGAnalyzer::runOptimizationPipeline(llvm::Function *F) {
  using namespace llvm;

  // Some LLVM passes used later in the pipeline scan for cut-offs, meaning that
  // further computation may not be done when they are reached; making some
  // optimizations opportunities missed. Hence, we set the involved thresholds
  // (e.g., the maximum value that MemorySSA uses to take into account
  // stores/phis) to have initial unbounded value.
  static constexpr const char *MemSSALimit = "memssa-check-limit";
  static constexpr const char *MemDepBlockLimit = "memdep-block-scan-limit";

  using TemporaryUOption = TemporaryLLVMOption<unsigned>;
  TemporaryUOption MemSSALimitOption(MemSSALimit, UINT_MAX);
  TemporaryUOption MemDepBlockLimitOption(MemDepBlockLimit, UINT_MAX);

  // TODO: break it down in the future, and check if some passes can be dropped
  {
    FunctionPassManager FPM;

    // First stage: simplify the IR, promote the CSVs to local variables,
    // compute subexpressions elimination and resolve redundant expressions in
    // order to compute the stack height.
    FPM.addPass(RemoveNewPCCallsPass());
    FPM.addPass(RemoveHelperCallsPass());
    FPM.addPass(PromoteGlobalToLocalPass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(SROA());
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(JumpThreadingPass());
    FPM.addPass(UnreachableBlockElimPass());
    FPM.addPass(InstCombinePass(true));
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(MergedLoadStoreMotionPass());
    FPM.addPass(GVN());

    // Second stage: add alias analysis info and canonicalize `i2p` + `add` into
    // `getelementptr` instructions. Since the IR may change remarkably, another
    // round of passes is necessary to take more optimization opportunities.
    FPM.addPass(SegregateDirectStackAccessesPass());
    FPM.addPass(EarlyCSEPass(true));
    FPM.addPass(InstCombinePass(true));
    FPM.addPass(GVN());

    // Third stage: if enabled, serialize the results and dump the functions on
    // disk with the alias information included as comments.
    if (IndirectBranchInfoSummaryPath.getNumOccurrences() == 1)
      FPM.addPass(IndirectBranchInfoPrinterPass(*OutputIBI));

    if (AAWriterPath.getNumOccurrences() == 1)
      FPM.addPass(AAWriterPass(*OutputAAWriter));

    ModuleAnalysisManager MAM;

    FunctionAnalysisManager FAM;
    FAM.registerPass([] {
      AAManager AA;
      AA.registerFunctionAnalysis<BasicAA>();
      AA.registerFunctionAnalysis<ScopedNoAliasAA>();

      return AA;
    });
    FAM.registerPass([this] {
      using LMA = LoadModelAnalysis;
      return LMA::fromModelWrapper(Binary);
    });
    FAM.registerPass([&] { return GeneratedCodeBasicInfoAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);
    PB.registerModuleAnalyses(MAM);

    FPM.run(*F, FAM);
  }
}

class ClobberedRegistersRegistry {
public:
  using CSVVector = llvm::SmallVector<GlobalVariable *, 16>;

private:
  const CSVVector &ABICSVs;
  std::set<GlobalVariable *> ClobberedRegs;

public:
  ClobberedRegistersRegistry(const CSVVector &ABICSVs) : ABICSVs(ABICSVs) {}

public:
  const std::set<GlobalVariable *> &getClobberedRegisters() const {
    return ClobberedRegs;
  }

public:
  void recordClobberedRegisters(llvm::CallBase *CI) {
    using namespace llvm;
    for (unsigned I = PreservedRegistersIndex; I < CI->arg_size(); ++I) {
      auto *Register = dyn_cast<ConstantInt>(CI->getArgOperand(I));
      if (Register == nullptr or Register->getZExtValue() != 0)
        ClobberedRegs.insert(ABICSVs[I - PreservedRegistersIndex]);
    }
  }

  void add(const std::set<GlobalVariable *> &Clobbered) {
    for (auto *GV : Clobbered)
      ClobberedRegs.insert(GV);
  }
};

/// Elect a final stack offset to tell whether the function is leaving
/// the stack pointer higher than it was at the function entry.
static std::optional<int64_t> electFSO(const auto &MaybeReturns) {
  auto It = std::min_element(MaybeReturns.begin(),
                             MaybeReturns.end(),
                             [](const auto &LHS, const auto &RHS) {
                               return std::get<1>(LHS) < std::get<1>(RHS);
                             });
  if (It == MaybeReturns.end())
    return {};
  return std::get<1>(*It);
}

FunctionSummary CFGAnalyzer::milkInfo(OutlinedFunction *OutlinedFunction,
                                      SortedVector<efa::BasicBlock> &&CFG) {
  using namespace llvm;
  using namespace efa::FunctionEdgeType;
  using namespace model::Architecture;
  int64_t CallPushSize = getCallPushSize(Binary->Architecture());

  using EdgeType = UpcastablePointer<efa::FunctionEdgeBase>;
  SmallVector<std::pair<CallBase *, EdgeType>, 4> IBIResult;

  // Temporary bins for IBIs that need extra processing
  SmallVector<std::tuple<CallBase *, int64_t, const FunctionSummary *>, 4>
    TailCalls;
  SmallVector<std::pair<CallBase *, int64_t>, 4> MaybeReturns;
  SmallVector<std::tuple<CallBase *, int64_t, const FunctionSummary *>, 4>
    MaybeIndirectTailCalls;

  for (CallBase *CI :
       callers(OutlinedFunction->IndirectBranchInfoMarker.get())) {
    if (CI->getParent()->getParent() != OutlinedFunction->Function.get())
      continue;

    Value *Argument = nullptr;
    bool JumpsToReturnAddress = false;

    // Is this indirect branch targeting jumping to the return address?
    {
      Argument = CI->getArgOperand(JumpsToReturnAddressIndex);
      auto *ConstantOffset = dyn_cast<ConstantInt>(Argument);
      if (ConstantOffset)
        JumpsToReturnAddress = ConstantOffset->getSExtValue() == 0;
    }

    efa::BasicBlock Block = blockFromIndirectBranchInfo(CI, CFG);

    // Is this a tail call? If so, we are very interested in the FSO since
    // it's useful to determin the FSO of the caller
    auto *CalledSymbolArgument = CI->getArgOperand(CalledSymbolIndex);
    StringRef CalledSymbol = extractFromConstantStringPtr(CalledSymbolArgument);
    auto [Summary, IsTailCall] = Oracle.getCallSite(OutlinedFunction->Address,
                                                    Block.ID(),
                                                    MetaAddress::invalid(),
                                                    CalledSymbol);
    revng_assert(Summary->ElectedFSO.has_value());

    Argument = CI->getArgOperand(StackPointerOffsetIndex);
    auto *StackPointerOffset = dyn_cast<ConstantInt>(Argument);
    if (StackPointerOffset != nullptr) {
      int64_t FSO = StackPointerOffset->getSExtValue();
      if (JumpsToReturnAddress) {
        if (FSO >= CallPushSize) {
          // We're jumping to the return address with a non-negative stack
          // offset. This might be a return instruction. Note it down as a
          // candidate
          MaybeReturns.emplace_back(CI, FSO);
        } else {
          // We're jumping to the return address but we're leaving the stack
          // higher than it was initially, it might be the return instruction
          // of a function to inline. Record it as BrokenReturn.
          IBIResult.emplace_back(CI, makeIndirectEdge(BrokenReturn));
        }
      } else if (IsTailCall) {
        // We have a tail call for which the FSO is known! We can exploit this
        // to know this function's FSO!
        TailCalls.emplace_back(CI, FSO + *Summary->ElectedFSO, Summary);
      } else {
        Argument = CI->getArgOperand(ReturnValuePreservedIndex);
        auto *ReturnValuePreserved = dyn_cast<ConstantInt>(Argument);
        if (ReturnValuePreserved != nullptr
            and getLimitedValue(ReturnValuePreserved) == 0) {
          // Here the stack pointer offset is known and the value of the
          // return address (in the link register/top of the stack) has been
          // preserved.
          // This is definitely not a return but it might be an indirect
          // tail call.
          MaybeIndirectTailCalls.emplace_back(CI,
                                              FSO + *Summary->ElectedFSO,
                                              Summary);
        } else {
          // The return address is not preserved, something fishy is going on
          IBIResult.emplace_back(CI, makeIndirectEdge(LongJmp));
        }
      }
    } else {
      // We're leaving the stack pointer in an unknown state
      IBIResult.emplace_back(CI, makeIndirectEdge(LongJmp));
    }
  }

  // Elect a final stack offset
  std::optional<int64_t> MaybeWinFSO;

  // Elect a set of clobbered registers on all return points (i.e., actual
  // returns + actual tail calls)

  ClobberedRegistersRegistry ClobberedRegisters(ABICSVs);

  //
  // Process TailCalls
  //

  // Check all tail calls have voted the same FSO, emit a warning otherwise
  if (TailCalls.size() > 0) {
    MaybeWinFSO = electFSO(TailCalls);
    bool Different = false;
    for (auto [CI, FSO, Summary] : TailCalls) {
      if (FSO == *MaybeWinFSO) {
        ClobberedRegisters.recordClobberedRegisters(CI);
        ClobberedRegisters.add(Summary->ClobberedRegisters);
      } else {
        // We found a tail call that leads to a FSO that's not coherent with
        // the elected one. Purge it turning it into a LongJmp.
        Different = true;
        efa::BasicBlock &Block = blockFromIndirectBranchInfo(CI, CFG);
        revng_assert(Block.Successors().size() == 1);
        auto OldEdge = cast<efa::CallEdge>(Block.Successors().begin()->get());
        revng_assert(OldEdge->IsTailCall());
        Block.Successors() = { makeIndirectEdge(LongJmp) };
      }
    }

    if (Different) {
      revng_log(Log,
                "We have multiple tail calls which lead to uncoherent final "
                "stack sizes");
    }
  }

  //
  // Process MaybeReturns
  //

  // Use candidate returns to elect FSO, if necessary
  if (MaybeReturns.size() > 0) {
    if (not MaybeWinFSO.has_value())
      MaybeWinFSO = electFSO(MaybeReturns);
    revng_assert(MaybeWinFSO.has_value());
  }

  // Divide entries in MaybeReturns in Returns and BrokenReturns
  for (const auto &[CI, FSO] : MaybeReturns) {
    if (MaybeWinFSO.has_value() and FSO == *MaybeWinFSO) {
      // It's a return!
      IBIResult.emplace_back(CI, makeIndirectEdge(Return));
      ClobberedRegisters.recordClobberedRegisters(CI);
    } else {
      // It's jumping to the return address but leaves the stack pointer in a
      // state not compatible with being a return instruction, it might be the
      // return instruction of a function to inline. Record it as BrokenReturn.
      IBIResult.emplace_back(CI, makeIndirectEdge(BrokenReturn));
    }
  }

  //
  // Process MaybeIndirectTailCalls
  //

  // If no FSO was elected, retry focusing  on MaybeIndirectTailCalls
  if (not MaybeWinFSO.has_value()) {
    revng_assert(MaybeReturns.size() == 0);
    MaybeWinFSO = electFSO(MaybeIndirectTailCalls);
  }

  for (const auto &[CI, FSO, Summary] : MaybeIndirectTailCalls) {
    if (MaybeWinFSO.has_value() && FSO == *MaybeWinFSO) {
      auto NewEdge = makeCall(MetaAddress::invalid());
      auto *Call = cast<efa::CallEdge>(NewEdge.get());
      Call->IsTailCall() = true;
      auto *Argument = CI->getArgOperand(CalledSymbolIndex);
      Call->DynamicFunction() = extractFromConstantStringPtr(Argument);
      IBIResult.emplace_back(CI, std::move(NewEdge));
      ClobberedRegisters.recordClobberedRegisters(CI);
      ClobberedRegisters.add(Summary->ClobberedRegisters);
    } else {
      IBIResult.emplace_back(CI, makeIndirectEdge(LongJmp));
    }
  }

  //
  // Commit  IBIResults to the CFG
  //
  for (const auto &[CI, Edge] : IBIResult) {
    efa::BasicBlock &Block = blockFromIndirectBranchInfo(CI, CFG);
    Block.Successors().insert(std::move(Edge));
  }

  // Collect summary for information
  using efa::CallEdge;
  bool FoundReturn = false;
  bool FoundBrokenReturn = false;
  int BrokenReturnCount = 0;
  int NoReturnCount = 0;
  for (const auto &[CI, Edge] : IBIResult) {
    if (Edge->Type() == Return) {
      FoundReturn = true;
    } else if (Edge->Type() == FunctionCall
               and cast<CallEdge>(Edge.get())->IsTailCall()) {
      FoundReturn = true;
    } else if (Edge->Type() == BrokenReturn) {
      FoundBrokenReturn = true;
      BrokenReturnCount++;
    } else {
      NoReturnCount++;
    }
  }

  // Function is elected to inline if there is one and only one broken return
  AttributesSet Attributes;
  if (FoundReturn) {
    // Do nothing
  } else if (FoundBrokenReturn && BrokenReturnCount == 1
             && NoReturnCount == 0) {
    Attributes.insert(model::FunctionAttribute::Inline);
  } else {
    Attributes.insert(model::FunctionAttribute::NoReturn);
  }

  revng_assert(CFG.size() > 0);
  for (efa::BasicBlock &Block : CFG)
    revng_assert(Block.Successors().size() > 0);

  return FunctionSummary(Attributes,
                         ClobberedRegisters.getClobberedRegisters(),
                         {},
                         std::move(CFG),
                         MaybeWinFSO);
}

FunctionSummary CFGAnalyzer::analyze(llvm::BasicBlock *Entry) {
  using namespace llvm;
  using llvm::BasicBlock;
  using namespace ABIAnalyses;

  BasicBlockID EntryID = getBasicBlockID(Entry);

  IRBuilder<> Builder(M.getContext());
  ABIAnalysesResults ABIResults;

  // Detect function boundaries
  OutlinedFunction OutlinedFunction = outline(Entry);

  // Recover the control-flow graph of the function
  SortedVector<efa::BasicBlock> CFG = collectDirectCFG(&OutlinedFunction);
  revng_assert(CFG.size() > 0);

  // The analysis aims at identifying the callee-saved registers of a
  // function and establishing if a function returns properly, i.e., it
  // jumps to the return address (regular function). In order to achieve
  // this, the IR is crafted by loading the program counter, the stack
  // pointer, as well as the ABI registers CSVs respectively at function
  // prologue / epilogue. When the subtraction between their entry and end
  // values is found to be zero (after running an LLVM optimization
  // pipeline), we may infer if the function returns correctly, the stack
  // is left unanaltered, etc. Hence, upon every original indirect jump
  // (candidate exit point), a marker of this kind is installed:
  //
  //                           jumps to RA,    SP,    rax,   rbx,   rbp
  // call i64 @indirect_branch_info(i128 0, i64 8, i64 %8, i64 0, i64 0)
  //
  // Here, subsequently the opt pipeline computation, we may tell that the
  // function jumps to its return address (thus, it is not a longjmp /
  // tail call), `rax` register has been clobbered by the callee, whereas
  // `rbx` and `rbp` are callee-saved registers.
  createIBIMarker(&OutlinedFunction);

  Function *F = OutlinedFunction.Function.get();

  // Prevent DCE by making branch conditions opaque
  opaqueBranchConditions(F, Builder);

  // Store the values that build up the program counter in order to have them
  // constant-folded away by the optimization pipeline.
  materializePCValues(F, Builder);

  // Execute the optimization pipeline over the outlined function
  runOptimizationPipeline(F);

  // Squeeze out the results obtained from the optimization passes
  auto FunctionInfo = milkInfo(&OutlinedFunction, std::move(CFG));

  return FunctionInfo;
}

CallSummarizer::CallSummarizer(llvm::Module *M,
                               Function *PreCallHook,
                               Function *PostCallHook,
                               llvm::Function *RetHook,
                               GlobalVariable *SPCSV) :
  M(M),
  PreCallHook(PreCallHook),
  PostCallHook(PostCallHook),
  RetHook(RetHook),
  SPCSV(SPCSV),
  RegistersClobberedPool(M, false) {
  RegistersClobberedPool.setMemoryEffects(MemoryEffects::readOnly());
  RegistersClobberedPool.addFnAttribute(llvm::Attribute::NoUnwind);
  RegistersClobberedPool.addFnAttribute(llvm::Attribute::WillReturn);
}

using CSVSet = std::set<llvm::GlobalVariable *>;

void CallSummarizer::handleCall(MetaAddress CallerBlock,
                                llvm::IRBuilder<> &Builder,
                                MetaAddress Callee,
                                const CSVSet &ClobberedRegisters,
                                const std::optional<int64_t> &MaybeFSO,
                                bool IsNoReturn,
                                bool IsTailCall,
                                llvm::Value *SymbolNamePointer) {
  using namespace llvm;
  LLVMContext &Context = getContext(M);

  // Mark end of basic block with a pre-hook call
  Value *IsTailCallValue = ConstantInt::getBool(Context, IsTailCall);

  SmallVector<Value *, 4> Args = { CallerBlock.toValue(M),
                                   Callee.toValue(M),
                                   SymbolNamePointer,
                                   IsTailCallValue };

  Builder.CreateCall(PreCallHook, Args);

  clobberCSVs(Builder, ClobberedRegisters);

  // Adjust back the stack pointer
  if (not IsTailCall) {
    if (MaybeFSO.has_value()) {
      auto *StackPointer = createLoad(Builder, SPCSV);
      Value *Offset = ConstantInt::get(StackPointer->getType(), *MaybeFSO);
      auto *AdjustedStackPointer = Builder.CreateAdd(StackPointer, Offset);
      Builder.CreateStore(AdjustedStackPointer, SPCSV);
    }
  }

  // Mark end of basic block with a post-hook call
  Builder.CreateCall(PostCallHook, Args);
}

void CallSummarizer::handlePostNoReturn(llvm::IRBuilder<> &Builder) {
  Builder.CreateCall(M->getFunction("abort"));
  Builder.CreateUnreachable();
}

void CallSummarizer::handleIndirectJump(llvm::IRBuilder<> &Builder,
                                        MetaAddress Block,
                                        llvm::Value *SymbolNamePointer) {
  Builder.CreateCall(PreCallHook,
                     { Block.toValue(M),
                       MetaAddress::invalid().toValue(M),
                       SymbolNamePointer,
                       Builder.getTrue() });

  Builder.CreateCall(RetHook, { Block.toValue(M) });
}

void CallSummarizer::clobberCSVs(llvm::IRBuilder<> &Builder,
                                 const CSVSet &ClobberedRegisters) {
  using namespace llvm;

  // Prevent the store instructions from being optimized out by storing
  // the an opaque value into clobbered registers
  for (GlobalVariable *Register : ClobberedRegisters) {
    auto *CSVTy = Register->getValueType();
    auto Name = ("registers_clobbered_" + Twine(Register->getName())).str();
    auto *ClobberFunction = RegistersClobberedPool.get(Register->getName(),
                                                       CSVTy,
                                                       {},
                                                       Name);
    Builder.CreateStore(Builder.CreateCall(ClobberFunction), Register);
  }
}

} // namespace efa
