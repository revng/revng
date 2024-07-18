//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/ZipMapIterator.h"
#include "revng/EarlyFunctionAnalysis/AnalyzeRegisterUsage.h"
#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Register.h"
#include "revng/RegisterUsageAnalyses/Liveness.h"
#include "revng/RegisterUsageAnalyses/ReachingDefinitions.h"
#include "revng/Support/Assert.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

static Logger<> Log("rua-analyses");

namespace efa {

template void RUAResults::dump<Logger<true>>(Logger<true> &,
                                             const char *) const;

struct CallSite {
  using Node = rua::Function::Node;

  Node *Block = nullptr;
  MetaAddress Callee;

  void setCallee(MetaAddress Callee) {
    if (this->Callee.isValid()) {
      revng_assert(this->Callee == Callee);
    } else {
      this->Callee = Callee;
    }
  }
};

struct FunctionToAnalyze {
  using Node = rua::Function::Node;

  rua::Function Function;

  std::map<BasicBlockID, CallSite> CallSites;

  /// An artificial return node sinking all the return instructions
  Node *ReturnNode = nullptr;

  /// An artificial node guaranteed to be reachable from all nodes
  Node *SinkNode = nullptr;
};

static model::Register::Values
getRegister(Value *Pointer, model::Architecture::Values Architecture) {
  if (auto *CSV = dyn_cast<GlobalVariable>(Pointer)) {
    return model::Register::fromCSVName(CSV->getName(), Architecture);
  }
  return model::Register::Invalid;
}

static rua::Function::Node *findUnrechableNode(rua::Function::Node *Start) {
  df_iterator_default_set<rua::Function::Node *> Visited;
  for (auto &_ : llvm::inverse_depth_first_ext(Start, Visited))
    ;

  for (auto *Node : Start->getParent()->nodes()) {
    if (not Visited.contains(Node))
      return Node;
  }

  return nullptr;
}

static rua::OperationType::Values storeType(Value *V) {
  Function *Callee = nullptr;
  if (auto *Call = dyn_cast<CallInst>(V))
    Callee = getCalledFunction(Call);

  if (Callee != nullptr and FunctionTags::ClobbererFunction.isTagOf(Callee)) {
    return rua::OperationType::Clobber;
  } else {
    return rua::OperationType::Write;
  }
}

static FunctionToAnalyze
fromLLVMFunction(llvm::Function &F,
                 model::Architecture::Values Architecture,
                 Function *PreCallSiteHook,
                 Function *PostCallSiteHook,
                 Function *RetHook) {
  using namespace model;

  auto StackPointer = Architecture::getStackPointer(Architecture);

  SmallPtrSet<rua::Function::Node *, 16> Preserve;

  FunctionToAnalyze Result;
  rua::Function &Function = Result.Function;
  DenseMap<llvm::BasicBlock *, rua::BlockNode *> BlocksMap;

  Result.ReturnNode = Function.addNode();
  Result.ReturnNode->Label = "ReturnNode";
  Preserve.insert(Result.ReturnNode);

  Result.SinkNode = Function.addNode();
  Result.SinkNode->Label = "SinkNode";
  Result.ReturnNode->addSuccessor(Result.SinkNode);
  Preserve.insert(Result.SinkNode);

  uint8_t RegistersCount = 0;

  // Translate the function
  // Note: we use depth_first to avoid unreachable code
  for (llvm::BasicBlock *BB : llvm::depth_first(&F)) {
    // Create the node for the basic block
    auto *NewNode = Function.addNode();
    NewNode->Label = BB->getName();
    BlocksMap[BB] = NewNode;

    auto &Operations = NewNode->Operations;
    auto CreateOperation = [&Function,
                            &Operations,
                            &Architecture,
                            StackPointer](rua::OperationType::Values Type,
                                          Value *Pointer) {
      auto Register = getRegister(Pointer, Architecture);
      if (Register != model::Register::Invalid and Register != StackPointer) {
        Operations.emplace_back(Type, Function.registerIndex(Register));
      }
    };

    // Translate the basic block
    for (llvm::Instruction &I : *BB) {
      auto Call = dyn_cast<CallInst>(&I);

      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        // If reading a register, record it
        CreateOperation(rua::OperationType::Read, Load->getPointerOperand());
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        // If writing a register, record it
        CreateOperation(storeType(Store->getValueOperand()),
                        Store->getPointerOperand());
      } else if (Call != nullptr
                 and (isCallTo(Call, PreCallSiteHook)
                      or isCallTo(Call, PostCallSiteHook)
                      or isCallTo(Call, RetHook))) {
        // Handle markers

        // Get the pc
        BasicBlockID PC = BasicBlockID::fromValue(Call->getArgOperand(0));
        revng_assert(PC.isValid());

        if (isCallTo(Call, PreCallSiteHook)) {
          // We found a call to pre_hook. This a llvm::BasicBlock starting with
          // pre_hook, ending with post_hook and containing register clobbering
          auto &CallSite = Result.CallSites[PC];

          // Ensure it's the first instruction in the basic block
          revng_assert(&*BB->begin() == &I);
          revng_assert(isCallTo(&*BB->getTerminator()->getPrevNode(),
                                PostCallSiteHook));

          // Record the current node as before a call
          CallSite.Block = NewNode;
          Preserve.insert(NewNode);

          // Record the callee
          CallSite.setCallee(MetaAddress::fromValue(Call->getArgOperand(1)));

        } else if (isCallTo(Call, PostCallSiteHook)) {
          // Perform some safety checks
          revng_assert(&*BB->getTerminator()->getPrevNode() == &I);
          revng_assert(Result.CallSites.contains(PC));
          revng_assert(Result.CallSites.at(PC).Block == NewNode);
        } else if (isCallTo(Call, RetHook)) {
          // This is a proper return, link to the exit node
          NewNode->addSuccessor(Result.ReturnNode);
        }
      }
    }
  }

  // Set the entry node
  rua::Function::Node *Entry = BlocksMap[&F.getEntryBlock()];
  Function.setEntryNode(Entry);
  // Note: we do not add to Preserve the entry block

  // Create edges
  for (auto &[LLVMBlock, Node] : BlocksMap) {
    for (llvm::BasicBlock *Successor : successors(LLVMBlock)) {
      revng_assert(BlocksMap.count(Successor) != 0);
      Node->addSuccessor(BlocksMap[Successor]);
    }
  }

  // Ensure every node reaches the sink
  using Node = rua::Function::Node;
  Node *MaybeUnrechableNode = findUnrechableNode(Result.SinkNode);
  while (MaybeUnrechableNode != nullptr) {
    Result.SinkNode->addPredecessor(MaybeUnrechableNode);
    MaybeUnrechableNode = findUnrechableNode(Result.SinkNode);
  }

  // Perform some semplifications on the IR
  Result.Function.simplify(Preserve);

  return Result;
}

// Run the ABI analyses on the outlined function F. This function must have all
// the original function calls replaced with a basic block starting with a call
// to `precall_hook` followed by a summary of the side effects of the function
// followed by a call to `postcall_hook` and a basic block terminating
// instruction.
RUAResults analyzeRegisterUsage(Function *F,
                                const GeneratedCodeBasicInfo &GCBI,
                                model::Architecture::Values Architecture,
                                Function *PreCallSiteHook,
                                Function *PostCallSiteHook,
                                Function *RetHook) {
  RUAResults FinalResults;

  // TODO: can we avoid recreating this each time?
  revng_log(Log, "Building graph for " << F->getName());
  auto Function = fromLLVMFunction(*F,
                                   Architecture,
                                   PreCallSiteHook,
                                   PostCallSiteHook,
                                   RetHook);

  if (Log.isEnabled()) {
    Function.Function.dump(Log);
    Log << DoLog;
  }

  auto GetRegisterName = model::Register::getRegisterName;

  auto *M = F->getParent();
  auto GetCSV = [&M](model::Register::Values Register) {
    auto *Result = M->getGlobalVariable(model::Register::getCSVName(Register),
                                        true);
    revng_assert(Result != nullptr);
    return Result;
  };

  {
    // Run the liveness analysis
    revng_log(Log, "Running Liveness");
    rua::Liveness Liveness(Function.Function);
    auto AnalysisResult = MFP::getMaximalFixedPoint(Liveness,
                                                    &Function.Function,
                                                    Liveness.defaultValue(),
                                                    Liveness.defaultValue(),
                                                    { Function.ReturnNode });

    // Collect registers alive at the entry
    revng_log(Log, "Registers alive at the entry of the function:");
    rua::BlockNode *EntryNode = Function.Function.getEntryNode();
    const BitVector &EntryResult = AnalysisResult[EntryNode].OutValue;
    for (auto Register : Function.Function.registersInSet(EntryResult)) {
      // This register is alive at the entry of the function

      revng_log(Log, "  " << GetRegisterName(Register));
      FinalResults.ArgumentsRegisters.insert(GetCSV(Register));
    }

    for (const auto &[PC, CallSite] : Function.CallSites) {
      auto &ResultsCallSite = FinalResults.CallSites[PC];
      ResultsCallSite.CalleeAddress = CallSite.Callee;

      auto *PostNode = CallSite.Block;
      revng_log(Log,
                "Registers alive after the call to "
                  << CallSite.Callee.toString() << " at " << PC.toString()
                  << " (block " << PostNode->label() << ")");
      const BitVector &CallSiteResult = AnalysisResult.at(PostNode).InValue;
      for (auto Register : Function.Function.registersInSet(CallSiteResult)) {
        // This register is alive after the call site

        revng_log(Log, "  " << GetRegisterName(Register));
        ResultsCallSite.ReturnValuesRegisters.insert(GetCSV(Register));
      }
    }
  }

  {
    // Run the reaching definitions analysis
    revng_log(Log, "Running ReachingDefinitions");
    rua::ReachingDefinitions ReachingDefinitions(Function.Function);
    auto DefaultValue = ReachingDefinitions.defaultValue();
    auto *EntryNode = Function.Function.getEntryNode();
    auto AnalysisResult = MFP::getMaximalFixedPoint(ReachingDefinitions,
                                                    &Function.Function,
                                                    DefaultValue,
                                                    DefaultValue,
                                                    { EntryNode });

    auto Compute = [&AnalysisResult, &Function](rua::Function::Node *Node,
                                                bool Before) {
      const auto &SinkResults = AnalysisResult.at(Function.SinkNode).OutValue;

      const auto *NodeResults = &AnalysisResult.at(Node).OutValue;
      if (Before)
        NodeResults = &AnalysisResult.at(Node).InValue;

      return rua::ReachingDefinitions::compute(*NodeResults, SinkResults);
    };

    revng_log(Log,
              "Registers with at least one write that reaches the exit node of "
              "the function without ever being read:");
    rua::BlockNode *ExitNode = Function.ReturnNode;
    const BitVector &ExitResult = Compute(ExitNode, false);
    for (auto Register : Function.Function.registersInSet(ExitResult)) {
      // This register has at least one write that reaches the exit node of the
      // function without ever being read
      revng_log(Log, "  " << GetRegisterName(Register));

      FinalResults.ReturnValuesRegisters.insert(GetCSV(Register));
    }

    for (const auto &[PC, CallSite] : Function.CallSites) {
      auto &ResultsCallSite = FinalResults.CallSites[PC];
      ResultsCallSite.CalleeAddress = CallSite.Callee;
      revng_log(Log,
                "Registers with at least one write that reaches the call to "
                  << CallSite.Callee.toString() << " at " << PC.toString()
                  << " without ever being read:");

      auto *PreNode = CallSite.Block;
      const BitVector &CallSiteResult = Compute(PreNode, true);
      for (auto Register : Function.Function.registersInSet(CallSiteResult)) {
        // This register has at least one write that reaches the call site
        // without ever being read

        revng_log(Log, "  " << GetRegisterName(Register));

        ResultsCallSite.ArgumentsRegisters.insert(GetCSV(Register));
      }
    }
  }

  return FinalResults;
}

template<typename T>
void RUAResults::dump(T &Output, const char *Prefix) const {
  Output << Prefix << "Arguments:\n";
  for (auto *CSV : ArgumentsRegisters) {
    Output << Prefix << "  " << CSV->getName().str() << '\n';
  }

  Output << Prefix << "Call site:\n";
  for (auto &[PC, StateMap] : CallSites) {
    Output << Prefix << "  Call in basic block " << PC.toString() << '\n';
    Output << Prefix << "  "
           << "  "
           << "Arguments:\n";
    for (auto *CSV : StateMap.ArgumentsRegisters) {
      Output << Prefix << "      " << CSV->getName().str() << '\n';
    }
    Output << Prefix << "  "
           << "  "
           << "Return values:\n";
    for (auto *CSV : StateMap.ReturnValuesRegisters) {
      Output << Prefix << "      " << CSV->getName().str() << '\n';
    }
  }

  Output << Prefix << "Return values:\n";
  for (auto *CSV : ReturnValuesRegisters) {
    Output << Prefix << "  " << CSV->getName().str() << '\n';
  }
}

} // namespace efa
