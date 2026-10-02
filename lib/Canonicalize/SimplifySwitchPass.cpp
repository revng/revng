//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Without this Pass the devirtualization process effectively handles indirect
// jumps, particularly those arising from C's switch statement. However, the
// output is too low-level, especially when switch statements involve jump
// tables. This results in emitted memory accesses instead of clear switch
// statements based on variable values. For instance, rather than a readable
// switch statement reflecting _argument_rdi values, the code directly accesses
// the jump table's entries. To address this readability issue, we propose
// introducing the SimplifySwtichPass. This pass should refine the output of
// switch statements by re-running the devirtualization process and adjusting
// them to a more human-readable format. This enhancement is crucial for
// improving code comprehension.

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

#include "revng/Canonicalize/SimplifySwitch.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/ValueMaterializer/DataFlowRangeAnalysis.h"
#include "revng/ValueMaterializer/ValueMaterializer.h"

using namespace llvm;

static Logger Log("switch-opt");

class RawBinaryMemoryOracle final : public MemoryOracle {
private:
  RawBinaryView &BinaryView;
  model::Architecture::Values Architecture = model::Architecture::Invalid;

public:
  RawBinaryMemoryOracle(RawBinaryView &BinaryView,
                        model::Architecture::Values Architecture) :
    BinaryView(BinaryView), Architecture(Architecture) {}

  MaterializedValue load(uint64_t LoadAddress, unsigned LoadSize) final {
    auto Address = MetaAddress::fromGeneric(Architecture, LoadAddress);
    return BinaryView.load(Address,
                           LoadSize,
                           model::Architecture::isLittleEndian(Architecture));
  }
};

static BasicBlock *getBlockFor(SwitchInst *Switch, Constant *C) {
  for (const auto &I : Switch->cases()) {
    const ConstantInt *CaseValue = I.getCaseValue();
    if (CaseValue == C)
      return I.getCaseSuccessor();
  }

  return nullptr;
}

static BidirectionalNode<DataFlowNode> *
findStartNode(const DataFlowGraph &DFG) {
  bool BailOut = false;
  BidirectionalNode<DataFlowNode> *Result = nullptr;
  for (const BidirectionalNode<DataFlowNode> *BidirectNode : DFG.nodes()) {
    const DataFlowNode &Node = *BidirectNode;
    // Ignore nodes that are constants in the DFG.
    if (isa<ConstantInt>(Node.Value))
      continue;

    if (Node.UseOracle) {
      if (Result != nullptr) {
        BailOut = true;
        break;
      } else {
        // TODO: Fix this const_cast.
        BidirectionalNode<DataFlowNode>
          *N = const_cast<BidirectionalNode<DataFlowNode> *>(BidirectNode);
        Result = N;
      }
    }
  }

  // We found more than one start node.
  if (Result == nullptr or BailOut)
    return nullptr;

  return Result;
}

static bool handleSwitch(SwitchInst *Switch,
                         LazyValueInfo &LVI,
                         DataFlowRangeAnalysis &DFRA,
                         DominatorTree &DT,
                         RawBinaryMemoryOracle &MO) {
  // Skip empty switches.
  if (Switch->getNumCases() == 0)
    return false;

  Value *Condition = Switch->getCondition();
  DataFlowGraph::Limits Limits(1000 /*MaxPhiLike*/, 1 /*MaxLoad*/);

  auto AVIOracle = Oracle::AdvancedValueInfo;
  ::ValueMaterializer Results = ::ValueMaterializer::getValuesFor(Switch,
                                                                  Condition,
                                                                  MO,
                                                                  LVI,
                                                                  DFRA,
                                                                  DT,
                                                                  Limits,
                                                                  AVIOracle);
  // We can not do too much without any values materialized.
  if (not Results.values())
    return false;

  // The number of values materialized should match the number of cases in
  // the switch.
  if (Results.values()->size() != Switch->getNumCases()) {
    revng_log(Log, "Unexpected number of values found!");
    return false;
  }

  const DataFlowGraph &DFG = Results.dataFlowGraph();
  // Identify the only node in the data flow graph that's associated to an
  // Oracle and use it as a switch condition.
  BidirectionalNode<DataFlowNode> *StartNode = findStartNode(DFG);
  if (!StartNode) {
    revng_log(Log, "Have not found a proper start node!");
    return false;
  }

  std::map<ConstantInt *, BasicBlock *> NewLabels;
  for (const llvm::APInt &Value : *StartNode->OracleRange) {
    std::optional<MaterializedValue> OldValue = DFG.materializeOne(StartNode,
                                                                   MO,
                                                                   Value);
    if (not OldValue) {
      revng_log(Log,
                "Have not found materialized value for "
                  << Value.getZExtValue());
      return false;
    }
    Constant *ConstantForOld = toLLVMConstant(Switch->getContext(),
                                              (*OldValue).value());
    Constant *ConstantForTheValue = toLLVMConstant(Switch->getContext(), Value);

    BasicBlock *BlockForValue = getBlockFor(Switch, ConstantForOld);

    if (not BlockForValue) {
      revng_log(Log,
                "Have not found case/BB for "
                  << ConstantForOld->getUniqueInteger().getZExtValue());
      return false;
    }
    NewLabels[cast<ConstantInt>(ConstantForTheValue)] = BlockForValue;
  }

  // We have not found all the cases for the new switch.
  if (Switch->getNumCases() != NewLabels.size())
    return false;

  // Recreate the new switch.
  auto *NewSwitch = SwitchInst::Create(StartNode->Value,
                                       Switch->getDefaultDest(),
                                       NewLabels.size(),
                                       Switch);

  for (auto &NewLabel : NewLabels)
    NewSwitch->addCase(NewLabel.first, NewLabel.second);

  Switch->replaceAllUsesWith(NewSwitch);

  return true;
}

namespace revng::pypeline::piperuns {

SimplifySwitch::SimplifySwitch(const class Model &Model,
                               llvm::StringRef Config,
                               llvm::StringRef DynamicConfig,
                               const BinariesContainer &BinariesContainer,
                               LLVMFunctionContainer &ModuleContainer) :
  ModuleContainer(ModuleContainer),
  Binary(*Model.get().get()),
  BinaryView(makeBinaryView(Model, BinariesContainer)) {
}

void SimplifySwitch::runOnFunction(const model::Function &ModelFunction) {
  MetaAddress Entry = ModelFunction.Entry();
  llvm::Module &M = ModuleContainer.getModule(ObjectID(Entry));
  llvm::Function &Function = getUniqueIsolatedFunction(M, Entry);

  // LazyValueInfo is usually handed out by an analysis pass, but the things it
  // needs are few and cheap enough to build here, which saves us from running
  // a pass manager just for this.
  TargetLibraryInfoImpl TLIImpl(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLIImpl, &Function);
  AssumptionCache AC(Function);
  LazyValueInfo LVI(&AC, &M.getDataLayout(), &TLI);

  DominatorTree DT(Function);
  DataFlowRangeAnalysis DFRA(M);
  RawBinaryMemoryOracle MO(BinaryView, Binary.Architecture());

  for (BasicBlock &BB : Function) {
    for (Instruction &I : make_early_inc_range(BB)) {
      SwitchInst *Switch = dyn_cast<SwitchInst>(&I);
      if (not Switch)
        continue;

      if (handleSwitch(Switch, LVI, DFRA, DT, MO))
        Switch->eraseFromParent();
    }
  }
}

} // namespace revng::pypeline::piperuns
