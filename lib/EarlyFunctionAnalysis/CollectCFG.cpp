//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SCCIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/Model/Binary.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/YAMLTraits.h"

namespace {

struct CallGraphNodeData {
  CallGraphNodeData(MetaAddress Address) : Address(Address) {}
  MetaAddress Address;
};
using CallGraphNode = ForwardNode<CallGraphNodeData>;
using CallGraph = GenericGraph<CallGraphNode>;

/// \return the `AlwaysInline` model functions directly called by \p CFG
static std::set<MetaAddress>
alwaysInlineCallees(const efa::ControlFlowGraph &CFG,
                    const model::Binary &Binary) {
  std::set<MetaAddress> Result;

  for (const efa::BasicBlock &Block : CFG.Blocks()) {
    for (const auto &Edge : Block.Successors()) {
      auto *Call = llvm::dyn_cast<efa::CallEdge>(Edge.get());
      if (Call == nullptr)
        continue;

      MetaAddress Callee = Call->Destination().notInlinedAddress();
      if (Callee.isInvalid())
        continue;

      auto It = Binary.Functions().find(Callee);
      if (It == Binary.Functions().end())
        continue;

      if (It->Attributes().contains(model::FunctionAttribute::AlwaysInline))
        Result.insert(Callee);
    }
  }

  return Result;
}

/// \return the functions taking part in a cycle of the call graph rooted at
///         \p Entry
///
/// Inlining the members of a cycle does not terminate, so we pretend they are
/// not marked as `AlwaysInline` at all.
static std::set<MetaAddress> inCycle(CallGraphNode *Entry) {
  std::set<MetaAddress> Result;

  for (auto I = llvm::scc_begin(Entry), End = llvm::scc_end(Entry); I != End;
       ++I) {
    if (not I.hasCycle())
      continue;

    for (const CallGraphNode *Node : *I)
      Result.insert(Node->Address);
  }

  return Result;
}

} // namespace

namespace revng::pypeline::piperuns {

using FSO = efa::FunctionSummaryOracle;

CollectCFG::CollectCFG(const class Model &Model,
                       llvm::StringRef Config,
                       llvm::StringRef DynamicConfig,
                       LLVMRootContainer &Input,
                       CFGMap &Output) :
  Model(Model),
  Output(Output),
  Root(Input.getModule()),
  Globals(*Model.get().get(), Input.getModule()),
  GCBI(*Model.get().get(), Input.getModule()),
  Oracle(FSO::importBasicPrototypeData(Input.getModule(),
                                       Globals,
                                       *Model.get().get())),
  Analyzer(Input.getModule(), GCBI, Root, Globals, Model.get(), Oracle) {
}

void CollectCFG::runOnFunction(const model::Function &Function) {
  MetaAddress EntryAddress = Function.Entry();
  const model::Binary &Binary = *Model.get().get();

  // Recover the control-flow graph of the function
  TupleTree<efa::FunctionBundle> New;
  efa::ControlFlowGraph &Main = New->MainFunction();
  Main.Entry() = EntryAddress;
  Main.Blocks() = std::move(Analyzer.analyze(EntryAddress).CFG);

  if (DebugNames) {
    auto Function = Binary.Functions().at(EntryAddress);
    Main.Name() = Function.Name();
  }

  if (Main.Blocks().size() > 0)
    revng_assert(Main.Blocks().contains(BasicBlockID(Main.Entry())));

  // Run final steps on the CFG
  Main.simplify(Binary);

  if (Main.Blocks().size() > 0)
    revng_assert(Main.Blocks().contains(BasicBlockID(Main.Entry())));

  collectAlwaysInlineFunctions(*New);

  Output.getElement(ObjectID(EntryAddress)) = std::move(New);
}

efa::ControlFlowGraph CollectCFG::analyzeFunction(const MetaAddress &Entry) {
  const model::Binary &Binary = *Model.get().get();

  efa::ControlFlowGraph Result;
  Result.Entry() = Entry;
  Result.Blocks() = std::move(Analyzer.analyze(Entry).CFG);

  if (DebugNames)
    Result.Name() = Binary.Functions().at(Entry).Name();

  Result.simplify(Binary);

  return Result;
}

void CollectCFG::collectAlwaysInlineFunctions(efa::FunctionBundle &Bundle) {
  const model::Binary &Binary = *Model.get().get();
  const MetaAddress &EntryAddress = Bundle.MainFunction().Entry();

  // Explore the `AlwaysInline` call graph reachable from the function at hand,
  // recording the control-flow graph of each function we meet
  CallGraph Graph;
  std::map<MetaAddress, CallGraphNode *> Nodes;
  std::map<MetaAddress, efa::ControlFlowGraph> CFGs;

  Nodes[EntryAddress] = Graph.addNode(EntryAddress);

  OnceQueue<MetaAddress> ToAnalyze;
  ToAnalyze.insert(EntryAddress);

  while (not ToAnalyze.empty()) {
    MetaAddress Caller = ToAnalyze.pop();

    std::set<MetaAddress> Callees;
    if (Caller == EntryAddress) {
      Callees = alwaysInlineCallees(Bundle.MainFunction(), Binary);
    } else {
      efa::ControlFlowGraph CFG = analyzeFunction(Caller);
      Callees = alwaysInlineCallees(CFG, Binary);
      CFGs.emplace(Caller, std::move(CFG));
    }

    for (const MetaAddress &Callee : Callees) {
      CallGraphNode *&Node = Nodes[Callee];
      if (Node == nullptr)
        Node = Graph.addNode(Callee);

      Nodes.at(Caller)->addSuccessor(Node);
      ToAnalyze.insert(Callee);
    }
  }

  // A function taking part in a cycle cannot be inlined, and neither can the
  // functions only reachable through it
  std::set<MetaAddress> Cyclic = inCycle(Nodes.at(EntryAddress));

  OnceQueue<MetaAddress> ToInline;
  ToInline.insert(EntryAddress);

  while (not ToInline.empty()) {
    MetaAddress Address = ToInline.pop();

    if (Address != EntryAddress)
      Bundle.AlwaysInlineFunctions().insert(std::move(CFGs.at(Address)));

    for (const CallGraphNode *Callee : Nodes.at(Address)->successors())
      if (not Cyclic.contains(Callee->Address))
        ToInline.insert(Callee->Address);
  }
}

} // namespace revng::pypeline::piperuns
