/// \file FunctionMetadata.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

namespace efa {

struct FunctionCFGNodeData {
  FunctionCFGNodeData(MetaAddress Start) : Start(Start) {}
  MetaAddress Start;
};

using FunctionCFGNode = ForwardNode<FunctionCFGNodeData>;
using FunctionCFG = GenericGraph<FunctionCFGNode, 16, true>;

/// A helper data structure to simplify working with the graph
/// when verifying the CFG.
struct FunctionCFGVerificationHelper {
public:
  FunctionCFG Graph;
  std::map<MetaAddress, FunctionCFGNode *> Map;

public:
  FunctionCFGVerificationHelper(const efa::FunctionMetadata &Metadata,
                                const model::Binary &Binary) {
    using G = FunctionCFG;
    std::tie(Graph, Map) = buildControlFlowGraph<G>(Metadata.ControlFlowGraph,
                                                    Metadata.Entry,
                                                    Binary);
  }

public:
  bool allNodesAreReachable() const {
    revng_assert(Graph.size() == Map.size());
    if (Map.size() == 0)
      return true;

    // Ensure all the nodes are reachable from the entry node
    df_iterator_default_set<FunctionCFGNode *> Visited;
    revng_assert(Graph.getEntryNode() != nullptr);
    for (auto &Ignore : depth_first_ext(Graph.getEntryNode(), Visited))
      ;
    return Visited.size() == Graph.size();
  }

  bool hasAtMostOneInvalidExit() const {
    FunctionCFGNode *Exit = nullptr;
    for (const auto &[Address, Node] : Map) {
      if (Address.isInvalid()) {
        if (Node->hasSuccessors() || Exit != nullptr)
          return false;
        Exit = Node;
      } else {
        if (!Node->hasSuccessors())
          return false;
      }
    }

    return true;
  }
};

const efa::BasicBlock *FunctionMetadata::findBlock(GeneratedCodeBasicInfo &GCBI,
                                                   llvm::BasicBlock *BB) const {
  llvm::BasicBlock *JumpTargetBB = GCBI.getJumpTargetBlock(BB);
  if (JumpTargetBB == nullptr)
    return nullptr;

  MetaAddress CallerBlockAddress = GCBI.getPCFromNewPC(JumpTargetBB);
  revng_assert(CallerBlockAddress.isValid());
  auto It = ControlFlowGraph.find(CallerBlockAddress);

  while (It == ControlFlowGraph.end()) {

    llvm::BasicBlock *PredecessorJumpTargetBB = nullptr;
    for (llvm::BasicBlock *Predecessor : predecessors(JumpTargetBB)) {
      if (GCBI.isTranslated(Predecessor)) {
        auto *NewJT = GCBI.getJumpTargetBlock(Predecessor);
        if (PredecessorJumpTargetBB != nullptr) {
          revng_assert(PredecessorJumpTargetBB == NewJT,
                       "Jump target is not in the CFG but it has mutiple "
                       "predecessors");
        }
        PredecessorJumpTargetBB = NewJT;
      }
    }
    JumpTargetBB = PredecessorJumpTargetBB;

    revng_assert(JumpTargetBB != nullptr);
    CallerBlockAddress = GCBI.getPCFromNewPC(JumpTargetBB);
    revng_assert(CallerBlockAddress.isValid());
    It = ControlFlowGraph.find(CallerBlockAddress);
  }

  return &*It;
}

void FunctionMetadata::simplify(const model::Binary &Binary) {
  // If A does not end with a call and A.end == B.start and A is the only
  // predecessor of B and B is the only successor of A, merge

  // Create quick map of predecessors
  std::map<MetaAddress, SmallVector<MetaAddress, 2>> Predecessors;
  for (efa::BasicBlock &Block : ControlFlowGraph) {
    for (auto &Successor : Block.Successors) {
      if (Successor->Type == efa::FunctionEdgeType::DirectBranch
          and Successor->Destination.isValid()) {
        Predecessors[Successor->Destination].push_back(Block.Start);
      } else if (auto *Call = dyn_cast<efa::CallEdge>(Successor.get())) {
        if (not Call->IsTailCall
            and not Call->hasAttribute(Binary,
                                       model::FunctionAttribute::NoReturn)) {
          Predecessors[Block.End].push_back(Block.Start);
        }
      }
    }
  }

  // Identify blocks that need to be merged in their predecessor
  SmallVector<std::pair<MetaAddress, MetaAddress>, 4> ToMerge;
  for (efa::BasicBlock &Block : ControlFlowGraph) {
    // Ignore entry block entirely
    if (Block.End == Entry)
      continue;

    // Do we have only one successor?
    if (Block.Successors.size() != 1)
      continue;

    // Is the successor a direct branch to the end of the block?
    auto &OnlySuccessor = *Block.Successors.begin();
    if (not(OnlySuccessor->Type == efa::FunctionEdgeType::DirectBranch
            and OnlySuccessor->Destination == Block.End))
      continue;

    // Does the only successor has only one predeccessor?
    auto PredecessorsAddress = Predecessors.at(Block.End);
    if (PredecessorsAddress.size() != 1)
      continue;

    // Are we the only predecessor?
    if (*PredecessorsAddress.begin() != Block.Start)
      continue;

    ToMerge.emplace_back(Block.Start, Block.End);
  }

  for (auto [PredecessorAddress, BlockAddress] : llvm::reverse(ToMerge)) {
    efa::BasicBlock &Predecessor = ControlFlowGraph.at(PredecessorAddress);
    efa::BasicBlock &Block = ControlFlowGraph.at(BlockAddress);

    // Safety checks
    revng_assert(Predecessor.Successors.size() == 1);
    revng_assert(Predecessor.End == Block.Start);

    // Merge Block into Predecessor
    Predecessor.End = Block.End;
    Predecessor.Successors = std::move(Block.Successors);

    // Drop Block
    ControlFlowGraph.erase(BlockAddress);
  }
}

bool FunctionMetadata::verify(const model::Binary &Binary) const {
  return verify(Binary, false);
}

bool FunctionMetadata::verify(const model::Binary &Binary, bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(Binary, VH);
}

bool FunctionMetadata::verify(const model::Binary &Binary,
                              model::VerifyHelper &VH) const {
  const auto &Function = Binary.Functions.at(Entry);

  if (ControlFlowGraph.size() == 0)
    return VH.fail("The function has no CFG");

  // Populate graph
  FunctionCFGVerificationHelper Helper(*this, Binary);

  // Ensure all the nodes are reachable from the entry node
  if (not Helper.allNodesAreReachable())
    return VH.fail();

  // Ensure the only node with no successors is invalid
  if (not Helper.hasAtMostOneInvalidExit())
    return VH.fail();

  // Verify blocks
  if (ControlFlowGraph.size() > 0) {
    bool HasEntry = false;
    for (const BasicBlock &Block : ControlFlowGraph) {

      if (Block.Start == Entry) {
        if (HasEntry)
          return VH.fail("Multiple entry point blocks found");
        HasEntry = true;
      }

      if (Block.Successors.size() == 0)
        return VH.fail("A block has no successors", Block);

      for (const auto &Edge : Block.Successors)
        if (not Edge->verify(VH))
          return VH.fail();
    }

    if (not HasEntry) {
      return VH.fail("The function CFG does not contain a block starting at "
                     "the entry point",
                     *this);
    }
  }

  // Check function calls
  for (const auto &Block : ControlFlowGraph) {
    for (const auto &Edge : Block.Successors) {
      if (Edge->Type == efa::FunctionEdgeType::FunctionCall) {
        // We're in a direct call, get the callee
        const auto *Call = dyn_cast<CallEdge>(Edge.get());

        if (not Call->DynamicFunction.empty()) {
          // It's a dynamic call
          auto It = Binary.ImportedDynamicFunctions.find(Call->DynamicFunction);

          // If missing, fail
          if (It == Binary.ImportedDynamicFunctions.end())
            return VH.fail("Can't find callee \"" + Call->DynamicFunction
                           + "\"");
        } else if (Call->isDirect()) {
          // Regular call
          auto It = Binary.Functions.find(Call->Destination);

          // If missing, fail
          if (It == Binary.Functions.end())
            return VH.fail("Can't find callee");
        }
      }
    }
  }

  return true;
}

void FunctionMetadata::dump() const {
  serialize(dbg, *this);
}

void FunctionMetadata::dumpCFG(const model::Binary &Binary) const {
  auto [Graph, _] = buildControlFlowGraph<FunctionCFG>(ControlFlowGraph,
                                                       Entry,
                                                       Binary);
  raw_os_ostream Stream(dbg);
  WriteGraph(Stream, &Graph);
}

bool FunctionEdgeBase::verify() const {
  return verify(false);
}

bool FunctionEdgeBase::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool FunctionEdgeBase::verify(model::VerifyHelper &VH) const {
  using namespace efa::FunctionEdgeType;

  switch (Type) {
  case Invalid:
  case Count:
    return VH.fail();

  case DirectBranch:
    if (Destination.isInvalid())
      return VH.fail();
    break;
  case FunctionCall: {
    const auto &Call = cast<const CallEdge>(*this);
    if (Destination.isValid() and not Call.DynamicFunction.empty())
      return VH.fail("Dynamic function has destination address");
  } break;

  case Return:
  case BrokenReturn:
  case LongJmp:
  case Killer:
  case Unreachable:
    if (Destination.isValid())
      return VH.fail();
    break;
  }

  return true;
}

void FunctionEdgeBase::dump() const {
  serialize(dbg, *this);
}

void CallEdge::dump() const {
  serialize(dbg, *this);
}

model::Identifier BasicBlock::name() const {
  using llvm::Twine;
  return model::Identifier(std::string("bb_") + Start.toString());
}

void BasicBlock::dump() const {
  serialize(dbg, *this);
}

bool BasicBlock::verify() const {
  return verify(false);
}

bool BasicBlock::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool BasicBlock::verify(model::VerifyHelper &VH) const {
  if (Start.isInvalid() or End.isInvalid())
    return VH.fail();

  for (auto &Edge : Successors)
    if (not Edge->verify(VH))
      return VH.fail();

  return true;
}

} // namespace efa

template<>
struct llvm::DOTGraphTraits<efa::FunctionCFG *> : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool Simple = false) : DefaultDOTGraphTraits(Simple) {}

  static std::string
  getNodeLabel(const efa::FunctionCFGNode *Node, const efa::FunctionCFG *) {
    return Node->Start.toString();
  }

  static std::string getNodeAttributes(const efa::FunctionCFGNode *Node,
                                       const efa::FunctionCFG *Graph) {
    if (Node->Start == Graph->getEntryNode()->Start) {
      return "shape=box,peripheries=2";
    }

    return "";
  }
};
