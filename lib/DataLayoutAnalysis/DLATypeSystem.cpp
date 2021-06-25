//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <string>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DebugHelper.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAHelpers.h"

using namespace llvm;

using NodeAllocatorT = SpecificBumpPtrAllocator<dla::LayoutTypeSystemNode>;

void *operator new(size_t, NodeAllocatorT &NodeAllocator) {
  return NodeAllocator.Allocate();
}

namespace dla {

void OffsetExpression::print(llvm::raw_ostream &OS) const {
  OS << "Off: " << Offset;
  auto NStrides = Strides.size();
  revng_assert(NStrides == TripCounts.size());
  if (not Strides.empty()) {
    for (decltype(NStrides) N = 0; N < NStrides; ++N) {
      OS << ", {" << Strides[N] << ',';
      if (TripCounts[N].has_value())
        OS << TripCounts[N].value();
      else
        OS << "none";
      OS << '}';
    }
  }
}

void LayoutTypePtr::print(raw_ostream &Out) const {
  Out << '{';
  Out << "0x";
  Out.write_hex(reinterpret_cast<const unsigned long long>(V));
  Out << " [";
  if (isa<Function>(V)) {
    Out << "fname: " << V->getName();
  } else {
    if (auto *I = dyn_cast<Instruction>(V))
      Out << "In Func: " << I->getFunction()->getName() << " Instr: ";
    else if (auto *A = dyn_cast<Argument>(V))
      Out << "In Func: " << A->getParent()->getName() << " Arg: ";

    Out.write_escaped(getName(V));
  }
  Out << "], 0x";
  Out.write_hex(FieldIdx);
  Out << '}';
}

void LayoutTypeSystemNode::print(llvm::raw_ostream &OS) const {
  OS << "LTSN ID: " << ID;
}

namespace {

static constexpr size_t str_len(const char *S) {
  return S ? (*S ? (1 + str_len(S + 1)) : 0UL) : 0UL;
}

// We use \l here instead of \n, because graphviz has this sick way of saying
// that the text in the node labels should be left-justified
static constexpr const char DoRet[] = "\\l";
static constexpr const char NoRet[] = "";
static_assert(sizeof(DoRet) == (str_len(DoRet) + 1));
static_assert(sizeof(NoRet) == (str_len(NoRet) + 1));

static constexpr const char Equal[] = "Equal";
static constexpr const char Inherits[] = "Inherits from";
static constexpr const char Instance[] = "Has Instance of: ";
static constexpr const char Unexpected[] = "Unexpected!";
static_assert(sizeof(Equal) == (str_len(Equal) + 1));
static_assert(sizeof(Inherits) == (str_len(Inherits) + 1));
static_assert(sizeof(Instance) == (str_len(Instance) + 1));
static_assert(sizeof(Unexpected) == (str_len(Unexpected) + 1));
} // end unnamed namespace

void LayoutTypeSystem::dumpDotOnFile(const char *FName) const {
  std::error_code EC;
  raw_fd_ostream DotFile(FName, EC);
  revng_check(not EC, "Could not open file for printing LayoutTypeSystem dot");

  DotFile << "digraph LayoutTypeSystem {\n";
  DotFile << "  // List of nodes\n";

  for (const LayoutTypeSystemNode *L : getLayoutsRange()) {

    DotFile << "  node_" << L->ID << " [shape=rect,label=\"NODE ID: " << L->ID
            << " Size: " << L->Size << " InterferingChild: ";

    llvm::SmallVector<const llvm::Use *, 8> PtrUses;
    switch (L->InterferingInfo) {
    case Unknown:
      DotFile << 'U';
      break;
    case AllChildrenAreInterfering:
      DotFile << 'A';
      break;
    case AllChildrenAreNonInterfering:
      DotFile << 'N';
      break;
    default:
      revng_unreachable();
    }

    DebugPrinter->printNodeContent(*this, L, DotFile);
    DotFile << "\"];\n";
  }

  DotFile << "  // List of edges\n";

  for (LayoutTypeSystemNode *L : getLayoutsRange()) {

    uint64_t SrcNodeId = L->ID;

    for (const auto &PredP : L->Predecessors) {
      const TypeLinkTag *PredTag = PredP.second;
      const auto SameLink = [&](auto &OtherPair) {
        return SrcNodeId == OtherPair.first->ID and PredTag == OtherPair.second;
      };
      revng_assert(std::any_of(PredP.first->Successors.begin(),
                               PredP.first->Successors.end(),
                               SameLink));
    }

    std::string Extra;
    for (const auto &SuccP : L->Successors) {
      const TypeLinkTag *EdgeTag = SuccP.second;
      const auto SameLink = [&](auto &OtherPair) {
        return SrcNodeId == OtherPair.first->ID and EdgeTag == OtherPair.second;
      };
      revng_assert(std::any_of(SuccP.first->Predecessors.begin(),
                               SuccP.first->Predecessors.end(),
                               SameLink));
      const auto *TgtNode = SuccP.first;
      const char *EdgeLabel = nullptr;
      size_t LabelSize = 0;
      Extra.clear();
      switch (EdgeTag->getKind()) {
      case TypeLinkTag::LK_Equality: {
        EdgeLabel = Equal;
        LabelSize = sizeof(Equal) - 1;
      } break;
      case TypeLinkTag::LK_Instance: {
        EdgeLabel = Instance;
        LabelSize = sizeof(Instance) - 1;
        Extra = dumpToString(EdgeTag->getOffsetExpr());
      } break;
      case TypeLinkTag::LK_Inheritance: {
        EdgeLabel = Inherits;
        LabelSize = sizeof(Inherits) - 1;
      } break;
      default: {
        EdgeLabel = Unexpected;
        LabelSize = sizeof(Unexpected) - 1;
      } break;
      }
      DotFile << "  node_" << SrcNodeId << " -> node_" << TgtNode->ID
              << " [label=\"" << StringRef(EdgeLabel, LabelSize) << Extra
              << "\"];\n";
    }
  }

  DotFile << "}\n";
}

LayoutTypeSystemNode *LayoutTypeSystem::createArtificialLayoutType() {
  using LTSN = LayoutTypeSystemNode;
  LTSN *New = new (NodeAllocator) LayoutTypeSystemNode(NID);
  revng_assert(New);
  ++NID;
  EqClasses.growBy1();
  bool Success = Layouts.insert(New).second;
  revng_assert(Success);
  return New;
}

static void
fixPredSucc(LayoutTypeSystemNode *From, LayoutTypeSystemNode *Into) {

  // Helper lambdas
  const auto IsFrom = [From](const LayoutTypeSystemNode::Link &L) {
    return L.first == From;
  };
  const auto IsInto = [Into](const LayoutTypeSystemNode::Link &L) {
    return L.first == Into;
  };

  // All the predecessors of all the successors of From are updated so that they
  // point to Into
  for (auto &[Neighbor, Tag] : From->Successors) {
    auto PredBegin = Neighbor->Predecessors.begin();
    auto PredEnd = Neighbor->Predecessors.end();
    auto It = std::find_if(PredBegin, PredEnd, IsFrom);
    auto End = std::find_if_not(It, PredEnd, IsFrom);
    while (It != End) {
      auto Next = std::next(It);
      auto Extracted = Neighbor->Predecessors.extract(It);
      revng_assert(Extracted);
      Neighbor->Predecessors.insert({ Into, Extracted.value().second });
      It = Next;
    }
  }

  // All the successors of all the predecessors of From are updated so that they
  // point to Into
  for (auto &[Neighbor, Tag] : From->Predecessors) {
    auto SuccBegin = Neighbor->Successors.begin();
    auto SuccEnd = Neighbor->Successors.end();
    auto It = std::find_if(SuccBegin, SuccEnd, IsFrom);
    auto End = std::find_if_not(It, SuccEnd, IsFrom);
    while (It != End) {
      auto Next = std::next(It);
      auto Extracted = Neighbor->Successors.extract(It);
      revng_assert(Extracted);
      Neighbor->Successors.insert({ Into, Extracted.value().second });
      It = Next;
    }
  }

  // Merge all the predecessors and successors.
  {
    Into->Predecessors.insert(From->Predecessors.begin(),
                              From->Predecessors.end());
    Into->Successors.insert(From->Successors.begin(), From->Successors.end());
  }

  // Remove self-references from predecessors and successors.
  {
    const auto RemoveSelfEdges = [IsFrom, IsInto](auto &NeighborsSet) {
      auto It = NeighborsSet.begin();
      while (It != NeighborsSet.end()) {
        auto Next = std::next(It);
        if (IsInto(*It) or IsFrom(*It))
          NeighborsSet.erase(It);
        It = Next;
      }
    };
    RemoveSelfEdges(Into->Predecessors);
    RemoveSelfEdges(Into->Successors);
  }
}

static Logger<> MergeLog("dla-merge-nodes");

using LayoutTypeSystemNodePtrVec = std::vector<LayoutTypeSystemNode *>;

void LayoutTypeSystem::mergeNodes(const LayoutTypeSystemNodePtrVec &ToMerge) {
  revng_assert(ToMerge.size() > 1ULL);
  LayoutTypeSystemNode *Into = ToMerge[0];
  const unsigned IntoID = Into->ID;

  for (LayoutTypeSystemNode *From : llvm::drop_begin(ToMerge, 1)) {
    revng_assert(From != Into);
    revng_log(MergeLog, "Merging: " << From->ID << " Into: " << Into->ID);

    EqClasses.join(IntoID, From->ID);

    fixPredSucc(From, Into);
    Into->InterferingInfo = Unknown;

    // Remove From from Layouts
    bool Erased = Layouts.erase(From);
    revng_assert(Erased);
    From->~LayoutTypeSystemNode();
    NodeAllocator.Deallocate(From);
  }
}

void LayoutTypeSystem::removeNode(LayoutTypeSystemNode *ToRemove) {
  // Join the node's eq class with the removed class
  EqClasses.remove(ToRemove->ID);
  revng_log(MergeLog, "Removing " << ToRemove->ID << "\n");

  const auto IsToRemove = [ToRemove](const LayoutTypeSystemNode::Link &L) {
    return L.first == ToRemove;
  };

  for (auto &[Neighbor, Tag] : ToRemove->Successors) {
    auto PredBegin = Neighbor->Predecessors.begin();
    auto PredEnd = Neighbor->Predecessors.end();
    auto It = std::find_if(PredBegin, PredEnd, IsToRemove);
    auto End = std::find_if_not(It, PredEnd, IsToRemove);
    Neighbor->Predecessors.erase(It, End);
  }

  for (auto &[Neighbor, Tag] : ToRemove->Predecessors) {
    auto SuccBegin = Neighbor->Successors.begin();
    auto SuccEnd = Neighbor->Successors.end();
    auto It = std::find_if(SuccBegin, SuccEnd, IsToRemove);
    auto End = std::find_if_not(It, SuccEnd, IsToRemove);
    Neighbor->Successors.erase(It, End);
  }

  bool Erased = Layouts.erase(ToRemove);
  revng_assert(Erased);
  ToRemove->~LayoutTypeSystemNode();
  NodeAllocator.Deallocate(ToRemove);
}

static void moveEdgesWithoutSumming(LayoutTypeSystemNode *OldSrc,
                                    LayoutTypeSystemNode *NewSrc,
                                    LayoutTypeSystemNode *Tgt) {
  // First, move successor edges from OldSrc to NewSrc
  {
    const auto IsTgt = [Tgt](const LayoutTypeSystemNode::Link &L) {
      return L.first == Tgt;
    };

    auto &OldSucc = OldSrc->Successors;
    auto &NewSucc = NewSrc->Successors;

    auto OldSuccEnd = OldSucc.end();
    auto OldToTgtIt = std::find_if(OldSucc.begin(), OldSuccEnd, IsTgt);
    auto OldToTgtEnd = std::find_if_not(OldToTgtIt, OldSuccEnd, IsTgt);

    // Here we can move the edge descriptors directly to NewSucc, because we
    // don't need to update the offset.
    while (OldToTgtIt != OldToTgtEnd) {
      auto Next = std::next(OldToTgtIt);
      NewSucc.insert(OldSucc.extract(OldToTgtIt));
      OldToTgtIt = Next;
    }
  }

  // Then, move predecessor edges from OldSrc to NewSrc
  {
    const auto IsOldSrc = [OldSrc](const LayoutTypeSystemNode::Link &L) {
      return L.first == OldSrc;
    };

    auto &TgtPred = Tgt->Predecessors;

    auto TgtPredEnd = TgtPred.end();
    auto TgtToOldIt = std::find_if(TgtPred.begin(), TgtPredEnd, IsOldSrc);
    auto TgtToOldEnd = std::find_if_not(TgtToOldIt, TgtPredEnd, IsOldSrc);

    // Here we can extract, the edge descriptors, update they key (representing
    // the predecessor) and re-insert them, becasue we don't need to change the
    // offset.
    while (TgtToOldIt != TgtToOldEnd) {
      auto Next = std::next(TgtToOldIt);
      auto OldPredEdge = TgtPred.extract(TgtToOldIt);

      OldPredEdge.value().first = NewSrc;
      TgtPred.insert(std::move(OldPredEdge));

      TgtToOldIt = Next;
    }
  }
}

void LayoutTypeSystem::moveEdges(LayoutTypeSystemNode *OldSrc,
                                 LayoutTypeSystemNode *NewSrc,
                                 LayoutTypeSystemNode *Tgt,
                                 int64_t OffsetToSum) {

  if (not OldSrc or not NewSrc or not Tgt)
    return;

  if (not OffsetToSum)
    return moveEdgesWithoutSumming(OldSrc, NewSrc, Tgt);

  // First, move successor edges from OldSrc to NewSrc
  {
    const auto IsTgt = [Tgt](const LayoutTypeSystemNode::Link &L) {
      return L.first == Tgt;
    };

    auto &OldSucc = OldSrc->Successors;

    auto OldSuccEnd = OldSucc.end();
    auto OldToTgtIt = std::find_if(OldSucc.begin(), OldSuccEnd, IsTgt);
    auto OldToTgtEnd = std::find_if_not(OldToTgtIt, OldSuccEnd, IsTgt);

    // Add new instance links with adjusted offsets from NewSrc to Tgt.
    // Using the addInstanceLink methods already marks injects NewSrc among the
    // predecessors of Tgt, so after this we only need to remove OldSrc from
    // Tgt's predecessors and we're done.
    while (OldToTgtIt != OldToTgtEnd) {
      auto Next = std::next(OldToTgtIt);
      auto OldSuccEdge = OldSucc.extract(OldToTgtIt);

      const TypeLinkTag *EdgeTag = OldSuccEdge.value().second;
      switch (EdgeTag->getKind()) {

      case TypeLinkTag::LK_Inheritance: {
        revng_assert(OffsetToSum > 0LL);
        addInstanceLink(NewSrc, Tgt, OffsetExpression(OffsetToSum));
      } break;

      case TypeLinkTag::LK_Instance: {
        OffsetExpression NewOE = EdgeTag->getOffsetExpr();
        NewOE.Offset += OffsetToSum;
        revng_assert(NewOE.Offset >= 0LL);
        addInstanceLink(NewSrc, Tgt, std::move(NewOE));
      } break;

      case TypeLinkTag::LK_Equality:
      default:
        revng_unreachable("unexpected edge kind");
      }

      OldToTgtIt = Next;
    }
  }

  // Then, remove all the remaining info in Tgt that represent the fact that
  // OldSrc was a predecessor.
  {
    const auto IsOldSrc = [OldSrc](const LayoutTypeSystemNode::Link &L) {
      return L.first == OldSrc;
    };

    auto &TgtPred = Tgt->Predecessors;

    auto TgtPredEnd = TgtPred.end();
    auto TgtToOldIt = std::find_if(TgtPred.begin(), TgtPredEnd, IsOldSrc);
    auto TgtToOldEnd = std::find_if_not(TgtToOldIt, TgtPredEnd, IsOldSrc);

    TgtPred.erase(TgtToOldIt, TgtToOldEnd);
  }
}

static Logger<> VerifyDLALog("dla-verify-strict");

bool LayoutTypeSystem::verifyConsistency() const {
  for (LayoutTypeSystemNode *NodePtr : Layouts) {
    if (not NodePtr) {
      if (VerifyDLALog.isEnabled())
        revng_check(false);
      return false;
    }
    // Check that predecessors and successors are consistent
    for (auto &P : NodePtr->Predecessors) {
      if (P.first == nullptr) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }

      // same edge with same tag
      auto It = P.first->Successors.find({ NodePtr, P.second });
      if (It == P.first->Successors.end()) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
    for (auto &P : NodePtr->Successors) {
      if (P.first == nullptr) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }

      // same edge with same tag
      auto It = P.first->Predecessors.find({ NodePtr, P.second });
      if (It == P.first->Predecessors.end()) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }

    // Check that there are no self-edges
    for (auto &P : NodePtr->Predecessors) {
      LayoutTypeSystemNode *Pred = P.first;
      if (Pred == NodePtr) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }

    for (auto &P : NodePtr->Successors) {
      LayoutTypeSystemNode *Succ = P.first;
      if (Succ == NodePtr) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }
  return true;
}

bool LayoutTypeSystem::verifyDAG() const {
  if (not verifyConsistency())
    return false;

  if (not verifyInheritanceDAG())
    return false;

  if (not verifyInstanceDAG())
    return false;

  std::set<const LayoutTypeSystemNode *> SCCHeads;

  // A graph is a DAG if and only if all its strongly connected components have
  // size 1
  std::set<const LayoutTypeSystemNode *> Visited;
  for (const auto &Node : llvm::nodes(this)) {
    revng_assert(Node != nullptr);
    if (Visited.count(Node))
      continue;

    auto I = scc_begin(Node);
    auto E = scc_end(Node);
    for (; I != E; ++I) {
      Visited.insert(I->begin(), I->end());
      if (I.hasCycle()) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }

  return true;
}

bool LayoutTypeSystem::verifyInheritanceDAG() const {
  if (not verifyConsistency())
    return false;

  // A graph is a DAG if and only if all its strongly connected components have
  // size 1
  std::set<const LayoutTypeSystemNode *> Visited;
  for (const auto &Node : llvm::nodes(this)) {
    revng_assert(Node != nullptr);
    if (Visited.count(Node))
      continue;

    using GraphNodeT = const LayoutTypeSystemNode *;
    using InheritanceNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceEdge>;
    auto I = scc_begin(InheritanceNodeT(Node));
    auto E = scc_end(InheritanceNodeT(Node));
    for (; I != E; ++I) {
      Visited.insert(I->begin(), I->end());
      if (I.hasCycle()) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }

  return true;
}

bool LayoutTypeSystem::verifyInstanceDAG() const {
  if (not verifyConsistency())
    return false;

  // A graph is a DAG if and only if all its strongly connected components have
  // size 1
  std::set<const LayoutTypeSystemNode *> Visited;
  for (const auto &Node : llvm::nodes(this)) {
    revng_assert(Node != nullptr);
    if (Visited.count(Node))
      continue;

    using GraphNodeT = const LayoutTypeSystemNode *;
    using InstanceNodeT = EdgeFilteredGraph<GraphNodeT, isInstanceEdge>;
    auto I = scc_begin(InstanceNodeT(Node));
    auto E = scc_end(InstanceNodeT(Node));
    for (; I != E; ++I) {
      Visited.insert(I->begin(), I->end());
      if (I.hasCycle()) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }

  return true;
}

bool LayoutTypeSystem::verifyNoEquality() const {
  if (not verifyConsistency())
    return false;
  for (const auto &Node : llvm::nodes(this)) {
    using LTSN = LayoutTypeSystemNode;
    for (const auto &Edge : llvm::children_edges<const LTSN *>(Node)) {
      if (isEqualityEdge(Edge)) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }
  return true;
}

bool LayoutTypeSystem::verifyLeafs() const {
  for (const auto &Node : llvm::nodes(this)) {
    if (isLeaf(Node)) {
      if (Node->Size > 0) {
        if (VerifyDLALog.isEnabled())
          revng_check(false);
        return false;
      }
    }
  }
  return true;
}

bool LayoutTypeSystem::verifyInheritanceTree() const {
  using GraphNodeT = const LayoutTypeSystemNode *;
  using InheritanceNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceEdge>;
  using GT = GraphTraits<InheritanceNodeT>;
  for (GraphNodeT Node : llvm::nodes(this)) {
    auto Beg = GT::child_begin(Node);
    auto End = GT::child_end(Node);
    if ((Beg != End) and (std::next(Beg) != End)) {
      if (VerifyDLALog.isEnabled())
        revng_check(false);
      return false;
    }
  }
  return true;
}

unsigned VectEqClasses::growBy1() {
  ++NElems;
  grow(NElems);
  return NElems;
}

void VectEqClasses::remove(const unsigned A) {
  if (RemovedID)
    join(A, *RemovedID);
  else
    RemovedID = A;
}

bool VectEqClasses::isRemoved(const unsigned ID) const {
  // No removed nodes
  if (not RemovedID)
    return false;

  // Uncompressed map
  if (getNumClasses() == 0)
    return (findLeader(ID) == findLeader(*RemovedID));

  // Compressed map
  unsigned ElementEqClass = lookupEqClass(ID);
  unsigned RemovedEqClass = lookupEqClass(*RemovedID);
  return (ElementEqClass == RemovedEqClass);
}

std::optional<unsigned> VectEqClasses::getEqClassID(const unsigned ID) const {
  unsigned EqID = lookupEqClass(ID);
  bool IsRemoved = (RemovedID) ? lookupEqClass(*RemovedID) == EqID : false;

  if (IsRemoved)
    return {};
  return EqID;
}

std::vector<unsigned> VectEqClasses::getEqClass(const unsigned ElemID) const {
  std::vector<unsigned> EqClass;

  for (unsigned OtherID = 0; OtherID < NElems; OtherID++)
    if (haveSameEqClass(ElemID, OtherID))
      EqClass.push_back(OtherID);

  return EqClass;
}

bool VectEqClasses::haveSameEqClass(unsigned ID1, unsigned ID2) const {
  // Uncompressed map
  if (getNumClasses() == 0)
    return findLeader(ID1) == findLeader(ID2);

  // Compressed map
  return lookupEqClass(ID1) == lookupEqClass(ID2);
}

void TSDebugPrinter::printNodeContent(const LayoutTypeSystem &TS,
                                      const LayoutTypeSystemNode *N,
                                      llvm::raw_fd_ostream &File) const {
  auto EqClasses = TS.getEqClasses();

  File << DoRet;
  if (EqClasses.isRemoved(N->ID))
    File << "Removed" << DoRet;

  File << "Equivalence Class: [";
  for (auto ID : EqClasses.getEqClass(N->ID))
    File << ID << ", ";
  File << "]" << DoRet;
}

} // end namespace dla
