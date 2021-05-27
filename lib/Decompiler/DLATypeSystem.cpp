//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <string>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DebugHelper.h"
#include "revng/Support/IRHelpers.h"

#include "DLATypeSystem.h"

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

  unsigned AccessSizeID = 0;
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

    const auto LayoutToTypePtrsIt = LayoutToTypePtrsMap.find(L);
    if (LayoutToTypePtrsIt != LayoutToTypePtrsMap.end()) {
      DotFile << DoRet;
      const auto &TypePtrSet = LayoutToTypePtrsIt->second;
      revng_assert(not TypePtrSet.empty());
      StringRef Ret = (TypePtrSet.size() > 1) ?
                        StringRef(DoRet, sizeof(DoRet) - 1) :
                        StringRef(NoRet, sizeof(NoRet) - 1);

      for (const dla::LayoutTypePtr &P : TypePtrSet) {
        P.print(DotFile);
        DotFile << Ret;

        // Collect uses for which P is a pointer operand, so that we can print
        // them later for debug
        const llvm::Value &PtrV = P.getValue();
        for (const Use &U : PtrV.uses()) {

          const llvm::Value *PtrOp = nullptr;
          const User *Usr = U.getUser();
          if (auto *Load = dyn_cast<LoadInst>(Usr))
            PtrOp = Load->getPointerOperand();
          else if (auto *Store = dyn_cast<StoreInst>(Usr))
            PtrOp = Store->getPointerOperand();
          else
            continue;

          if (&PtrV == PtrOp)
            PtrUses.push_back(&U);
        }
      }
    }

    DotFile << "\"];\n";

    for (uint64_t AccessSize : L->AccessSizes) {
      DotFile << "  access_size_" << AccessSizeID
              << " [label=\"Access Size: " << AccessSize;

      bool Found = false;
      for (const llvm::Use *U : PtrUses) {
        if (AccessSize == getLoadStoreSizeFromPtrOpUse(*this, U)) {
          auto *I = cast<Instruction>(U->getUser());
          DotFile << "\\\\n"
                  << "In : " << I->getFunction()->getName() << " : ";
          DotFile.write_escaped(dumpToString(I));
          Found = true;
        }
      }

      DotFile << "\"];\n";
      DotFile << "  node_" << L->ID << " -> access_size_" << AccessSizeID
              << ";\n";

      revng_assert(Found);
      ++AccessSizeID;
    }
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
  bool Success = Layouts.insert(New).second;
  revng_assert(Success);
  return New;
}

static void assertGetLayoutTypePreConditions(const Value *V, unsigned Id) {
  // We accept only integers, pointer, and function types (which are actually
  // used for representing return types of functions)
  const Type *VT = V->getType();
  revng_assert(isa<FunctionType>(VT) or isa<IntegerType>(VT)
               or isa<PointerType>(VT));
  // The only case where we accept Id != max are Functions that return structs
  revng_assert(Id == std::numeric_limits<unsigned>::max()
               or cast<Function>(V)->getReturnType()->isStructTy());
}

LayoutTypeSystemNode *
LayoutTypeSystem::getLayoutType(const Value *V, unsigned Id) {

  if (V == nullptr)
    return nullptr;

  // Check pre-conditions
  assertGetLayoutTypePreConditions(V, Id);

  LayoutTypePtr Key(V, Id);
  return TypePtrToLayoutMap.at(Key);
}

std::pair<LayoutTypeSystemNode *, bool>
LayoutTypeSystem::getOrCreateLayoutType(const Value *V, unsigned Id) {

  if (V == nullptr)
    return std::make_pair(nullptr, false);

  // Check pre-conditions
  assertGetLayoutTypePreConditions(V, Id);

  LayoutTypePtr Key(V, Id);
  auto HintIt = TypePtrToLayoutMap.lower_bound(Key);
  if (HintIt != TypePtrToLayoutMap.end()
      and not TypePtrToLayoutMap.key_comp()(Key, HintIt->first)) {
    return std::make_pair(HintIt->second, false);
  }

  LayoutTypeSystemNode *Res = createArtificialLayoutType();

  // Add the mapping between the new LayoutTypeSystemNode and the LayoutTypePtr
  // that is associated to V.
  const auto &[_, Ok] = LayoutToTypePtrsMap[Res].insert(Key);
  TypePtrToLayoutMap.emplace_hint(HintIt, Key, Res);
  revng_assert(Ok);
  return std::make_pair(Res, true);
}

static void assertGetLayoutTypePreConditions(const Value &V) {
  const Type *VTy = V.getType();
  // We accept only integers, pointer, structs and and function types (which
  // are actually used for representing return types of functions)
  revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy)
               or isa<StructType>(VTy) or isa<FunctionType>(VTy));
}

SmallVector<LayoutTypeSystemNode *, 2>
LayoutTypeSystem::getLayoutTypes(const Value &V) {
  assertGetLayoutTypePreConditions(V);
  SmallVector<LayoutTypeSystemNode *, 2> Results;
  const Type *VTy = V.getType();
  if (const auto *F = dyn_cast<Function>(&V)) {
    auto *RetTy = F->getReturnType();
    if (auto *StructTy = dyn_cast<StructType>(RetTy)) {
      unsigned FieldId = 0;
      unsigned FieldNum = StructTy->getNumElements();
      for (; FieldId < FieldNum; ++FieldId) {
        auto FieldTy = StructTy->getElementType(FieldId);
        revng_assert(isa<IntegerType>(FieldTy) or isa<PointerType>(FieldTy));
        Results.push_back(getLayoutType(&V, FieldId));
      }
    } else {
      revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy));
      Results.push_back(getLayoutType(&V));
    }
  } else if (auto *StructTy = dyn_cast<StructType>(VTy)) {
    revng_assert(not isa<LoadInst>(V));

    if (isa<CallInst>(&V) or isa<PHINode>(&V)) {

      // Special handling for StructInitializers
      const Function *Callee = getCallee(cast<Instruction>(&V));
      if (Callee) {
        auto CTags = FunctionTags::TagsSet::from(Callee);
        if (CTags.contains(FunctionTags::StructInitializer)) {

          revng_assert(not Callee->isVarArg());

          auto *RetTy = cast<StructType>(Callee->getReturnType());
          revng_assert(RetTy->getNumElements() == Callee->arg_size());

          bool OnlyReturnUses = true;
          bool HasReturnUse = false;
          auto *Call = cast<CallInst>(&V);
          for (const User *U : Call->users()) {
            if (isa<ReturnInst>(U)) {
              HasReturnUse = true;

              const Function *Caller = Call->getFunction();

              if (Results.empty())
                Results = getLayoutTypes(*Caller);
              else
                revng_assert(Results == getLayoutTypes(*Caller));

              revng_assert(Results.size() == Callee->arg_size());
            } else {
              OnlyReturnUses = false;
            }
          }
          revng_assert(not HasReturnUse or OnlyReturnUses);
        }
      }

      // If Results are full, we have detected a call to a struct_initializer
      // that is returned, so we are done. Otherwise the have to look to for
      // extractvalue instructions that are extracting values from the return
      // value of the struct_initializer call.
      if (Results.empty()) {

        auto *I = cast<Instruction>(&V);
        const auto ExtractedValues = getExtractedValuesFromInstruction(I);

        Results.resize(ExtractedValues.size(), {});

        for (auto &Group : llvm::enumerate(ExtractedValues)) {
          const auto &ExtractedSet = Group.value();
          const auto FieldId = Group.index();
          // Inside here we're working on a signle field of the struct.
          // ExtractedSet contains all the ExtractValueInst that extract the
          // same field of the struct.
          // We get or create a layout type for each of them, but they should
          // all be the same.
          std::optional<LayoutTypeSystemNode *> FieldNode;
          for (const llvm::ExtractValueInst *Ext : ExtractedSet) {
            LayoutTypeSystemNode *ExtNode = getLayoutType(Ext);
            if (FieldNode.has_value()) {
              LayoutTypeSystemNode *Node = FieldNode.value();
              revng_assert(not Node or not ExtNode or (Node == ExtNode));
              if (not Node)
                Node = ExtNode;
            } else {
              FieldNode = ExtNode;
            }
          }
          Results[FieldId] = FieldNode.value_or(nullptr);
        }
      }

    } else {

      SmallVector<const Value *, 2> LeafVals;
      if (auto *Ins = dyn_cast<InsertValueInst>(&V))
        LeafVals = getInsertValueLeafOperands(Ins);
      else
        LeafVals.resize(StructTy->getNumElements(), nullptr);

      for (const Value *LeafVal : LeafVals)
        Results.push_back(getLayoutType(LeafVal));
    }
  } else {
    // For non-struct and non-function types we only add a LayoutTypeSystemNode
    Results.push_back(getLayoutType(&V));
  }
  return Results;
}

SmallVector<std::pair<LayoutTypeSystemNode *, bool>, 2>
LayoutTypeSystem::getOrCreateLayoutTypes(const Value &V) {
  assertGetLayoutTypePreConditions(V);
  using GetOrCreateResult = std::pair<LayoutTypeSystemNode *, bool>;
  SmallVector<GetOrCreateResult, 2> Results;
  const Type *VTy = V.getType();
  if (const auto *F = dyn_cast<Function>(&V)) {
    auto *RetTy = F->getReturnType();
    if (auto *StructTy = dyn_cast<StructType>(RetTy)) {
      unsigned FieldId = 0;
      unsigned FieldNum = StructTy->getNumElements();
      for (; FieldId < FieldNum; ++FieldId) {
        auto FieldTy = StructTy->getElementType(FieldId);
        revng_assert(isa<IntegerType>(FieldTy) or isa<PointerType>(FieldTy));
        Results.push_back(getOrCreateLayoutType(&V, FieldId));
      }
    } else {
      revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy));
      Results.push_back(getOrCreateLayoutType(&V));
    }
  } else if (auto *StructTy = dyn_cast<StructType>(VTy)) {
    revng_assert(not isa<LoadInst>(V));

    if (isa<CallInst>(&V) or isa<PHINode>(&V)) {

      // Special handling for StructInitializers
      const Function *Callee = getCallee(cast<Instruction>(&V));
      if (Callee) {
        auto CTags = FunctionTags::TagsSet::from(Callee);
        if (CTags.contains(FunctionTags::StructInitializer)) {

          revng_assert(not Callee->isVarArg());

          auto *RetTy = cast<StructType>(Callee->getReturnType());
          revng_assert(RetTy->getNumElements() == Callee->arg_size());

          bool OnlyReturnUses = true;
          bool HasReturnUse = false;
          auto *Call = cast<CallInst>(&V);
          for (const User *U : Call->users()) {
            if (isa<ReturnInst>(U)) {
              HasReturnUse = true;

              const Function *Caller = Call->getFunction();

              if (Results.empty())
                Results = getOrCreateLayoutTypes(*Caller);
              else
                revng_assert(Results == getOrCreateLayoutTypes(*Caller));

              revng_assert(Results.size() == Callee->arg_size());
            } else {
              OnlyReturnUses = false;
            }
          }
          revng_assert(not HasReturnUse or OnlyReturnUses);
        }
      }

      // If Results are full, we have detected a call to a struct_initializer
      // that is returned, so we are done. Otherwise the have to look to for
      // extractvalue instructions that are extracting values from the return
      // value of the struct_initializer call.
      if (Results.empty()) {

        auto *I = cast<Instruction>(&V);
        const auto ExtractedValues = getExtractedValuesFromInstruction(I);

        Results.resize(ExtractedValues.size(), {});

        for (auto &Group : llvm::enumerate(ExtractedValues)) {
          const auto &ExtractedSet = Group.value();
          const auto FieldId = Group.index();
          // Inside here we're working on a signle field of the struct.
          // ExtractedSet contains all the ExtractValueInst that extract the
          // same field of the struct.
          // We get or create a layout type for each of them, but they should
          // all be the same.
          std::optional<GetOrCreateResult> FieldResult;
          for (const llvm::ExtractValueInst *Ext : ExtractedSet) {
            GetOrCreateResult ExtResult = getOrCreateLayoutType(Ext);
            if (FieldResult.has_value()) {
              auto &[Node, New] = FieldResult.value();
              const auto &[ExtNode, ExtNew] = ExtResult;
              revng_assert(not ExtNew or ExtNode);
              if (not Node) {
                Node = ExtNode;
              } else if (ExtNode and ExtNode != Node) {
                bool AddedLink = addEqualityLink(Node, ExtNode).second;
                New |= AddedLink;
              }
              New |= ExtNew;
            } else {
              FieldResult = ExtResult;
            }
          }
          Results[FieldId] = FieldResult.value_or(GetOrCreateResult{});
        }
      }

    } else {

      SmallVector<const Value *, 2> LeafVals;
      if (auto *Ins = dyn_cast<InsertValueInst>(&V))
        LeafVals = getInsertValueLeafOperands(Ins);
      else
        LeafVals.resize(StructTy->getNumElements(), nullptr);

      for (const Value *LeafVal : LeafVals)
        Results.push_back(getOrCreateLayoutType(LeafVal));
    }
  } else {
    // For non-struct and non-function types we only add a LayoutTypeSystemNode
    Results.push_back(getOrCreateLayoutType(&V));
  }
  return Results;
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
  auto &IntoTypePtrs = LayoutToTypePtrsMap.at(Into);
  for (LayoutTypeSystemNode *From : llvm::drop_begin(ToMerge, 1)) {
    revng_assert(From != Into);
    revng_log(MergeLog, "Merging: " << From << " Into: " << Into);

    auto ToMergeLayoutToTypePtrsIt = LayoutToTypePtrsMap.find(From);
    revng_assert(ToMergeLayoutToTypePtrsIt != LayoutToTypePtrsMap.end());

    Into->AccessSizes.insert(From->AccessSizes.begin(),
                             From->AccessSizes.end());

    // Update LayoutToTypePtrsMap, the map that maps each LayoutTypeSystemNode *
    // to the set of LayoutTypePtrs that are associated to it.
    auto &MergedTypePtrs = ToMergeLayoutToTypePtrsIt->second;
    IntoTypePtrs.insert(MergedTypePtrs.begin(), MergedTypePtrs.end());

    // Update TypePtrToLayoutMap, the inverse map of LayoutToTypePtrsMap
    for (auto P : MergedTypePtrs) {
      revng_assert(TypePtrToLayoutMap.at(P) == From);
      TypePtrToLayoutMap.at(P) = Into;
    }

    fixPredSucc(From, Into);
    Into->InterferingInfo = Unknown;

    // Clear stuff in LayoutTypeToPtrsMap, because now From must be removed.
    LayoutToTypePtrsMap.erase(ToMergeLayoutToTypePtrsIt);

    // Remove From from Layouts
    bool Erased = Layouts.erase(From);
    revng_assert(Erased);
    From->~LayoutTypeSystemNode();
    NodeAllocator.Deallocate(From);
  }
}

void LayoutTypeSystem::removeNode(LayoutTypeSystemNode *ToRemove) {
  revng_assert(ToRemove);
  auto It = LayoutToTypePtrsMap.find(ToRemove);
  revng_assert(It != LayoutToTypePtrsMap.end());
  for (auto P : It->second)
    TypePtrToLayoutMap.erase(P);
  LayoutToTypePtrsMap.erase(It);

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
      if (not hasValidLayout(Node)) {
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

} // end namespace dla
