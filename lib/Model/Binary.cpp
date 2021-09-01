/// \file Binary.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/VerifyHelper.h"

using namespace llvm;

namespace model {

struct FunctionCFGNodeData {
  FunctionCFGNodeData(MetaAddress Start) : Start(Start) {}
  MetaAddress Start;
};

using FunctionCFGNode = ForwardNode<FunctionCFGNodeData>;

/// Graph data structure to represent the CFG for verification purposes
struct FunctionCFG : public GenericGraph<FunctionCFGNode> {
private:
  MetaAddress Entry;
  std::map<MetaAddress, FunctionCFGNode *> Map;

public:
  FunctionCFG(MetaAddress Entry) : Entry(Entry) {}

public:
  MetaAddress entry() const { return Entry; }
  FunctionCFGNode *entryNode() const { return Map.at(Entry); }

public:
  FunctionCFGNode *get(MetaAddress MA) {
    FunctionCFGNode *Result = nullptr;
    auto It = Map.find(MA);
    if (It == Map.end()) {
      Result = addNode(MA);
      Map[MA] = Result;
    } else {
      Result = It->second;
    }

    return Result;
  }

  bool allNodesAreReachable() const {
    if (Map.size() == 0)
      return true;

    // Ensure all the nodes are reachable from the entry node
    df_iterator_default_set<FunctionCFGNode *> Visited;
    for (auto &Ignore : depth_first_ext(entryNode(), Visited))
      ;
    return Visited.size() == size();
  }

  bool hasOnlyInvalidExits() const {
    for (auto &[Address, Node] : Map)
      if (Address.isValid() and not Node->hasSuccessors())
        return false;
    return true;
  }
};

model::TypePath
Binary::getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize) {
  PrimitiveType Temporary(V, ByteSize);
  Type::Key PrimitiveKey{ TypeKind::Primitive, Temporary.ID };
  auto It = Types.find(PrimitiveKey);

  // If we couldn't find it, create it
  if (It == Types.end()) {
    auto *NewPrimitiveType = new PrimitiveType(V, ByteSize);
    It = Types.insert(UpcastableType(NewPrimitiveType)).first;
  }

  return getTypePath(It->get());
}

TypePath Binary::recordNewType(UpcastablePointer<Type> &&T) {
  auto It = Types.insert(T).first;
  return getTypePath(It->get());
}

bool Binary::verifyTypes() const {
  return verifyTypes(false);
}

bool Binary::verifyTypes(bool Assert) const {
  VerifyHelper VH(Assert);
  return verifyTypes(VH);
}

bool Binary::verifyTypes(VerifyHelper &VH) const {
  // All types on their own should verify
  std::set<Identifier> Names;
  for (auto &Type : Types) {
    // Verify the type
    if (not Type.get()->verify(VH))
      return VH.fail();

    // Ensure the names are unique
    auto Name = Type->name();
    if (not Names.insert(Name).second)
      return VH.fail(Twine("Multiple types with the following name: ") + Name);
  }

  return true;
}

bool Binary::verify() const {
  return verify(false);
}

bool Binary::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Binary::verify(VerifyHelper &VH) const {
  for (const Function &F : Functions) {

    // Verify individual functions
    if (not F.verify(VH))
      return VH.fail();

    // Check function calls
    for (const BasicBlock &Block : F.CFG) {
      for (const auto &Edge : Block.Successors) {

        if (Edge->Type == model::FunctionEdgeType::FunctionCall) {
          // We're in a direct call, get the callee
          const auto *Call = dyn_cast<CallEdge>(Edge.get());
          auto It = Functions.find(Call->Destination);

          // If missing, fail
          if (It == Functions.end())
            return VH.fail();

          // If call and callee prototypes differ, fail
          const Function &Callee = *It;
          if (Call->Prototype != Callee.Prototype)
            return VH.fail();
        }
      }
    }
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : DynamicFunctions) {
    if (not DF.verify(VH))
      return VH.fail();
  }

  //
  // Verify the type system
  //
  return verifyTypes(VH);
}

static FunctionCFG getGraph(const Function &F) {
  using namespace FunctionEdgeType;

  FunctionCFG Graph(F.Entry);
  for (const BasicBlock &Block : F.CFG) {
    auto *Source = Graph.get(Block.Start);

    for (const auto &Edge : Block.Successors) {
      switch (Edge->Type) {
      case DirectBranch:
      case FakeFunctionCall:
      case FakeFunctionReturn:
      case Return:
      case BrokenReturn:
      case IndirectTailCall:
      case LongJmp:
      case Unreachable:
        Source->addSuccessor(Graph.get(Edge->Destination));
        break;

      case FunctionCall:
      case IndirectCall:
        // TODO: this does not handle noreturn function calls
        Source->addSuccessor(Graph.get(Block.End));
        break;

      case Killer:
        Source->addSuccessor(Graph.get(MetaAddress::invalid()));
        break;

      case Invalid:
        revng_abort();
        break;
      }
    }
  }

  return Graph;
}

Identifier Function::name() const {
  using llvm::Twine;
  if (not CustomName.empty()) {
    return CustomName;
  } else {
    auto AutomaticName = (Twine("function_") + Entry.toString()).str();
    return Identifier::fromString(AutomaticName);
  }
}

Identifier DynamicFunction::name() const {
  using llvm::Twine;
  if (not CustomName.empty())
    return CustomName;
  else
    return Identifier(SymbolName);
}

void Function::dumpCFG() const {
  FunctionCFG CFG = getGraph(*this);
  raw_os_ostream Stream(dbg);
  WriteGraph(Stream, &CFG);
}

bool Function::verify() const {
  return verify(false);
}

bool Function::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Function::verify(VerifyHelper &VH) const {
  if (Type == FunctionType::Fake)
    return VH.maybeFail(CFG.size() == 0);

  // Verify blocks
  bool HasEntry = false;
  for (const BasicBlock &Block : CFG) {

    if (Block.Start == Entry) {
      if (HasEntry)
        return VH.fail();
      HasEntry = true;
    }

    for (const auto &Edge : Block.Successors)
      if (not Edge->verify(VH))
        return VH.fail();
  }

  if (not HasEntry)
    return VH.fail();

  // Populate graph
  FunctionCFG Graph = getGraph(*this);

  // Ensure all the nodes are reachable from the entry node
  if (not Graph.allNodesAreReachable())
    return VH.fail();

  // Ensure the only node with no successors is invalid
  if (not Graph.hasOnlyInvalidExits())
    return VH.fail();

  // Prototype is present
  if (not Prototype.isValid())
    return VH.fail();

  // Prototype is valid
  if (not Prototype.get()->verify(VH))
    return VH.fail();

  const model::Type *FunctionType = Prototype.get();
  if (not(isa<RawFunctionType>(FunctionType)
          or isa<CABIFunctionType>(FunctionType)))
    return VH.fail();

  return true;
}

bool DynamicFunction::verify() const {
  return verify(false);
}

bool DynamicFunction::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool DynamicFunction::verify(VerifyHelper &VH) const {
  // Ensure we have a name
  if (SymbolName.size() == 0)
    return VH.fail("Dynamic functions must have a SymbolName");

  // Prototype is present
  if (not Prototype.isValid())
    return VH.fail();

  // Prototype is valid
  if (not Prototype.get()->verify(VH))
    return VH.fail();

  const model::Type *FunctionType = Prototype.get();
  if (not(isa<RawFunctionType>(FunctionType)
          or isa<CABIFunctionType>(FunctionType)))
    return VH.fail();

  return true;
}

bool FunctionEdge::verify() const {
  return verify(false);
}

bool FunctionEdge::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

static bool verifyFunctionEdge(VerifyHelper &VH, const FunctionEdge &E) {
  using namespace model::FunctionEdgeType;
  // WIP
  return VH.maybeFail(
    E.Type != FunctionEdgeType::Invalid
    /* and E.Destination.isValid() == hasDestination(E.Type) */);
}

bool FunctionEdge::verify(VerifyHelper &VH) const {
  if (auto *Call = dyn_cast<CallEdge>(this))
    return VH.maybeFail(Call->verify(VH));
  else
    return verifyFunctionEdge(VH, *this);
}

bool CallEdge::verify() const {
  return verify(false);
}

bool CallEdge::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool CallEdge::verify(VerifyHelper &VH) const {
  return VH.maybeFail(verifyFunctionEdge(VH, *this) and Prototype.isValid()
                      and Prototype.get()->verify(VH));
}

Identifier BasicBlock::name() const {
  using llvm::Twine;
  if (not CustomName.empty())
    return CustomName;
  else
    return Identifier(std::string("bb_") + Start.toString());
}

bool BasicBlock::verify() const {
  return verify(false);
}

bool BasicBlock::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool BasicBlock::verify(VerifyHelper &VH) const {
  if (Start.isInvalid() or End.isInvalid() or not CustomName.verify(VH))
    return VH.fail();

  for (auto &Edge : Successors)
    if (not Edge->verify(VH))
      return VH.fail();

  return true;
}

} // namespace model

template<>
struct llvm::DOTGraphTraits<model::FunctionCFG *>
  : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool Simple = false) : DefaultDOTGraphTraits(Simple) {}

  static std::string
  getNodeLabel(const model::FunctionCFGNode *Node, const model::FunctionCFG *) {
    return Node->Start.toString();
  }

  static std::string getNodeAttributes(const model::FunctionCFGNode *Node,
                                       const model::FunctionCFG *Graph) {
    if (Node->Start == Graph->entry()) {
      return "shape=box,peripheries=2";
    }

    return "";
  }
};
