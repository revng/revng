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

static FunctionCFG getGraph(const Binary &Binary, const Function &F) {
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
      case IndirectCall: {
        auto *CE = cast<model::CallEdge>(Edge.get());
        if (hasAttribute(Binary, *CE, model::FunctionAttribute::NoReturn))
          Source->addSuccessor(Graph.get(MetaAddress::invalid()));
        else
          Source->addSuccessor(Graph.get(Block.End));
        break;
      }

      case Killer:
        Source->addSuccessor(Graph.get(MetaAddress::invalid()));
        break;

      case Invalid:
      case Count:
        revng_abort();
        break;
      }
    }
  }

  return Graph;
}

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

void Binary::dumpCFG(const Function &F) const {
  FunctionCFG CFG = getGraph(*this, F);
  raw_os_ostream Stream(dbg);
  WriteGraph(Stream, &CFG);
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

void Binary::dump() const {
  serialize(dbg, *this);
}

std::string Binary::toString() const {
  std::string S;
  llvm::raw_string_ostream OS(S);
  serialize(OS, *this);
  return S;
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

    // Populate graph
    FunctionCFG Graph = getGraph(*this, F);

    // Ensure all the nodes are reachable from the entry node
    if (not Graph.allNodesAreReachable())
      return VH.fail();

    // Ensure the only node with no successors is invalid
    if (not Graph.hasOnlyInvalidExits())
      return VH.fail();

    // Check function calls
    for (const BasicBlock &Block : F.CFG) {
      for (const auto &Edge : Block.Successors) {

        if (Edge->Type == model::FunctionEdgeType::FunctionCall) {
          // We're in a direct call, get the callee
          const auto *Call = dyn_cast<CallEdge>(Edge.get());

          if (not Call->DynamicFunction.empty()) {
            // It's a dynamic call

            if (Call->Destination.isValid()) {
              return VH.fail("Destination must be invalid for dynamic function "
                             "calls");
            }

            auto It = ImportedDynamicFunctions.find(Call->DynamicFunction);

            // If missing, fail
            if (It == ImportedDynamicFunctions.end())
              return VH.fail("Can't find callee \"" + Call->DynamicFunction
                             + "\"");
          } else {
            // Regular call
            auto It = Functions.find(Call->Destination);

            // If missing, fail
            if (It == Functions.end())
              return VH.fail("Can't find callee");
          }
        }
      }
    }
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : ImportedDynamicFunctions) {
    if (not DF.verify(VH))
      return VH.fail();
  }

  //
  // Verify the type system
  //
  return verifyTypes(VH);
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

void Function::dump() const {
  serialize(dbg, *this);
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

void DynamicFunction::dump() const {
  serialize(dbg, *this);
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

void FunctionEdge::dump() const {
  serialize(dbg, *this);
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

  switch (E.Type) {
  case Invalid:
  case Count:
    return VH.fail();

  case DirectBranch:
  case FakeFunctionCall:
  case FakeFunctionReturn:
    if (E.Destination.isInvalid())
      return VH.fail();
    break;
  case FunctionCall: {
    const auto &Call = cast<const CallEdge>(E);
    if (not(E.Destination.isValid() == Call.DynamicFunction.empty()))
      return VH.fail();
  } break;

  case IndirectCall:
  case Return:
  case BrokenReturn:
  case IndirectTailCall:
  case LongJmp:
  case Killer:
  case Unreachable:
    if (E.Destination.isValid())
      return VH.fail();
    break;
  }

  return true;
}

bool FunctionEdge::verify(VerifyHelper &VH) const {
  if (auto *Call = dyn_cast<CallEdge>(this))
    return VH.maybeFail(Call->verify(VH));
  else
    return verifyFunctionEdge(VH, *this);
}

void CallEdge::dump() const {
  serialize(dbg, *this);
}

bool CallEdge::verify() const {
  return verify(false);
}

bool CallEdge::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool CallEdge::verify(VerifyHelper &VH) const {
  if (Type == model::FunctionEdgeType::FunctionCall) {
    // We're in a direct function call (either dynamic or not)
    bool IsDynamic = not DynamicFunction.empty();
    bool HasDestination = Destination.isValid();
    if (not HasDestination and not IsDynamic)
      return VH.fail("Direct call is missing Destination");
    else if (HasDestination and IsDynamic)
      return VH.fail("Dynamic function calls cannot have a valid Destination");

    bool HasPrototype = Prototype.isValid();
    if (HasPrototype)
      return VH.fail("Direct function calls must not have a prototype");
  } else {
    // We're in an indirect call site
    if (not Prototype.isValid() or not Prototype.get()->verify(VH))
      return VH.fail("Indirect call has must have a valid prototype");
  }

  return VH.maybeFail(verifyFunctionEdge(VH, *this));
}

Identifier BasicBlock::name() const {
  using llvm::Twine;
  if (not CustomName.empty())
    return CustomName;
  else
    return Identifier(std::string("bb_") + Start.toString());
}

void BasicBlock::dump() const {
  serialize(dbg, *this);
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
