/// A collection of helper functions to improve the quality of the model/make it
/// valid.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/Queue.h"
#include "revng/Model/Processing.h"
#include "revng/Support/Debug.h"

using namespace llvm;

namespace model {

template<typename T>
void purgeFunctions(T &Functions,
                    const std::set<const model::TypeDefinition *> &ToDelete) {
  for (auto &F : Functions)
    if (F.prototype() == nullptr or ToDelete.contains(F.prototype()))
      F.Prototype().reset();
}

/// Walks a Type's reference chain through arrays and pointers to its innermost
/// TypeDefinition. EnclosingPointer holds the innermost pointer crossed (if
/// any) and is used to replace its pointee with void when the target is
/// dropped.
class ResolvedTypeReference {
public:
  const model::TypeDefinition *ReferencedDefinition = nullptr;
  model::PointerType *EnclosingPointer = nullptr;

  explicit ResolvedTypeReference(model::Type *Type) {
    while (Type != nullptr) {
      if (auto *Defined = dyn_cast<model::DefinedType>(Type)) {
        ReferencedDefinition = &Defined->unwrap();
        return;
      }

      if (auto *Array = dyn_cast<model::ArrayType>(Type)) {
        Type = Array->ElementType().get();
      } else if (auto *Pointer = dyn_cast<model::PointerType>(Type)) {
        EnclosingPointer = Pointer;
        Type = Pointer->PointeeType().get();
      } else {
        return;
      }
    }
  }

  bool isValid() const { return ReferencedDefinition != nullptr; }
};

// The reverse dependency graph uses four kinds of nodes:
//
//  * DefinitionNode: a type definition. Successors are the nodes
//    (definition, field, or pointer) that depend on it.
//  * StructFieldNode: a specific field of a struct. Leaf node, so deletion
//    never propagates through it to the parent struct.
//  * UnionFieldNode: a specific entry of a union. Also a leaf node. When
//    all entries of a union are marked for removal, the union itself is
//    deleted and deletion cascades to its dependents.
//  * PointerNode: a pointer whose pointee (directly, without crossing
//    another pointer) depends on a type definition. Leaf node. When
//    reached, the pointer's pointee is replaced with void.
//
// Only direct dependencies (no pointer in the type chain) create
// definition or field edges. Pointer-mediated references get a PointerNode
// so they are fixed up in the same worklist pass.

struct DefinitionNodeData {
  model::TypeDefinition *Definition = nullptr;
};

struct StructFieldNodeData {
  model::StructDefinition *ParentStruct = nullptr;
  uint64_t FieldOffset = 0;
};

struct UnionFieldNodeData {
  model::UnionDefinition *ParentUnion = nullptr;
  uint64_t FieldIndex = 0;
};

struct PointerNodeData {
  model::PointerType *Pointer = nullptr;
};

struct NodeData {
  std::variant<DefinitionNodeData,
               StructFieldNodeData,
               UnionFieldNodeData,
               PointerNodeData>
    Data;
};

using GraphNode = ForwardNode<NodeData>;

class ReverseDependencyGraph {
  GenericGraph<GraphNode> Graph;
  std::map<const model::TypeDefinition *, GraphNode *> DefinitionToNode;

public:
  void addDefinitionNode(model::TypeDefinition *Definition) {
    DefinitionNodeData DefinitionData = { Definition };
    NodeData NewNodeContent = { DefinitionData };
    GraphNode *NewNode = Graph.addNode(NewNodeContent);
    DefinitionToNode[Definition] = NewNode;
  }

  GraphNode *getNode(const model::TypeDefinition *Definition) const {
    return DefinitionToNode.at(Definition);
  }

  void addSuccessorNode(const model::TypeDefinition *Predecessor,
                        NodeData Content) {
    GraphNode *NewNode = Graph.addNode(std::move(Content));
    DefinitionToNode.at(Predecessor)->addSuccessor(NewNode);
  }

  /// Register an edge from a type reference to a new leaf node.
  /// If the reference is pointer-mediated, a PointerNode is created
  /// instead of \p DirectNodeContent.
  void registerEdge(model::Type *TypeReference, NodeData DirectNodeContent) {
    if (TypeReference == nullptr)
      return;

    ResolvedTypeReference Resolved(TypeReference);
    if (not Resolved.isValid())
      return;

    if (Resolved.EnclosingPointer != nullptr) {
      PointerNodeData PointerData = { Resolved.EnclosingPointer };
      NodeData PointerContent = { PointerData };
      addSuccessorNode(Resolved.ReferencedDefinition, PointerContent);
    } else {
      addSuccessorNode(Resolved.ReferencedDefinition,
                       std::move(DirectNodeContent));
    }
  }

  /// Register an edge from a type reference to an existing definition
  /// node. If the reference is pointer-mediated, a PointerNode is
  /// created instead.
  void registerDefinitionEdge(model::Type *TypeReference,
                              model::TypeDefinition *DependentDefinition) {
    if (TypeReference == nullptr)
      return;

    ResolvedTypeReference Resolved(TypeReference);
    if (not Resolved.isValid())
      return;

    if (Resolved.EnclosingPointer != nullptr) {
      PointerNodeData PointerData = { Resolved.EnclosingPointer };
      NodeData PointerContent = { PointerData };
      addSuccessorNode(Resolved.ReferencedDefinition, PointerContent);
    } else {
      GraphNode *Source = DefinitionToNode.at(Resolved.ReferencedDefinition);
      GraphNode *Dependent = DefinitionToNode.at(DependentDefinition);
      Source->addSuccessor(Dependent);
    }
  }
};

using DefinitionPointerSet = std::set<const model::TypeDefinition *>;
unsigned dropTypesDependingOnDefinitions(TupleTree<model::Binary> &Model,
                                         const DefinitionPointerSet &Types) {
  ReverseDependencyGraph DependencyGraph;

  // Create a definition node for each type
  for (model::UpcastableTypeDefinition &Type : Model->TypeDefinitions())
    DependencyGraph.addDefinitionNode(Type.get());

  // Register edges: for struct/union types, create intermediate field nodes;
  // for other types, create direct definition-to-definition edges.
  for (model::UpcastableTypeDefinition &Type : Model->TypeDefinitions()) {
    if (Types.contains(Type.get()))
      continue;

    model::TypeDefinition *Definition = Type.get();

    if (auto *Struct = dyn_cast<model::StructDefinition>(Definition)) {
      for (model::StructField &Field : Struct->Fields()) {
        model::Type *TypeReference = Field.Type().get();
        StructFieldNodeData FieldData = { Struct, Field.Offset() };
        NodeData LeafContent = { FieldData };
        DependencyGraph.registerEdge(TypeReference, LeafContent);
      }
    } else if (auto *Union = dyn_cast<model::UnionDefinition>(Definition)) {
      for (model::UnionField &Field : Union->Fields()) {
        model::Type *TypeReference = Field.Type().get();
        UnionFieldNodeData FieldData = { Union, Field.Index() };
        NodeData LeafContent = { FieldData };
        DependencyGraph.registerEdge(TypeReference, LeafContent);
      }
    } else {
      for (model::Type *Edge : Definition->edges())
        DependencyGraph.registerDefinitionEdge(Edge, Definition);
    }
  }

  // Worklist-driven traversal of the reverse dependency graph. Definition
  // nodes contribute to ToDelete and propagate to their successors; field
  // nodes are leaves that mark a single field (and may cascade a union
  // deletion); pointer nodes replace the pointer's pointee with void.
  std::set<const model::TypeDefinition *> ToDelete;
  OnceQueue<GraphNode *> Worklist;

  // Collect field removal info (applied after traversal)
  std::set<std::pair<model::StructDefinition *, uint64_t>> StructFieldsToRemove;
  std::map<model::UnionDefinition *, std::set<uint64_t>> UnionFieldsToRemove;

  // Seed the worklist with the initial set of types to delete
  for (const model::TypeDefinition *Type : Types)
    Worklist.insert(DependencyGraph.getNode(Type));

  while (not Worklist.empty()) {
    GraphNode *Node = Worklist.pop();

    if (auto *Definition = std::get_if<DefinitionNodeData>(&Node->Data)) {
      ToDelete.insert(Definition->Definition);

      for (GraphNode *Successor : Node->successors())
        Worklist.insert(Successor);
    } else if (auto
                 *StructField = std::get_if<StructFieldNodeData>(&Node->Data)) {
      StructFieldsToRemove.emplace(StructField->ParentStruct,
                                   StructField->FieldOffset);
    } else if (auto *UnionField = std::get_if<UnionFieldNodeData>(&Node
                                                                     ->Data)) {
      model::UnionDefinition *ParentUnion = UnionField->ParentUnion;
      std::set<uint64_t> &Removed = UnionFieldsToRemove[ParentUnion];
      Removed.insert(UnionField->FieldIndex);

      if (Removed.size() == ParentUnion->Fields().size())
        Worklist.insert(DependencyGraph.getNode(ParentUnion));
    } else if (auto *Pointer = std::get_if<PointerNodeData>(&Node->Data)) {
      Pointer->Pointer->PointeeType() = model::PrimitiveType::makeVoid();
    } else {
      revng_abort("Unknown node kind.");
    }
  }

  // Apply struct field removals
  for (const auto &[Struct, Offset] : StructFieldsToRemove)
    if (not ToDelete.contains(Struct))
      Struct->Fields().erase(Offset);

  // Apply union field removals and reindex remaining entries
  for (auto &[Union, Indices] : UnionFieldsToRemove) {
    if (ToDelete.contains(Union))
      continue;

    for (uint64_t FieldIndex : Indices)
      Union->Fields().erase(FieldIndex);

    size_t NewIndex = 0;
    for (model::UnionField &Field : Union->Fields())
      Field.Index() = NewIndex++;
  }

  // Purge both dynamic and local functions depending on Types
  purgeFunctions(Model->ImportedDynamicFunctions(), ToDelete);
  purgeFunctions(Model->Functions(), ToDelete);

  // Purge types depending on unresolved Types
  for (auto It = Model->TypeDefinitions().begin();
       It != Model->TypeDefinitions().end();) {
    if (ToDelete.contains(It->get()))
      It = Model->TypeDefinitions().erase(It);
    else
      ++It;
  }

  return ToDelete.size();
}

} // namespace model
