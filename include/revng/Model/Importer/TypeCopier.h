#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"

// TODO: add support for handling multiple `FromModel`s. This would enable us
//       to invoke DestinationModel.initializeReferences() only once.
class TypeCopier {
private:
  TupleTree<model::Binary> &FromModel;
  TupleTree<model::Binary> &DestinationModel;

  // Track the copied types so we can fixup references later on
  llvm::DenseMap<uint64_t, uint64_t> OldToNew;
  llvm::DenseSet<model::TypeDefinition *> NewTypes;

  struct NodeData {
    const model::UpcastableTypeDefinition *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;
  std::optional<Graph> TypeGraph;
  std::map<const model::TypeDefinition *, Node *> TypeToNode;
  bool Finalized = false;
  llvm::df_iterator_default_set<Node *> Visited;

public:
  TypeCopier(TupleTree<model::Binary> &FromModel,
             TupleTree<model::Binary> &DestinationModel) :
    FromModel(FromModel), DestinationModel(DestinationModel) {}
  ~TypeCopier() { revng_assert(Finalized); }

  model::UpcastableType copyTypeInto(const model::TypeDefinition &Definition) {
    using namespace model;

    // Lazily build the graph of types
    if (not TypeGraph.has_value())
      buildGraph();

    // Collect all the dependent types
    UpcastableType Result;
    for (Node *N : depth_first_ext(TypeToNode.at(&Definition), Visited))
      ;

    if (hasBeenCopied(Definition.ID())) {
      // Return the copy
      TypeDefinition::Key Key = { OldToNew[Definition.ID()],
                                  Definition.Kind() };
      auto Reference = DestinationModel->getTypeDefinitionReference(Key);
      return model::DefinedType::make(std::move(Reference));
    } else {
      return cloneType(Definition);
    }
  }

  void finalize() {
    revng_assert(not Finalized);
    Finalized = true;

    if (not TypeGraph.has_value())
      return;

    // Copy remaining types
    for (const auto &SourceDefinition : FromModel->TypeDefinitions()) {
      if (not hasBeenCopied(SourceDefinition.get()->ID())
          and Visited.contains(TypeToNode.at(SourceDefinition.get()))) {
        cloneType(SourceDefinition);
      }
    }

    // TODO: consider fixing only the necessary references
    DestinationModel.initializeReferences();

    // Visit all references into the newly created types and remap them
    // according to the map
    auto Visitor = [this](auto &Element) {
      using T = std::decay_t<decltype(Element)>;
      if constexpr (std::is_same_v<T, model::TypeDefinitionReference>) {
        model::TypeDefinitionReference &Path = Element;
        if (Path.empty())
          return;

        // Extract ID from the key
        const TupleTreeKeyWrapper &TypeKey = Path.path().toArrayRef()[1];
        auto &&[ID, Kind] = *TypeKey.tryGet<model::TypeDefinition::Key>();
        revng_assert(OldToNew.count(ID) == 1);
        model::TypeDefinition::Key Key = { OldToNew[ID], Kind };
        Path = DestinationModel->getTypeDefinitionReference(Key);
      }
    };

    for (model::TypeDefinition *NewType : NewTypes)
      visitTupleTree(NewType, Visitor, [](const auto &) {});
  }

private:
  bool hasBeenCopied(uint64_t ID) const { return OldToNew.count(ID) != 0; }

  model::UpcastableType
  cloneType(const model::UpcastableTypeDefinition &SourceDefinition) {
    // Clone the type
    model::UpcastableTypeDefinition NewType = SourceDefinition;

    // Reset type ID: recordNewType will set it for us
    NewType->ID() = uint64_t(-1);

    // Record the type
    auto &&[D, Type] = DestinationModel->recordNewType(std::move(NewType));
    NewTypes.insert(&D);
    auto &&[_, Success] = OldToNew.insert({ SourceDefinition->ID(), D.ID() });
    revng_assert(Success);

    return Type;
  }

  void buildGraph() {
    TypeGraph = Graph();
    for (model::UpcastableTypeDefinition &T : FromModel->TypeDefinitions())
      TypeToNode[T.get()] = TypeGraph->addNode(NodeData{ &T });

    // Create type system edges
    for (model::UpcastableTypeDefinition &T : FromModel->TypeDefinitions())
      for (const model::Type *EdgeType : T->edges())
        if (const auto *Definition = EdgeType->skipToDefinition())
          TypeToNode.at(T.get())->addSuccessor(TypeToNode.at(Definition));
  }
};
