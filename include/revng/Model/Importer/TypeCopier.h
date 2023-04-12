#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"

class TypeCopier {
private:
  TupleTree<model::Binary> &FromModel;
  // Track the types we copied to avoid copy them twice.
  std::set<model::Type *> AlreadyCopied;

  struct NodeData {
    model::Type *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;
  std::optional<Graph> TypeGraph;
  std::map<const model::Type *, Node *> TypeToNode;

public:
  TypeCopier(TupleTree<model::Binary> &Model) : FromModel(Model) {}

  std::optional<model::TypePath>
  copyTypeInto(model::TypePath &Type,
               TupleTree<model::Binary> &DestinationModel) {
    ensureGraph();

    revng_assert(Type.isValid());

    std::optional<model::TypePath> Result = std::nullopt;
    llvm::df_iterator_default_set<Node *> VisitedFromTheType;
    for (Node *N :
         depth_first_ext(TypeToNode.at(Type.get()), VisitedFromTheType))
      ;

    for (const auto &P : FromModel->Types()) {
      if (not AlreadyCopied.contains(P.get())
          and VisitedFromTheType.contains(TypeToNode.at(P.get()))
          and not(llvm::isa<model::PrimitiveType>(P.get())
                  and DestinationModel->Types().count(P->key()) != 0)) {
        // Clone the pointer.
        UpcastablePointer<model::Type> NewType = P;
        NewType->OriginalName() = std::string(NewType->CustomName());
        NewType->CustomName() = "";
        AlreadyCopied.insert(P.get());

        revng_assert(DestinationModel->Types().count(NewType->key()) == 0);

        // Record the type.
        auto TheType = DestinationModel->recordNewType(std::move(NewType));

        // The first type that was visited is the function type itself.
        if (!Result)
          Result = TheType;
      }
    }

    return Result;
  }

private:
  void ensureGraph() {
    if (!TypeGraph) {
      TypeGraph = Graph();

      for (const UpcastablePointer<model::Type> &T : FromModel->Types()) {
        TypeToNode[T.get()] = TypeGraph->addNode(NodeData{ T.get() });
      }

      // Create type system edges
      for (const UpcastablePointer<model::Type> &T : FromModel->Types()) {
        for (const model::QualifiedType &QT : T->edges()) {
          auto *DependantType = QT.UnqualifiedType().get();
          TypeToNode.at(T.get())->addSuccessor(TypeToNode.at(DependantType));
        }
      }
    }
  }
};
