#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"

template<typename T>
concept HasCustomAndOriginalName = requires(const T &Element) {
  { Element.CustomName() } -> std::same_as<const model::Identifier &>;
  { Element.OriginalName() } -> std::same_as<const std::string &>;
};
static_assert(HasCustomAndOriginalName<model::Type>);
static_assert(HasCustomAndOriginalName<model::EnumEntry>);

class TypeCopier {
private:
  TupleTree<model::Binary> &FromModel;
  TupleTree<model::Binary> &DestinationModel;

  // Track the copied types so we can fixup references later on
  llvm::DenseMap<uint64_t, uint64_t> AlreadyCopied;
  llvm::DenseSet<model::Type *> NewTypes;

  struct NodeData {
    const UpcastablePointer<model::Type> *T;
  };
  using Node = ForwardNode<NodeData>;
  using Graph = GenericGraph<Node>;
  std::optional<Graph> TypeGraph;
  std::map<const model::Type *, Node *> TypeToNode;
  bool Finalized = false;

public:
  TypeCopier(TupleTree<model::Binary> &FromModel,
             TupleTree<model::Binary> &DestinationModel) :
    FromModel(FromModel), DestinationModel(DestinationModel) {}
  ~TypeCopier() { revng_assert(Finalized); }

  model::TypePath copyTypeInto(model::TypePath &Type) {
    ensureGraph();

    revng_assert(Type.isValid());

    model::TypePath Result;
    llvm::df_iterator_default_set<Node *> VisitedFromTheType;
    for (Node *N :
         depth_first_ext(TypeToNode.at(Type.get()), VisitedFromTheType))
      ;

    for (const auto &P : FromModel->Types()) {
      if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(P.get())) {
        // Ensure we have all the necessary primitive types
        DestinationModel->getPrimitiveType(Primitive->PrimitiveKind(),
                                           Primitive->Size());
      } else if (AlreadyCopied.count(P.get()->ID()) == 0
                 and VisitedFromTheType.contains(TypeToNode.at(P.get()))) {
        // Clone the type
        UpcastablePointer<model::Type> NewType = P;

        // Reset type ID: recordNewType will set it for us
        NewType->ID() = 0;

        // Adjust all CustomNames
        auto Visitor = [](auto &Element) {
          using T = std::decay_t<decltype(Element)>;
          if constexpr (HasCustomAndOriginalName<T>) {
            std::string CustomName = Element.CustomName().str().str();
            Element.CustomName() = model::Identifier();
            if (Element.OriginalName().empty())
              Element.OriginalName() = CustomName;
          }
        };
        visitTupleTree(NewType, Visitor, [](const auto &) {});

        revng_assert(!DestinationModel->Types().contains(NewType->key()));

        // Record the type
        auto TheType = DestinationModel->recordNewType(std::move(NewType));
        {
          model::Type *NewType = TheType.get();
          NewTypes.insert(NewType);
          AlreadyCopied.insert({ P.get()->ID(), NewType->ID() });
        }

        // Record the type we were looking for originally
        if (P->ID() == Type.get()->ID())
          Result = TheType;
      }
    }

    // TODO: consider fixing only the necessary references
    DestinationModel.initializeReferences();

    return Result;
  }

  void finalize() {
    revng_assert(not Finalized);
    Finalized = true;

    // Visit all references into the newly created types and remap them
    // according to the map
    auto Visitor = [this](auto &Element) {
      using T = std::decay_t<decltype(Element)>;
      if constexpr (std::is_same_v<T, model::TypePath>) {
        model::TypePath &Path = Element;
        if (Path.empty())
          return;

        // Extract ID from the key
        const TupleTreeKeyWrapper &TypeKey = Path.path().toArrayRef()[1];
        auto [ID, Kind] = *TypeKey.tryGet<model::Type::Key>();
        if (Kind != model::TypeKind::PrimitiveType) {
          revng_assert(AlreadyCopied.count(ID) == 1);
          Path = DestinationModel->getTypePath({ AlreadyCopied[ID], Kind });
        }
      }
    };

    for (model::Type *NewType : NewTypes)
      visitTupleTree(NewType, Visitor, [](const auto &) {});
  }

private:
  void ensureGraph() {
    if (!TypeGraph) {
      TypeGraph = Graph();

      for (const UpcastablePointer<model::Type> &T : FromModel->Types()) {
        TypeToNode[T.get()] = TypeGraph->addNode(NodeData{ &T });
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
