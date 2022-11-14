#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <memory>

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Processing.h"

namespace {
using ModelMap = std::map<std::string, TupleTree<model::Binary>>;

class TypeCopier {
private:
  TupleTree<model::Binary> &FromModel;

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
  copyPrototypeInto(model::TypePath &Prototype,
                    TupleTree<model::Binary> &DestinationModel) {
    ensureGraph();

    revng_assert(Prototype.isValid());
    revng_assert(Prototype.get()->Kind() == model::TypeKind::CABIFunctionType);

    std::optional<model::TypePath> Result = std::nullopt;
    llvm::df_iterator_default_set<Node *> VisitedFromThePrototype;
    for (Node *N : depth_first_ext(TypeToNode.at(Prototype.get()),
                                   VisitedFromThePrototype))
      ;

    for (const auto &P : FromModel->Types()) {
      if (VisitedFromThePrototype.contains(TypeToNode.at(P.get()))) {
        // Clone the pointer.
        UpcastablePointer<model::Type> NewType = P;
        NewType->OriginalName() = std::string(NewType->CustomName());
        NewType->CustomName() = "";
        // Record the type.
        auto TheType = DestinationModel->recordNewType(std::move(NewType));
        // The first type that was visited is the funciton type itself.
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

using TypeCopierMap = std::map<std::string, std::unique_ptr<TypeCopier>>;

template<typename T>
std::optional<model::TypePath>
findPrototypeHelper(T &Functions, llvm::StringRef FunctionName) {
  for (auto &Function : Functions) {
    // TODO: Introduce a new type of name `ExportedName` by marking the
    // `Function` as exported, explicitly.
    if (Function.OriginalName() != FunctionName)
      continue;

    if (!Function.Prototype().isValid())
      continue;

    return Function.Prototype();
  }

  return std::nullopt;
}

std::optional<std::pair<model::TypePath, std::string>>
findPrototype(llvm::StringRef FunctionName,
              ModelMap &ModelsOfDynamicLibraries) {
  for (auto &ModelOfDep : ModelsOfDynamicLibraries) {
    auto Prototype = findPrototypeHelper(ModelOfDep.second->Functions(),
                                         FunctionName);
    if (Prototype)
      return std::make_pair(*Prototype, ModelOfDep.first);

    Prototype = findPrototypeHelper(ModelOfDep.second
                                      ->ImportedDynamicFunctions(),
                                    FunctionName);
    if (Prototype)
      return std::make_pair(*Prototype, ModelOfDep.first);
  }

  return std::nullopt;
}
} // namespace
