#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <memory>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Processing.h"

namespace {
using ModelMap = std::map<std::string, TupleTree<model::Binary>>;

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
      if (not AlreadyCopied.contains(P.get())
          and VisitedFromThePrototype.contains(TypeToNode.at(P.get()))
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
using ModelFunctionAttributes = TrackingMutableSet<
  model::FunctionAttribute::Values>;

template<typename T>
std::optional<std::pair<model::TypePath, ModelFunctionAttributes>>
findPrototypeInLocalFunctions(T &Functions, llvm::StringRef FunctionName) {
  for (auto &Function : Functions) {
    if (Function.ExportedNames().size()) {
      bool FoundAlias = false;
      for (auto &Name : Function.ExportedNames()) {
        if (Name == FunctionName) {
          FoundAlias = true;
        }
        if (FoundAlias)
          break;
        continue;
      }
      if (!FoundAlias)
        continue;
    } else {
      // Rely on OriginalName only.
      if (Function.OriginalName() != FunctionName)
        continue;
    }

    if (!Function.Prototype().isValid())
      continue;

    return std::make_pair(Function.Prototype(), Function.Attributes());
  }

  return std::nullopt;
}

template<typename T>
std::optional<std::pair<model::TypePath, ModelFunctionAttributes>>
findPrototypeInDynamicFunctions(T &Functions, llvm::StringRef FunctionName) {
  for (auto &DynamicFunction : Functions) {
    // Rely on OriginalName only.
    if (DynamicFunction.OriginalName() != FunctionName)
      continue;

    if (!DynamicFunction.Prototype().isValid())
      continue;

    return std::make_pair(DynamicFunction.Prototype(),
                          DynamicFunction.Attributes());
  }

  return std::nullopt;
}

// This represents information about dynamic function we are about to copy into
// the Model.
struct FunctionInfo {
  model::TypePath Type;
  ModelFunctionAttributes Attributes;
  std::string ModuleName;
};

std::optional<FunctionInfo> findPrototype(llvm::StringRef FunctionName,
                                          ModelMap &ModelsOfDynamicLibraries) {
  for (auto &ModelOfDep : ModelsOfDynamicLibraries) {
    auto Prototype = findPrototypeInLocalFunctions(ModelOfDep.second
                                                     ->Functions(),
                                                   FunctionName);
    if (Prototype)
      return FunctionInfo{ (*Prototype).first,
                           (*Prototype).second,
                           ModelOfDep.first };

    Prototype = findPrototypeInDynamicFunctions(ModelOfDep.second
                                                  ->ImportedDynamicFunctions(),
                                                FunctionName);
    if (Prototype)
      return FunctionInfo{ (*Prototype).first,
                           (*Prototype).second,
                           ModelOfDep.first };
  }

  return std::nullopt;
}
} // namespace
