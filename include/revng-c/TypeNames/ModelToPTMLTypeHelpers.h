#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/PTMLC.h"

// This is used to encapsulate all the things necessary for type inlining.
struct TypeInlineHelper {
public:
  struct NodeData {
    model::Type *T;
  };

  using Node = BidirectionalNode<NodeData>;
  using Graph = GenericGraph<Node>;

  // This Graph is being used for the purpose of inlining types only.
  struct GraphInfo {
    // The bi-directional graph used to analyze type connections in both
    // directions.
    Graph TypeGraph;
    // This is being used for speeding up the counting of the type references.
    std::map<const model::Type *, Node *> TypeToNode;
  };

private:
  GraphInfo TypeGraph;
  std::unordered_map<const model::Type *, unsigned> TypeToNumOfRefs;
  std::set<const model::Type *> TypesToInline;

public:
  TypeInlineHelper(const model::Binary &Model);

public:
  const GraphInfo &getTypeGraph() const;
  const std::set<const model::Type *> &getTypesToInline() const;
  const std::unordered_map<const model::Type *, unsigned> &
  getTypeToNumOfRefs() const;

public:
  // Collect stack frame types per model::Function.
  std::unordered_map<const model::Function *, std::set<const model::Type *>>
  findStackTypesPerFunction(const model::Binary &Model) const;

  // Collect all stack frame types, since we want to dump them inline in the
  // function body.
  std::set<const model::Type *>
  collectStackTypes(const model::Binary &Model) const;

  // Find all nested types of the `RootType` that should be inlined into it.
  std::set<const model::Type *>
  getTypesToInlineInTypeTy(const model::Binary &Model,
                           const model::Type *RootType) const;

private:
  std::set<const model::Type *>
  findTypesToInline(const model::Binary &Model, const GraphInfo &TypeGraph);

  GraphInfo buildTypeGraph(const model::Binary &Model);

  std::unordered_map<const model::Type *, unsigned>
  calculateNumOfOccurences(const model::Binary &Model);

  bool isReachableFromRootType(const model::Type *Type,
                               const model::Type *RootType,
                               const GraphInfo &TypeGraph);

  // Helper function used for finding all nested (into `RootType`) inlinable
  // types.
  std::set<const model::Type *>
  getNestedTypesToInline(const model::Type *RootType,
                         const UpcastablePointer<model::Type> &NestedTy) const;
};

extern bool declarationIsDefinition(const model::Type *T);

extern void printForwardDeclaration(const model::Type &T,
                                    ptml::PTMLIndentedOstream &Header);

// Print a declaration for a Type. The last three arguments (`TypesToInline`
// `NameOfInlineInstance` and `Qualifiers`) are being used for printing types
// inline. If the `NameOfInlineInstance` and `Qualifiers` are set, it means that
// we should print the type inline. Types that can be inline are structs, unions
// and enums.
extern void
printDeclaration(Logger<> &Log,
                 const model::Type &T,
                 ptml::PTMLIndentedOstream &Header,
                 std::map<model::QualifiedType, std::string> &AdditionalNames,
                 const model::Binary &Model,
                 const std::set<const model::Type *> &TypesToInline,
                 llvm::StringRef NameOfInlineInstance = llvm::StringRef(),
                 const std::vector<model::Qualifier> *Qualifiers = nullptr);

// Print a definition for a Type. The last three arguments (`TypesToInline`
// `NameOfInlineInstance` and `Qualifiers`) are being used for printing types
// inline. If the `NameOfInlineInstance` and `Qualifiers` are set, it means that
// we should print the type inline. Types that can be inline are structs, unions
// and enums.
extern void
printDefinition(Logger<> &Log,
                const model::Type &T,
                ptml::PTMLIndentedOstream &Header,
                std::map<model::QualifiedType, std::string> &AdditionalNames,
                const model::Binary &Model,
                const std::set<const model::Type *> &TypesToInline,
                llvm::StringRef NameOfInlineInstance = llvm::StringRef(),
                const std::vector<model::Qualifier> *Qualifiers = nullptr);

// Print a definition for a struct. The last three arguments (`TypesToInline`
// `NameOfInlineInstance` and `Qualifiers`) are being used for printing types
// inline. If the `NameOfInlineInstance` and `Qualifiers` are set, it means that
// we should print the type inline.
extern void
printDefinition(Logger<> &Log,
                const model::StructType &S,
                ptml::PTMLIndentedOstream &Header,
                const std::set<const model::Type *> &TypesToInline,
                std::map<model::QualifiedType, std::string> &AdditionalNames,
                const model::Binary &Model,
                llvm::StringRef NameOfInlineInstance = llvm::StringRef(),
                const std::vector<model::Qualifier> *Qualifiers = nullptr);

// Checks if a Type is valid candidate to inline. Types that can be inline are
// structs, unions and enums.
extern bool isCandidateForInline(const model::Type *T);
