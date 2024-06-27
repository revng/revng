#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/PTMLC.h"

/// This is used to encapsulate all the things necessary for type inlining.
struct TypeInlineHelper {
public:
  struct NodeData {
    const model::TypeDefinition *T;
  };

  using Node = BidirectionalNode<NodeData>;
  using Graph = GenericGraph<Node>;

  /// This Graph is being used for the purpose of inlining types only.
  // TODO: this should be refactored to used the same DependencyGraph used in
  // ModelToHeader. Having a separate different implementation of the same idea
  // here is bug prone, and has in fact already cause bugs in the past.
  struct GraphInfo {
    /// The bi-directional graph used to analyze type connections in both
    /// directions.
    Graph TypeGraph;

    /// This is being used for speeding up the counting of the type references.
    std::map<const model::TypeDefinition *, Node *> TypeToNode;
  };

private:
  const model::Binary &Model;
  GraphInfo TypeGraph;
  std::set<const model::TypeDefinition *> TypesToInline;

public:
  TypeInlineHelper(const model::Binary &Model);

public:
  const std::set<const model::TypeDefinition *> &getTypesToInline() const;

public:
  /// Collect stack frame types per model::Function.
  std::unordered_map<const model::Function *,
                     std::set<const model::TypeDefinition *>>
  findTypesToInlineInStacks() const;

  /// Collect all stack frame types, since we want to print them inline in
  /// the function body.
  std::set<const model::TypeDefinition *> collectTypesInlinableInStacks() const;

  /// Find all nested types of the `RootType` that should be inlined into it.
  std::set<const model::TypeDefinition *>
  getTypesToInlineInTypeTy(const model::TypeDefinition &RootType) const;

private:
  // Helper function used for finding all nested (into `RootType`) inlinable
  // types.
  std::set<const model::TypeDefinition *>
  getNestedTypesToInline(const model::TypeDefinition &RootType,
                         const model::TypeDefinition &NestedTy) const;
};

/// Returns true for types we never print definitions for (like typedefs)
///
/// \note As a side-effect this also determines whether type can be inlined:
///       there's no point inlining a type for which we only emit
///       the declaration
inline bool declarationIsDefinition(const model::TypeDefinition &T) {
  return not llvm::isa<model::StructDefinition>(&T)
         and not llvm::isa<model::UnionDefinition>(&T)
         and not llvm::isa<model::EnumDefinition>(&T);
}

/// Print a forward declaration corresponding to a given `model::TypeDefinition`
///
/// \param T the type to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
extern void printForwardDeclaration(const model::TypeDefinition &T,
                                    ptml::PTMLIndentedOstream &Header,
                                    ptml::PTMLCBuilder &B);

/// Print a declaration corresponding to a given `model::TypeDefinition`.
///
/// \param Log the logger to print the debugging information to.
/// \param T the type to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
/// \param Model the binary this type belongs to.
/// \param AdditionalNames the cache containing the names of artificial types
///        (like array wrappers).
extern void
printDeclaration(Logger<> &Log,
                 const model::TypeDefinition &T,
                 ptml::PTMLIndentedOstream &Header,
                 ptml::PTMLCBuilder &B,
                 const model::Binary &Model,
                 std::map<model::UpcastableType, std::string> &AdditionalNames);

/// Print a definition corresponding to a given `model::TypeDefinition`.
///
/// \param Log the logger to print the debugging information to.
/// \param T the type to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
/// \param Model the binary this type belongs to.
/// \param AdditionalNames the cache containing the names of artificial types
///        (like array wrappers).
/// \param TypesToInline a set of types that should be inlined when printing
///        this definition.
/// \param ForEditing a helper flag to state the intention behind the output
///        definition:
///        * when set to true, the definition is intended to be parsed
///          back (after, say, the user edited it), as such some extra
///          information (like maximum enum value) is emitted.
///        * otherwise, the type is emitted as cleanly as possible, as it is
///          intended for people, not for compilers.
extern void
printDefinition(Logger<> &Log,
                const model::TypeDefinition &T,
                ptml::PTMLIndentedOstream &Header,
                ptml::PTMLCBuilder &B,
                const model::Binary &Model,
                std::map<model::UpcastableType, std::string> &AdditionalNames,
                const std::set<const model::TypeDefinition *>
                  &TypesToInline = {},
                bool ForEditing = false);

/// Print an inline definition corresponding to a given `model::TypeDefinition`.
///
/// \param Log the logger to print the debugging information to.
/// \param Name the name of the variable/field/etc being printed.
/// \param T the type to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
/// \param Model the binary this type belongs to.
/// \param AdditionalNames the cache containing the names of artificial types
///        (like array wrappers).
/// \param TypesToInline a set of types that should be inlined when printing
///        this definition.
extern void printInlineDefinition(Logger<> &Log,
                                  llvm::StringRef Name,
                                  const model::Type &T,
                                  ptml::PTMLIndentedOstream &Header,
                                  ptml::PTMLCBuilder &B,
                                  const model::Binary &Model,
                                  std::map<model::UpcastableType, std::string>
                                    &AdditionalNames,
                                  const std::set<const model::TypeDefinition *>
                                    &TypesToInline);

/// Print an inline definition corresponding to a given struct.
///
/// \param Log the logger to print the debugging information to.
/// \param Struct the type to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
/// \param Model the binary this type belongs to.
/// \param AdditionalNames the cache containing the names of artificial types
///        (like array wrappers).
/// \param TypesToInline a set of types that should be inlined when printing
///        this definition.
/// \param Suffix a string to append after the definition, usually is just
///        the name of the field/variable.
extern void printInlineDefinition(Logger<> &Log,
                                  const model::StructDefinition &Struct,
                                  ptml::PTMLIndentedOstream &Header,
                                  ptml::PTMLCBuilder &B,
                                  const model::Binary &Model,
                                  std::map<model::UpcastableType, std::string>
                                    &AdditionalNames,
                                  const std::set<const model::TypeDefinition *>
                                    &TypesToInline,
                                  std::string &&Suffix = "");

/// Print a declaration corresponding to a given typedef.
///
/// \param TD the typedef to print the declaration for.
/// \param Header the stream to output the results to.
/// \param B the ptml tagging helper.
extern void printDeclaration(const model::TypedefDefinition &TD,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B);
