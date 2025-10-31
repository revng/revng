/// \file DeduplicateEquivalentTypes.cpp
/// Implementation of deduplication of identical types.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Support/Debug.h"

using namespace llvm;
using namespace model;

static Logger Log("model-types-deduplication");

static RegisterModelPass R("deduplicate-equivalent-types",
                           "Best-effort deduplication of types that are "
                           "structurally equivalent",
                           model::deduplicateEquivalentTypes);

using Comparator = bool(model::TypeDefinition *, model::TypeDefinition *);
static void compareAll(SmallVector<model::TypeDefinition *> &ToTest,
                       std::function<Comparator> Compare) {
  for (auto It = ToTest.begin(), End = ToTest.end(); It != End; ++It) {
    model::TypeDefinition *Left = *It;

    auto IsDuplicate = [Compare, Left](model::TypeDefinition *Right) {
      return Compare(Left, Right);
    };

    // Compare with all those after It and delete them if match
    End = ToTest.erase(std::remove_if(It + 1, End, IsDuplicate), End);
  }
}

class TypeSystemDeduplicator {
private:
  struct TypeNode {
    model::TypeDefinition *T;
  };
  using Node = ForwardNode<TypeNode>;
  using Graph = GenericGraph<Node>;

private:
  std::vector<model::TypeDefinition *> Types;
  EquivalenceClasses<model::TypeDefinition *> StrongEquivalence;
  EquivalenceClasses<model::TypeDefinition *> WeakEquivalence;
  Graph TypeGraph;
  std::map<const model::TypeDefinition *, Node *> TypeToNode;
  std::vector<model::TypeDefinition *> VisitOrder;

private:
  TypeSystemDeduplicator(TupleTree<model::Binary> &Model) {
    for (auto &T : Model->TypeDefinitions())
      Types.push_back(&*T);
  }

public:
  static EquivalenceClasses<model::TypeDefinition *>
  run(TupleTree<model::Binary> &Model) {
    TypeSystemDeduplicator Helper(Model);
    Helper.computeWeakEquivalenceClasses();
    Helper.createTypeGraph();
    Helper.computeVisitOrder();
    Helper.computeStrongEquivalenceClasses();
    return std::move(Helper.StrongEquivalence);
  }

private:
  void computeWeakEquivalenceClasses() {
    revng_log(Log, "Computing weak equivalence classes");
    LoggerIndent Indent(Log);

    auto ComputeKey = [](model::TypeDefinition *T) {
      return std::pair{ T->Name(), T->Kind() };
    };

    // Sort types by the key (the name)
    llvm::sort(Types,
               [&ComputeKey](model::TypeDefinition *Left,
                             model::TypeDefinition *Right) {
                 return ComputeKey(Left) < ComputeKey(Right);
               });

    auto GroupStart = Types.begin();
    auto End = Types.end();
    while (GroupStart != End) {
      // Find group end and collect types
      auto GroupKey = ComputeKey(*GroupStart);
      std::string GroupName = (TypeDefinitionKind::getName(GroupKey.second)
                               + " " + GroupKey.first)
                                .str();
      revng_log(Log, "Considering \"" << GroupName << "\"");
      LoggerIndent Indent2(Log);

      auto GroupEnd = GroupStart;
      SmallVector<model::TypeDefinition *> ToTest;
      do {
        ToTest.push_back(*GroupEnd);
        ++GroupEnd;
      } while (GroupEnd != End and ComputeKey(*GroupEnd) == GroupKey);

      if (not GroupKey.first.empty()) {

        auto Compare = [this](model::TypeDefinition *Left,
                              model::TypeDefinition *Right) -> bool {
          revng_assert(Left != Right
                       and not WeakEquivalence.isEquivalent(Left, Right));
          if (Left->localCompare(*Right)) {
            revng_log(Log,
                      Left->ID()
                        << " and " << Right->ID() << " are weakly equivalent");

            // Record as weakly equivalent
            WeakEquivalence.unionSets(Left, Right);

            return true;
          } else {
            // This is kind of unusual
            if (Log.isEnabled()) {
              Log << "The following types have same kind and name but are "
                     "locally different.";
              model::TypeDefinition *M = Left;
              upcast(M, [](auto &U) { serialize(Log, U); });
              upcast(Right, [](auto &U) { serialize(Log, U); });
              Log << DoLog;
            }

            return false;
          }
        };

        compareAll(ToTest, Compare);

        revng_log(Log,
                  GroupName << " has " << ToTest.size()
                            << " non-weakly equivalent types");
      }

      GroupStart = GroupEnd;
    }
  }

  /// Create a graph of the non-local parts (including pointers)
  void createTypeGraph() {
    revng_log(Log, "Building the type graph");

    // Create nodes
    for (model::TypeDefinition *T : Types)
      TypeToNode[T] = TypeGraph.addNode(TypeNode{ T });

    // Create edges
    for (const model::TypeDefinition *Type : Types)
      for (const model::Type *EdgeType : Type->edges())
        addEdge(*Type, *EdgeType);
  }

  /// Compute a visit order: post order in the leaders of WeakEquivalence
  void computeVisitOrder() {
    revng_log(Log, "Computing visit order");

    Node *Entry = TypeGraph.addNode(TypeNode{ nullptr });
    for (Node *N : TypeGraph.nodes())
      Entry->addSuccessor(N);

    std::set<model::TypeDefinition *> Inserted;
    for (Node *N : post_order(Entry)) {

      auto LeaderIt = WeakEquivalence.findLeader(N->T);
      if (LeaderIt != WeakEquivalence.member_end()
          and not Inserted.contains(*LeaderIt)) {
        VisitOrder.push_back(*LeaderIt);
        Inserted.insert(*LeaderIt);
      }
    }

    TypeGraph.removeNode(Entry);
  }

  void computeStrongEquivalenceClasses() {
    revng_log(Log, "Computing strong equivalence classes");
    LoggerIndent Indent(Log);

    for (model::TypeDefinition *Leader : VisitOrder) {
      revng_log(Log, "Considering " << Leader->Name());
      LoggerIndent Indent2(Log);

      revng_assert(not Leader->Name().empty());
      auto LeaderIt = WeakEquivalence.findValue(Leader);
      revng_assert(LeaderIt->isLeader());

      SmallVector<model::TypeDefinition *> ToTest;
      std::copy(WeakEquivalence.member_begin(LeaderIt),
                WeakEquivalence.member_end(),
                std::back_inserter(ToTest));

      auto Compare = [this](model::TypeDefinition *Left,
                            model::TypeDefinition *Right) {
        if (Left == Right or StrongEquivalence.isEquivalent(Left, Right))
          return true;

        revng_log(Log, "Comparing " << Left->ID() << " and " << Right->ID());
        LoggerIndent Indent(Log);

        bool Result = deepCompare(Left, Right);

        revng_log(Log, "Comparison result: " << (Result ? "true" : "false"));

        return Result;
      };

      compareAll(ToTest, Compare);
    }
  }

private:
  void addEdge(const model::TypeDefinition &T, const model::Type &Type) {
    if (const model::TypeDefinition *Definition = Type.skipToDefinition())
      TypeToNode.at(&T)->addSuccessor(TypeToNode.at(Definition));
  }

  bool deepCompare(model::TypeDefinition *LeftType,
                   model::TypeDefinition *RightType) {
    Node *Left = TypeToNode.at(LeftType);
    Node *Right = TypeToNode.at(RightType);

    // Create a bidirectional map associating left and right nodes
    std::map<Node *, Node *> LeftToRight;
    std::map<Node *, Node *> RightToLeft;

    // Initialize the map with the two considered nodes
    LeftToRight[Left] = Right;
    RightToLeft[Right] = Left;

    // Perform a depth-first visit from here.
    // Note: we will stop the visit on successors that are already known to be
    // strongly equivalent.
    df_iterator_default_set<Node *> Visited;

    for (Node *LeftNode : depth_first_ext(Left, Visited)) {
      revng_log(Log, "Visiting " << LeftNode->T->ID());
      LoggerIndent Indent2(Log);

      auto RightIt = LeftToRight.find(LeftNode);
      if (RightIt == LeftToRight.end()) {
        revng_log(Log, "Mismatch");
        return false;
      }
      Node *RightNode = RightIt->second;

      // Zip out edges of the node pair: consider the destinations.
      for (auto &&[LeftSuccessor, RightSuccessor] :
           zip(LeftNode->successors(), RightNode->successors())) {
        revng_log(Log, "Visiting successor " << LeftSuccessor->T->ID());
        if (not compareSuccessor(LeftToRight,
                                 RightToLeft,
                                 Visited,
                                 LeftSuccessor,
                                 RightSuccessor))
          return false;
      }
    }

    // All those in the map are strongly equivalent
    for (auto &&[LeftNode, RightMode] : LeftToRight)
      StrongEquivalence.unionSets(Left->T, Right->T);

    return true;
  }

  /// Support function for deepCompare
  bool compareSuccessor(std::map<Node *, Node *> &LeftToRight,
                        std::map<Node *, Node *> &RightToLeft,
                        df_iterator_default_set<Node *> &Visited,
                        Node *Left,
                        Node *Right) {
    // If any of them is in the associating map, the other needs to match.
    // If it doesn't, the two nodes are not equivalent.
    auto LeftToRightIt = LeftToRight.find(Left);
    if (LeftToRightIt != LeftToRight.end()) {
      if (LeftToRightIt->second != Right) {
        revng_log(Log,
                  "We were expecting " << LeftToRightIt->second->T->ID()
                                       << " but we got " << Right->T->ID());
        return false;
      } else {
        revng_assert(RightToLeft.at(Right) == Left);
        return true;
      }
    }

    auto RightToLeftIt = RightToLeft.find(Right);
    if (RightToLeftIt != RightToLeft.end()) {
      if (RightToLeftIt->second != Left) {
        revng_log(Log,
                  "We were expecting " << RightToLeftIt->second->T->ID()
                                       << " but we got " << Left->T->ID());
        return false;
      } else {
        revng_assert(LeftToRight.at(Left) == Right);
        return true;
      }
    }

    // Record the match before actually verifying it. In any case,
    // even if we fail we'll discard this.
    LeftToRight[Left] = Right;
    RightToLeft[Right] = Left;

    // Verify the match
    if (Right->T == Left->T
        or StrongEquivalence.isEquivalent(Right->T, Left->T)) {
      // Strong equivalence, no need to visit me
      Visited.insert(Left);
      return true;
    } else if (WeakEquivalence.isEquivalent(Right->T, Left->T)
               or (Left->T->Name().empty() and Right->T->Name().empty()
                   and Left->T->localCompare(*Right->T))) {
      // Weak equivalence
      return true;
    } else {
      // Otherwise, the nodes are not equivalent
      revng_assert(not Left->T->localCompare(*Right->T));
      revng_log(Log,
                Left->T->ID()
                  << " and " << Right->T->ID() << " are locally different");
      return false;
    }
  }
};

void model::deduplicateEquivalentTypes(TupleTree<model::Binary> &Model) {
  revng_log(Log, "Deduplicating the model");
  LoggerIndent Indent(Log);

  auto EquivalentTypes = TypeSystemDeduplicator::run(Model);

  std::set<model::TypeDefinition *> ToErase;
  std::map<DefinitionReference, DefinitionReference> Replacements;
  for (auto LeaderIt = EquivalentTypes.begin(), End = EquivalentTypes.end();
       LeaderIt != End;
       ++LeaderIt) {

    if (!LeaderIt->isLeader())
      continue;

    auto LeaderR = Model->getDefinitionReference(LeaderIt->getData()->key());

    for (model::TypeDefinition *ToCollapse :
         make_range(++EquivalentTypes.member_begin(LeaderIt),
                    EquivalentTypes.member_end())) {
      Replacements[Model->getDefinitionReference(ToCollapse->key())] = LeaderR;
      ToErase.insert(ToCollapse);
    }
  }

  revng_log(Log, "Dropping " << ToErase.size() << " duplicated types");

  // Update references
  Model.replaceReferences(Replacements);

  // Actually drop the types
  llvm::erase_if(Model->TypeDefinitions(),
                 [&ToErase](model::UpcastableTypeDefinition &P) {
                   return ToErase.contains(P.get());
                 });
}
