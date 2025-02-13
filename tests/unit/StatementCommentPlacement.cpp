/// \file Model.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE Model
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/CommentPlacementHelper.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

struct TestStatement {
  std::string Value;
  SortedVector<MetaAddress> Addresses;
};
struct TestGraphNode {
  std::vector<TestStatement> Statements;
};
struct TestStatementGraph : GenericGraph<ForwardNode<TestGraphNode>> {
  using GenericGraph<ForwardNode<TestGraphNode>>::GenericGraph;

  SortedVector<MetaAddress> &addresses(llvm::StringRef StatementValue) {
    for (Node *N : nodes())
      for (TestStatement &Statement : N->Statements)
        if (Statement.Value == StatementValue)
          return Statement.Addresses;

    revng_abort("Unknown statement requested");
  }
};

template<>
struct yield::StatementTraits<TestStatementGraph::Node *> {

  using StatementType = const TestStatement *;
  using LocationType = const SortedVector<MetaAddress>;

  static RangeOf<StatementType> auto
  getStatements(const TestStatementGraph::Node *Node) {
    return Node->Statements
           | std::views::transform([](auto &&R) { return &R; });
  }

  static LocationType getAddresses(StatementType Statement) {
    return Statement->Addresses;
  }
};

using TSGNode = TestStatementGraph::Node;
using CommentPlacementHelper = yield::CommentPlacementHelper<TSGNode *>;

static TSGNode *makeNode(std::vector<TestStatement> &&Statements,
                         TestStatementGraph &Graph) {
  return Graph.addNode(std::move(Statements));
}

static MetaAddress operator""_ma(const char *String, uint64_t Size) {
  return MetaAddress::fromString(llvm::StringRef{ String, Size });
}

static TestStatementGraph buildTheTestGraph() {
  TestStatementGraph Result;

  // (A -> B) -> (C) -> (D) -> (E) -> (F) -> (G) -> (H)
  //       |             ^      ^             |
  //       |---> (I) ----|      |---- (J) <---|

  Result = TestStatementGraph{};
  // NOTE: the default addresses provided are arbitrary, individual tests
  //       should always set the addresses for relevant statements.
  auto *AB = makeNode({ { "A", { "0x1001:Generic64"_ma } },
                        { "B", { "0x1002:Generic64"_ma } } },
                      Result);
  auto *C = makeNode({ { "C", { "0x1003:Generic64"_ma } } }, Result);
  AB->addSuccessor(C);
  auto *D = makeNode({ { "D", { "0x1004:Generic64"_ma } } }, Result);
  C->addSuccessor(D);
  auto *E = makeNode({ { "E", { "0x1005:Generic64"_ma } } }, Result);
  D->addSuccessor(E);
  auto *F = makeNode({ { "F", { "0x1006:Generic64"_ma } } }, Result);
  E->addSuccessor(F);
  auto *G = makeNode({ { "G", { "0x1007:Generic64"_ma } } }, Result);
  F->addSuccessor(G);
  auto *H = makeNode({ { "H", { "0x1008:Generic64"_ma } } }, Result);
  G->addSuccessor(H);

  auto *I = makeNode({ { "I", { "0x1009:Generic64"_ma } } }, Result);
  AB->addSuccessor(I);
  I->addSuccessor(D);
  auto *J = makeNode({ { "J", { "0x100a:Generic64"_ma } } }, Result);
  G->addSuccessor(J);
  J->addSuccessor(E);

  Result.setEntryNode(AB);

  return Result;
}

static std::string emitTestGraph(const TestStatementGraph &Graph,
                                 const CommentPlacementHelper &CM,
                                 const model::Function &Function) {
  std::string Result;

  for (const TestStatementGraph::Node *Node : Graph.nodes()) {
    for (const TestStatement &Statement : Node->Statements) {
      for (const auto &C : CM.getComments(&Statement))
        Result += "\n" + Function.Comments().at(C.CommentIndex).Body() + "\n";

      Result += Statement.Value;
    }
  }

  for (const auto &C : CM.getHomelessComments())
    Result += "\n" + Function.Comments().at(C.CommentIndex).Body() + "\n";

  return Result;
}

static void checkResult(llvm::StringRef Result, llvm::StringRef Expected) {
  if (Result != Expected) {
    std::string Error = "Emitted:\n" + Result.str()
                        + "\n\nDiffers from the expected:\n" + Expected.str()
                        + "\n\n";
    revng_abort(Error.c_str());
  }
}

using CommentVector = std::vector<model::StatementComment>;
static CommentVector deserialize(llvm::StringRef Serialized) {
  return llvm::cantFail(fromString<CommentVector>(Serialized));
}

// Basic test with no comments to see if the underlying infrastructure
// works
BOOST_AUTO_TEST_CASE(NoComments) {
  TestStatementGraph Graph = buildTheTestGraph();

  model::Function Function;
  Function.Comments() = {};

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// The simples possible case: a single comment with a single address that has
// a single statement that perfectly matches it.
BOOST_AUTO_TEST_CASE(SimpleComment) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x1:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: SimpleComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "AB\n"
                                        "SimpleComment\n"
                                        "CDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// This case is similar to the previous one in how there's a perfect match,
// but there are multiple addresses involved.
BOOST_AUTO_TEST_CASE(MultiAddressComment) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x1:Generic64"_ma,
                           "0x2:Generic64"_ma,
                           "0x3:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: MultiAddressComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x2:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "AB\n"
                                        "MultiAddressComment\n"
                                        "CDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// In this case, the match is not perfect, but there's only one viable option.
BOOST_AUTO_TEST_CASE(ImperfectlyMatchedComment) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x1:Generic64"_ma,
                           "0x2:Generic64"_ma,
                           "0x3:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: ImperfectlyMatchedComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "AB\n"
                                        "ImperfectlyMatchedComment\n"
                                        "CDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// In this case, there are multiple matches and, as the result, comment is
// emitted more than once.
BOOST_AUTO_TEST_CASE(CommentWithMultipleMatches) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("I") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentWithMultipleMatches\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "AB\n"
                                        "CommentWithMultipleMatches\n"
                                        "CDEFGH\n"
                                        "CommentWithMultipleMatches\n"
                                        "IJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// But, if one more match is added, it doesn't necessarily mean that an extra
// duplicate is necessary, for example adding the same address to statement `B`
// (that dominates every other statement) leads to the other two stamenets not
// having comments attached.
BOOST_AUTO_TEST_CASE(CommentWithDominatedMatches) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("B") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("C") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("I") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentWithDominatedMatches\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "A\n"
                                        "CommentWithDominatedMatches\n"
                                        "BCDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// To illustrate domination even more clearly, here's a simple sequential
// example. The only way to get to `F` is through `E`, as such there's no point
// duplicating the comment.
BOOST_AUTO_TEST_CASE(CommentWithSequentialMatches) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("E") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("F") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentWithSequentialMatches\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCD\n"
                                        "CommentWithSequentialMatches\n"
                                        "EFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// And of course, whether two statements belong to the same block or not,
// the behavior is the same:
BOOST_AUTO_TEST_CASE(CommentWithMatchesWithinABlock) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("A") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("B") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentWithMatchesWithinABlock\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "\nCommentWithMatchesWithinABlock\n"
                                        "ABCDEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// Speaking about blocks, domination is especially relevant when loops are
// involved, in this case because `J` is on a backedge (so it can only be
// reached after going through the main body of the loop), the comment is only
// emitted for `F`.
BOOST_AUTO_TEST_CASE(CommentWithinALoop) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("F") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("J") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentWithinALoop\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCDE\n"
                                        "CommentWithinALoop\n"
                                        "FGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// Similarly, there's no guarantee that the backedge of the loop dominates
// its exit point, as such we have to emit both in cases like that.
BOOST_AUTO_TEST_CASE(CommentInABackedge) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("J") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("H") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentInABackedge\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCDEFG\n"
                                        "CommentInABackedge\n"
                                        "HI\n"
                                        "CommentInABackedge\n"
                                        "J";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// Unless the entire loop is dominated, that is.
BOOST_AUTO_TEST_CASE(CommentInADominatedLoop) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("J") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("H") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };
  Graph.addresses("D") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: CommentInADominatedLoop\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABC\n"
                                        "CommentInADominatedLoop\n"
                                        "DEFGHIJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// If the comment doesn't match *any* statements, it's considered "homeless"
// and emitted separately (in this case, at the very end).
BOOST_AUTO_TEST_CASE(HomelessComment) {
  TestStatementGraph Graph = buildTheTestGraph();

  model::Function Function;
  Function.Comments() = deserialize("- Body: HomelessComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCDEFGHIJ\n"
                                        "HomelessComment\n";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// Note that comments are attached to specific *locations* as opposed to
// specific statements, so if there's a tie (see example below), one of
// the locations is selected arbitrarily (lexicographically, to be exact).
BOOST_AUTO_TEST_CASE(LexicographicallyPlacedComment) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x2:Generic64"_ma, "0x3:Generic64"_ma };
  Graph.addresses("I") = { "0x1:Generic64"_ma, "0x2:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: LexicographicallyPlacedComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "ABCDEFGH\n"
                                        "LexicographicallyPlacedComment\n"
                                        "IJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}

// Also, multiple comments can be attached to the same location, even if
// the match is not exact, for example:
// (note that all the duplication and domination rules illustrated in previous
// tests still apply)
BOOST_AUTO_TEST_CASE(MultipleCommentsOnTheSameLocation) {
  TestStatementGraph Graph = buildTheTestGraph();

  Graph.addresses("C") = { "0x3:Generic64"_ma };
  Graph.addresses("I") = { "0x3:Generic64"_ma };

  model::Function Function;
  Function.Comments() = deserialize("- Body: FirstComment\n"
                                    "  Location:\n"
                                    "    - 0x1:Generic64\n"
                                    "    - 0x3:Generic64\n"
                                    "- Body: SecondComment\n"
                                    "  Location:\n"
                                    "    - 0x2:Generic64\n"
                                    "    - 0x3:Generic64\n");

  CommentPlacementHelper CM(Function, Graph);

  constexpr std::string_view Expected = "AB\n"
                                        "FirstComment\n\n"
                                        "SecondComment\n"
                                        "CDEFGH\n"
                                        "FirstComment\n\n"
                                        "SecondComment\n"
                                        "IJ";
  checkResult(emitTestGraph(Graph, CM, Function), Expected);
}
