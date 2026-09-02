//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE RegisterUsageAnalyses
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/RegisterUsageAnalyses/Liveness.h"
#include "revng/RegisterUsageAnalyses/ReachingDefinitions.h"

using namespace rua;
using namespace llvm;

struct TestGraphManager {
  llvm::LLVMContext Context;
  llvm::Module Module{ "rua-unit-test", Context };

  /// Two distinct dummy CSVs, used purely as opaque tokens to populate the rua
  /// index space (indices 0 and 1); the tests below address them by raw index.
  /// They are owned by the module above and never dereferenced by the analyses.
  std::unique_ptr<llvm::GlobalVariable> CSVs[2] = {
    std::make_unique<llvm::GlobalVariable>(Module,
                                           llvm::Type::getInt64Ty(Context),
                                           false,
                                           llvm::GlobalValue::ExternalLinkage,
                                           nullptr,
                                           "csv0"),
    std::make_unique<llvm::GlobalVariable>(Module,
                                           llvm::Type::getInt64Ty(Context),
                                           false,
                                           llvm::GlobalValue::ExternalLinkage,
                                           nullptr,
                                           "csv1"),
  };

  struct Graph {
    Graph() = delete;
    Graph(rua::Function &&Function,
          rua::BlockNode *Entry,
          rua::BlockNode *Exit,
          rua::BlockNode *Sink) :
      Function(std::move(Function)), Entry(Entry), Exit(Exit), Sink(Sink) {}
    rua::Function Function;
    rua::BlockNode *Entry = nullptr;
    rua::BlockNode *Exit = nullptr;
    rua::BlockNode *Sink = nullptr;
  };

  Graph singleNode(rua::Block::OperationsVector &&Operations) {
    rua::Function F;
    F.csvIndex(CSVs[0].get());
    F.csvIndex(CSVs[1].get());
    auto *Entry = F.addNode();
    F.setEntryNode(Entry);
    Entry->Operations = Operations;
    return { std::move(F), Entry, Entry, Entry };
  }

  Graph diamond(rua::Block::OperationsVector &&Header,
                rua::Block::OperationsVector &&Left,
                rua::Block::OperationsVector &&Right,
                rua::Block::OperationsVector &&Footer) {
    rua::Function F;
    F.csvIndex(CSVs[0].get());
    F.csvIndex(CSVs[1].get());

    auto *HeaderBlock = F.addNode();
    F.setEntryNode(HeaderBlock);
    HeaderBlock->Operations = Header;

    auto *LeftBlock = F.addNode();
    LeftBlock->Operations = Left;

    auto *RightBlock = F.addNode();
    RightBlock->Operations = Right;

    auto *FooterBlock = F.addNode();
    FooterBlock->Operations = Footer;

    HeaderBlock->addSuccessor(LeftBlock);
    HeaderBlock->addSuccessor(RightBlock);
    LeftBlock->addSuccessor(FooterBlock);
    RightBlock->addSuccessor(FooterBlock);

    return { std::move(F), HeaderBlock, FooterBlock, FooterBlock };
  }

  Graph loop(rua::Block::OperationsVector &&Header,
             rua::Block::OperationsVector &&LoopHeader,
             rua::Block::OperationsVector &&LoopBody,
             rua::Block::OperationsVector &&Footer) {
    rua::Function F;
    F.csvIndex(CSVs[0].get());
    F.csvIndex(CSVs[1].get());

    auto *HeaderBlock = F.addNode();
    F.setEntryNode(HeaderBlock);
    HeaderBlock->Operations = Header;

    auto *LoopHeaderBlock = F.addNode();
    LoopHeaderBlock->Operations = LoopHeader;

    auto *LoopBodyBlock = F.addNode();
    LoopBodyBlock->Operations = LoopBody;

    auto *FooterBlock = F.addNode();
    FooterBlock->Operations = Footer;

    // digraph {
    //   Header -> LoopHeader -> LoopBody -> LoopHeader -> Footer;
    // }
    HeaderBlock->addSuccessor(LoopHeaderBlock);
    LoopHeaderBlock->addSuccessor(LoopBodyBlock);
    LoopHeaderBlock->addSuccessor(FooterBlock);
    LoopBodyBlock->addSuccessor(LoopHeaderBlock);

    return { std::move(F), HeaderBlock, FooterBlock, FooterBlock };
  };

  Graph noReturn(rua::Block::OperationsVector &&Header,
                 rua::Block::OperationsVector &&NoReturn,
                 rua::Block::OperationsVector &&Exit) {
    rua::Function F;
    F.csvIndex(CSVs[0].get());
    F.csvIndex(CSVs[1].get());

    auto *HeaderBlock = F.addNode();
    F.setEntryNode(HeaderBlock);
    HeaderBlock->Operations = Header;

    auto *NoReturnBlock = F.addNode();
    NoReturnBlock->Operations = NoReturn;

    auto *ExitBlock = F.addNode();
    ExitBlock->Operations = Exit;

    HeaderBlock->addSuccessor(NoReturnBlock);
    HeaderBlock->addSuccessor(ExitBlock);

    auto *SinkBlock = F.addNode();
    NoReturnBlock->addSuccessor((SinkBlock));
    ExitBlock->addSuccessor((SinkBlock));

    return { std::move(F), HeaderBlock, ExitBlock, SinkBlock };
  }
};

BOOST_AUTO_TEST_CASE(LivenessTest) {
  TestGraphManager Manager;

  auto RunAnalysis = [](rua::Function &Function, BlockNode *Entry) {
    Liveness LA(Function);
    auto DefaultValue = LA.defaultValue();
    std::vector<const BidirectionalNode<Block> *> ExtremalLabels{ Entry };
    // Backward analysis: `getEntryNode(Inverse<...>)` is unreliable (returns
    // the forward entry), and a single exit doesn't reach no-return blocks.
    // Use `All` to seed RPOT from every node.
    using InverseGT = llvm::GraphTraits<llvm::Inverse<const rua::Function *>>;
    auto GetMaximalFixedPoint = mfp::getMaximalFixedPoint<Liveness, InverseGT>;
    return GetMaximalFixedPoint(mfp::MFPConfiguration<Liveness>{
      .Instance = &LA,
      .Flow = &Function,
      .Bottom = &DefaultValue,
      .ExtremalValue = &DefaultValue,
      .ExtremalLabels = &ExtremalLabels,
      .EntryLabels = mfp::All{} });
  };

  auto RunOnSingleNode =
    [&RunAnalysis,
     &Manager](rua::Block::OperationsVector &&Operations) -> BitVector {
    auto Graph = Manager.singleNode(std::move(Operations));
    return RunAnalysis(Graph.Function, Graph.Entry)[Graph.Entry].OutValue;
  };

  BitVector Result;

  // Only read a register
  Result = RunOnSingleNode({
    Operation(OperationType::Read, 0),
  });
  revng_check(Result.size() == 1);
  revng_check(Result[0]);

  // Read then write
  Result = RunOnSingleNode({
    Operation(OperationType::Read, 0),
    Operation(OperationType::Write, 0),
  });
  revng_check(Result.size() == 1);
  revng_check(Result[0]);

  // Write then read
  Result = RunOnSingleNode({
    Operation(OperationType::Write, 0),
    Operation(OperationType::Read, 0),
  });
  revng_check(Result.size() == 1);
  revng_check(not Result[0]);

  // Write another register before reading the target register
  Result = RunOnSingleNode({
    Operation(OperationType::Write, 1),
    Operation(OperationType::Read, 0),
    Operation(OperationType::Write, 0),
  });
  revng_check(Result.size() == 2);
  revng_check(Result[0]);
  revng_check(not Result[1]);

  auto RunOnDiamond =
    [&RunAnalysis,
     &Manager](rua::Block::OperationsVector &&Header,
               rua::Block::OperationsVector &&Left,
               rua::Block::OperationsVector &&Right,
               rua::Block::OperationsVector &&Footer) -> BitVector {
    auto Graph = Manager.diamond(std::move(Header),
                                 std::move(Left),
                                 std::move(Right),
                                 std::move(Footer));
    return RunAnalysis(Graph.Function, Graph.Exit)[Graph.Entry].OutValue;
  };

  // Read in footer
  Result = RunOnDiamond({}, {}, {}, { Operation(OperationType::Read, 0) });
  revng_assert(Result.size() == 1 and Result[0]);

  // Read in header
  Result = RunOnDiamond({ Operation(OperationType::Read, 0) }, {}, {}, {});
  revng_assert(Result.size() == 1);
  revng_assert(Result[0]);

  // Read in left
  Result = RunOnDiamond({}, { Operation(OperationType::Read, 0) }, {}, {});
  revng_assert(Result.size() == 1 and Result[0]);

  // Read on one path, write on the other
  Result = RunOnDiamond({},
                        { Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Read, 0) },
                        {});
  revng_assert(Result.size() == 1 and Result[0]);

  // Read in footer, write on both paths
  Result = RunOnDiamond({},
                        { Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Read, 0) });
  revng_assert(Result.size() == 1 and not Result[0]);

  // Read in footer, write on one path
  Result = RunOnDiamond({},
                        {},
                        { Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Read, 0) });
  revng_assert(Result.size() == 1 and Result[0]);

  auto RunOnLoop =
    [&RunAnalysis,
     &Manager](rua::Block::OperationsVector &&Header,
               rua::Block::OperationsVector &&LoopHeader,
               rua::Block::OperationsVector &&LoopBody,
               rua::Block::OperationsVector &&Footer) -> BitVector {
    auto Graph = Manager.loop(std::move(Header),
                              std::move(LoopHeader),
                              std::move(LoopBody),
                              std::move(Footer));
    return RunAnalysis(Graph.Function, Graph.Exit)[Graph.Entry].OutValue;
  };

  // Read in loop header, clobber in loop body
  Result = RunOnLoop({},
                     { Operation(OperationType::Read, 0) },
                     { Operation(OperationType::Write, 0) },
                     {});
  revng_assert(Result.size() == 1 and Result[0]);

  // Vice-versa
  Result = RunOnLoop({},
                     { Operation(OperationType::Write, 0) },
                     { Operation(OperationType::Read, 0) },
                     {});
  revng_assert(Result.size() == 1 and not Result[0]);

  auto RunOnNoReturn =
    [&RunAnalysis, &Manager](rua::Block::OperationsVector &&Header,
                             rua::Block::OperationsVector &&NoReturn,
                             rua::Block::OperationsVector &&Exit) -> BitVector {
    auto Graph = Manager.noReturn(std::move(Header),
                                  std::move(NoReturn),
                                  std::move(Exit));
    return RunAnalysis(Graph.Function, Graph.Exit)[Graph.Entry].OutValue;
  };

  // Read in noreturn block
  Result = RunOnNoReturn({}, { Operation(OperationType::Read, 0) }, {});
  revng_check(Result.size() == 1);
  revng_check(Result[0]);

  // Read in noreturn block, but write in entry
  Result = RunOnNoReturn({ Operation(OperationType::Write, 0) },
                         { Operation(OperationType::Read, 0) },
                         {});
  revng_check(Result.size() == 1);
  revng_check(not Result[0]);
}

BOOST_AUTO_TEST_CASE(ReachingDefinitionsTest) {
  TestGraphManager Manager;

  auto RunAnalysis = [](TestGraphManager::Graph &&F) {
    ReachingDefinitions RD(F.Function);
    auto DefaultValue = RD.defaultValue();
    std::vector ExtremalLabels{ F.Entry };

    mfp::MFPConfiguration<ReachingDefinitions> Configuration{
      .Instance = &RD,
      .Flow = &F.Function,
      .Bottom = &DefaultValue,
      .ExtremalValue = &DefaultValue,
      .ExtremalLabels = &ExtremalLabels
    };

    using namespace mfp;
    auto Results = getMaximalFixedPoint<ReachingDefinitions>(Configuration);
    return ReachingDefinitions::compute(Results[F.Exit].OutValue,
                                        Results[F.Sink].OutValue);
  };

  auto RunOnSingleNode = [&RunAnalysis,
                          &Manager](rua::Block::OperationsVector &&Operations) {
    return RunAnalysis(Manager.singleNode(std::move(Operations)));
  };

  auto Result = RunOnSingleNode({ Operation(OperationType::Write, 0) });
  revng_assert(Result[0]);

  Result = RunOnSingleNode({ Operation(OperationType::Write, 0),
                             Operation(OperationType::Read, 0) });
  revng_assert(not Result[0]);

  Result = RunOnSingleNode({ Operation(OperationType::Write, 0),
                             Operation(OperationType::Read, 0),
                             Operation(OperationType::Write, 0) });
  revng_assert(Result[0]);

  auto RunOnDiamond = [&RunAnalysis,
                       &Manager](rua::Block::OperationsVector &&Header,
                                 rua::Block::OperationsVector &&Left,
                                 rua::Block::OperationsVector &&Right,
                                 rua::Block::OperationsVector &&Footer) {
    return RunAnalysis(Manager.diamond(std::move(Header),
                                       std::move(Left),
                                       std::move(Right),
                                       std::move(Footer)));
  };

  // Write read on all paths
  Result = RunOnDiamond({ Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Read, 0) },
                        { Operation(OperationType::Read, 0) },
                        {});
  revng_assert(not Result[0]);

  // Write read on one paths
  Result = RunOnDiamond({ Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Read, 0) },
                        {},
                        {});
  revng_assert(not Result[0]);

  // Distinct writes on all paths
  Result = RunOnDiamond({},
                        { Operation(OperationType::Write, 0) },
                        { Operation(OperationType::Write, 0) },
                        {});
  revng_assert(Result[0]);

  // Write on only one path
  Result = RunOnDiamond({}, {}, { Operation(OperationType::Write, 0) }, {});
  revng_assert(Result[0]);

  auto RunOnLoop = [&RunAnalysis,
                    &Manager](rua::Block::OperationsVector &&Header,
                              rua::Block::OperationsVector &&LoopHeader,
                              rua::Block::OperationsVector &&LoopBody,
                              rua::Block::OperationsVector &&Footer) {
    return RunAnalysis(Manager.loop(std::move(Header),
                                    std::move(LoopHeader),
                                    std::move(LoopBody),
                                    std::move(Footer)));
  };

  // We read in the loop header, the write in the loop is always read
  Result = RunOnLoop({},
                     { Operation(OperationType::Read, 0) },
                     { Operation(OperationType::Write, 0) },
                     {});
  revng_assert(not Result[0]);

  // We read in the loop body, the write in the loop is read on at least one
  // path
  Result = RunOnLoop({},
                     { Operation(OperationType::Write, 0) },
                     { Operation(OperationType::Read, 0) },
                     {});
  revng_assert(not Result[0]);

  // We read in the function entry, the write in the loop is never read
  Result = RunOnLoop({ Operation(OperationType::Read, 0) },
                     { Operation(OperationType::Write, 0) },
                     {},
                     {});
  revng_assert(Result[0]);

  auto RunOnNoReturn = [&RunAnalysis,
                        &Manager](rua::Block::OperationsVector &&Header,
                                  rua::Block::OperationsVector &&NoReturn,
                                  rua::Block::OperationsVector &&Exit) {
    return RunAnalysis(Manager.noReturn(std::move(Header),
                                        std::move(NoReturn),
                                        std::move(Exit)));
  };

  // Dead write in the function entry
  Result = RunOnNoReturn({ Operation(OperationType::Write, 0) }, {}, {});
  revng_assert(Result[0]);

  // Dead write in the function exit
  Result = RunOnNoReturn({}, {}, { Operation(OperationType::Write, 0) });
  revng_assert(Result[0]);

  // Dead write in the noreturn block
  Result = RunOnNoReturn({}, { Operation(OperationType::Write, 0) }, {});
  revng_assert(not Result[0]);

  // Dead write in the exit block and the noreturn block
  Result = RunOnNoReturn({},
                         { Operation(OperationType::Write, 0) },
                         { Operation(OperationType::Write, 0) });
  revng_assert(Result[0]);
}
