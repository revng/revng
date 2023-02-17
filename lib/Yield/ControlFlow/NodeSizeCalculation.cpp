/// \file NodeSizeCalculation.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/ControlFlow/NodeSizeCalculation.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Graph.h"

static yield::Graph::Size
operator+(const yield::Graph::Size &LHS, const yield::Graph::Size &RHS) {
  return yield::Graph::Size(LHS.W + RHS.W, LHS.H + RHS.H);
}

constexpr static yield::Graph::Size textSize(std::string_view Text) {
  size_t LineCount = 0;
  size_t MaximumLineLength = 0;

  size_t PreviousPosition = 0;
  size_t CurrentPosition = Text.find('\n');
  while (CurrentPosition != std::string_view::npos) {
    size_t CurrentLineLength = CurrentPosition - PreviousPosition;
    if (CurrentLineLength > MaximumLineLength)
      MaximumLineLength = CurrentLineLength;
    ++LineCount;

    PreviousPosition = CurrentPosition;
    CurrentPosition = Text.find('\n', CurrentPosition + 1);
  }

  size_t LastLineLength = Text.size() - PreviousPosition;
  if (LastLineLength > MaximumLineLength)
    MaximumLineLength = LastLineLength;
  if (LastLineLength != 0)
    ++LineCount;

  return yield::Graph::Size(MaximumLineLength, LineCount);
}

constexpr static size_t firstLineSize(std::string_view Text) {
  size_t FirstLineEnd = Text.find('\n');
  if (FirstLineEnd == std::string_view::npos)
    return Text.size();
  else
    return Text.size() - FirstLineEnd;
}

static yield::Graph::Size
fontSize(yield::Graph::Size &&Input,
         yield::Graph::Dimension FontSize,
         const yield::cfg::Configuration &Configuration) {
  Input.W *= FontSize * Configuration.HorizontalFontFactor;
  Input.H *= FontSize * Configuration.VerticalFontFactor;
  return std::move(Input);
}

static yield::Graph::Size
fontSize(yield::Graph::Size &&Input,
         yield::Graph::Dimension HorizontalFontSize,
         yield::Graph::Dimension VerticalFontSize,
         const yield::cfg::Configuration &Configuration) {
  Input.W *= HorizontalFontSize * Configuration.HorizontalFontFactor;
  Input.H *= VerticalFontSize * Configuration.VerticalFontFactor;
  return std::move(Input);
}

static yield::Graph::Size
singleLineSize(std::string_view Text,
               float FontSize,
               const yield::cfg::Configuration &Configuration) {
  yield::Graph::Size Result = fontSize(textSize(Text), FontSize, Configuration);

  Result.W += Configuration.HorizontalInstructionMarginSize * 2;
  Result.H += Configuration.VerticalInstructionMarginSize * 2;

  return Result;
}

static yield::Graph::Size
linkSize(const BasicBlockID &Address,
         const yield::Function &Function,
         const model::Binary &Binary,
         size_t IndicatorSize = 0,
         const BasicBlockID &NextAddress = BasicBlockID::invalid()) {
  yield::Graph::Size Indicator(IndicatorSize, 0);

  if (not Address.isValid())
    return Indicator + textSize("an unknown location");

  if (const auto *F = yield::tryGetFunction(Binary, Address)) {
    return Indicator + textSize(F->name().str().str());
  } else if (NextAddress == Address) {
    return Indicator + textSize("the next instruction");
  } else if (auto Iterator = Function.ControlFlowGraph().find(Address);
             Iterator != Function.ControlFlowGraph().end()) {
    return Indicator + textSize("basic_block_at_" + Address.toString());
  } else {
    return Indicator + textSize("instruction_at_" + Address.toString());
  }
}

static yield::Graph::Size &
appendSize(yield::Graph::Size &Original, const yield::Graph::Size &AddOn) {
  if (AddOn.W > Original.W)
    Original.W = AddOn.W;
  Original.H += AddOn.H;

  return Original;
}

static yield::Graph::Size
instructionSize(const yield::Instruction &Instruction,
                const yield::cfg::Configuration &Configuration,
                size_t CommentIndicatorSize,
                bool IsInDelayedSlot = false) {
  // Instruction body.
  yield::Graph::Size Result = fontSize(textSize(Instruction.Disassembled()),
                                       Configuration.InstructionFontSize,
                                       Configuration);

  // Comment and delayed slot notice.
  yield::Graph::Size CommentSize;
  if (!Instruction.Comment().empty()) {
    CommentSize = textSize(Instruction.Comment());
    revng_assert(CommentSize.H == 1, "Multi line comments are not supported.");
    CommentSize.W += CommentIndicatorSize + 1;
  }
  if (IsInDelayedSlot) {
    auto DelayedSlotNoticeSize = textSize("delayed");
    DelayedSlotNoticeSize.W += CommentIndicatorSize + 1;
    appendSize(CommentSize, DelayedSlotNoticeSize);
  }

  auto CommentBlockSize = fontSize(yield::Graph::Size(CommentSize),
                                   Configuration.CommentFontSize,
                                   Configuration.InstructionFontSize,
                                   Configuration);
  if (CommentSize.H > 1) {
    Result.W = std::max(firstLineSize(Instruction.Comment()) + Result.W
                          + CommentIndicatorSize + 1,
                        CommentBlockSize.W);
    auto OneLine = fontSize(yield::Graph::Size(1, 1),
                            Configuration.CommentFontSize,
                            Configuration.InstructionFontSize,
                            Configuration);
    Result.H += CommentBlockSize.H - OneLine.H;
  } else {
    Result.W += CommentBlockSize.W;
  }

  // Error.
  if (!Instruction.Error().empty())
    appendSize(Result,
               fontSize(textSize(Instruction.Error())
                          + yield::Graph::Size(CommentIndicatorSize + 1, 0),
                        Configuration.CommentFontSize,
                        Configuration));

  // Annotation.
  yield::Graph::Size RawBytesLengthWithOffsets{ 0, 0 };
  RawBytesLengthWithOffsets.W += Instruction.RawBytes().size() * 3;
  RawBytesLengthWithOffsets.W += CommentIndicatorSize + 5;
  appendSize(Result,
             fontSize(textSize(Instruction.Address().toString())
                        + RawBytesLengthWithOffsets,
                      Configuration.AnnotationFontSize,
                      Configuration));

  // Account for the padding.
  Result.W += Configuration.HorizontalInstructionMarginSize * 2;
  Result.H += Configuration.VerticalInstructionMarginSize * 2;

  return Result;
}

static yield::Graph::Size
basicBlockSize(const yield::BasicBlock &BasicBlock,
               const yield::Function &Function,
               const model::Binary &Binary,
               const yield::cfg::Configuration &Configuration) {
  // Account for the size of the label
  namespace A = model::Architecture;
  auto LabelIndicator = A::getAssemblyLabelIndicator(Binary.Architecture());
  yield::Graph::Size Result = fontSize(linkSize(BasicBlock.ID(),
                                                Function,
                                                Binary,
                                                LabelIndicator.size()),
                                       Configuration.LabelFontSize,
                                       Configuration);

  namespace A = model::Architecture;
  auto CommentIndicator = A::getAssemblyCommentIndicator(Binary.Architecture());

  // Account for the sizes of each instruction.
  auto FromIterator = BasicBlock.Instructions().begin();
  auto ToIterator = std::prev(BasicBlock.Instructions().end());
  for (auto Iterator = FromIterator; Iterator != ToIterator; ++Iterator) {
    appendSize(Result,
               instructionSize(*Iterator,
                               Configuration,
                               CommentIndicator.size()));
  }
  appendSize(Result,
             instructionSize(*ToIterator++,
                             Configuration,
                             CommentIndicator.size(),
                             BasicBlock.HasDelaySlot()));
  revng_assert(ToIterator == BasicBlock.Instructions().end());

  return Result;
}

void yield::cfg::calculateNodeSizes(Graph &Graph,
                                    const yield::Function &Function,
                                    const model::Binary &Binary,
                                    const Configuration &Configuration) {
  for (yield::Node *Node : Graph.nodes()) {
    revng_assert(Node != nullptr);

    if (Node->Address.isValid()) {
      // A normal node.
      if (auto Iterator = Function.ControlFlowGraph().find(Node->Address);
          Iterator != Function.ControlFlowGraph().end()) {
        Node->Size = basicBlockSize(*Iterator, Function, Binary, Configuration);
      } else if (const auto *F = tryGetFunction(Binary, Node->Address)) {
        Node->Size = singleLineSize(F->name().str(),
                                    Configuration.InstructionFontSize,
                                    Configuration);
      } else {
        revng_abort("The value of this node is not a known address");
      }
    } else {
      // An entry node.
      Node->Size = { 30, 30 };
    }

    Node->Size.W += Configuration.InternalNodeMarginSize * 2;
    Node->Size.H += Configuration.InternalNodeMarginSize * 2;
  }
}
