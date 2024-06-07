/// \file NodeSizeCalculation.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/GraphLayout/Graphs.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/ControlFlow/NodeSizeCalculation.h"
#include "revng/Yield/Function.h"

static yield::layout::Size operator+(const yield::layout::Size &LHS,
                                     const yield::layout::Size &RHS) {
  return yield::layout::Size(LHS.W + RHS.W, LHS.H + RHS.H);
}

constexpr static yield::layout::Size textSize(std::string_view Text) {
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

  return yield::layout::Size(MaximumLineLength, LineCount);
}

static yield::layout::Size textSize(const yield::TaggedString &Tagged) {
  revng_assert(!Tagged.Content().empty());
  revng_assert(Tagged.Content().find('\n') == std::string::npos);

  return yield::layout::Size(Tagged.Content().size(), 1);
}

static yield::layout::Size
textSize(const SortedVector<yield::TaggedString> &Tagged) {
  revng_assert(!Tagged.empty());

  size_t LineLength = 0;
  for (yield::TaggedString String : Tagged)
    LineLength += textSize(String).W;

  return yield::layout::Size(LineLength, 1);
}

constexpr static size_t firstLineSize(std::string_view Text) {
  size_t FirstLineEnd = Text.find('\n');
  if (FirstLineEnd == std::string_view::npos)
    return Text.size();
  else
    return Text.size() - FirstLineEnd;
}

static yield::layout::Size
fontSize(yield::layout::Size &&Input,
         yield::layout::Dimension FontSize,
         const yield::cfg::Configuration &Configuration) {
  Input.W *= FontSize * Configuration.HorizontalFontFactor;
  Input.H *= FontSize * Configuration.VerticalFontFactor;
  return std::move(Input);
}

static yield::layout::Size
fontSize(yield::layout::Size &&Input,
         yield::layout::Dimension HorizontalFontSize,
         yield::layout::Dimension VerticalFontSize,
         const yield::cfg::Configuration &Configuration) {
  Input.W *= HorizontalFontSize * Configuration.HorizontalFontFactor;
  Input.H *= VerticalFontSize * Configuration.VerticalFontFactor;
  return std::move(Input);
}

static yield::layout::Size
singleLineSize(std::string_view Text,
               float Font,
               const yield::cfg::Configuration &Configuration) {
  yield::layout::Size Result = fontSize(textSize(Text), Font, Configuration);

  Result.W += Configuration.HorizontalInstructionMarginSize * 2;
  Result.H += Configuration.VerticalInstructionMarginSize * 2;

  return Result;
}

static yield::layout::Size &appendSize(yield::layout::Size &Original,
                                       const yield::layout::Size &AddOn) {
  if (AddOn.W > Original.W)
    Original.W = AddOn.W;
  Original.H += AddOn.H;

  return Original;
}

static yield::layout::Size
instructionSize(const yield::Instruction &Instruction,
                const yield::cfg::Configuration &Configuration,
                size_t CommentIndicatorSize,
                bool IsInDelayedSlot = false) {
  // Instruction body.
  yield::layout::Size DisassembledSize = textSize(Instruction.Disassembled());
  for (const auto &Directive : Instruction.PrecedingDirectives())
    appendSize(DisassembledSize, textSize(Directive.Tags()));
  for (const auto &Directive : Instruction.FollowingDirectives())
    appendSize(DisassembledSize, textSize(Directive.Tags()));

  yield::layout::Size Result = fontSize(std::move(DisassembledSize),
                                        Configuration.InstructionFontSize,
                                        Configuration);

  // Comment and delayed slot notice.
  yield::layout::Size CommentSize;
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

  auto CommentBlockSize = fontSize(yield::layout::Size(CommentSize),
                                   Configuration.CommentFontSize,
                                   Configuration.InstructionFontSize,
                                   Configuration);
  if (CommentSize.H > 1) {
    Result.W = std::max(firstLineSize(Instruction.Comment()) + Result.W
                          + CommentIndicatorSize + 1,
                        CommentBlockSize.W);
    auto OneLine = fontSize(yield::layout::Size(1, 1),
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
                          + yield::layout::Size(CommentIndicatorSize + 1, 0),
                        Configuration.CommentFontSize,
                        Configuration));

  // Annotation.
  yield::layout::Size RawBytesLengthWithOffsets{ 0, 0 };
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

static yield::layout::Size
basicBlockSize(const yield::BasicBlock &BasicBlock,
               const yield::Function &Function,
               const model::Binary &Binary,
               const yield::cfg::Configuration &Configuration) {
  // Account for the size of the label
  namespace A = model::Architecture;
  auto LabelIndicator = A::getAssemblyLabelIndicator(Binary.Architecture());
  yield::layout::Size Indicator(LabelIndicator.size(), 0);
  yield::layout::Size Result = fontSize(textSize(BasicBlock.Label())
                                          + Indicator,
                                        Configuration.LabelFontSize,
                                        Configuration);
  Result.H += Configuration.VerticalInstructionMarginSize * 2;

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

void yield::cfg::calculateNodeSizes(PreLayoutGraph &Graph,
                                    const yield::Function &Function,
                                    const model::Binary &Binary,
                                    const Configuration &Configuration) {
  for (PreLayoutNode *Node : Graph.nodes()) {
    revng_assert(Node != nullptr);

    if (!Node->isEmpty()) {
      // A normal node.
      BasicBlockID BasicBlock = Node->getBasicBlock();
      if (auto Iterator = Function.Blocks().find(BasicBlock);
          Iterator != Function.Blocks().end()) {
        Node->Size = basicBlockSize(*Iterator, Function, Binary, Configuration);
      } else if (const auto *F = tryGetFunction(Binary, BasicBlock)) {
        Node->Size = singleLineSize(F->name().str(),
                                    Configuration.InstructionFontSize,
                                    Configuration);
      } else {
        revng_abort("The value of this node is not a known address");
      }
    } else {
      // A node without any content, most likely an entry/exit marker.
      Node->Size = { 30, 30 };
    }

    Node->Size.W += Configuration.InternalNodeMarginSize * 2;
    Node->Size.H += Configuration.InternalNodeMarginSize * 2;
  }
}
