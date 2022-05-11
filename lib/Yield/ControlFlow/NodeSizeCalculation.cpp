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
linkSize(const MetaAddress &Address,
         const yield::Function &Function,
         const model::Binary &Binary,
         size_t IndicatorSize = 0,
         const MetaAddress &NextAddress = MetaAddress::invalid()) {
  yield::Graph::Size Indicator(IndicatorSize, 0);

  if (Address.isInvalid())
    return Indicator + textSize("an unknown location");

  if (auto Iterator = Binary.Functions.find(Address);
      Iterator != Binary.Functions.end()) {
    return Indicator + textSize(Iterator->name().str().str());
  } else if (NextAddress == Address) {
    return Indicator + textSize("the next instruction");
  } else if (auto Iterator = Function.ControlFlowGraph.find(Address);
             Iterator != Function.ControlFlowGraph.end()) {
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
  yield::Graph::Size Result = fontSize(textSize(Instruction.Disassembled),
                                       Configuration.InstructionFontSize,
                                       Configuration);

  // Comment and delayed slot notice.
  yield::Graph::Size CommentSize;
  if (!Instruction.Comment.empty()) {
    CommentSize = textSize(Instruction.Comment);
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
    Result.W = std::max(firstLineSize(Instruction.Comment) + Result.W
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
  if (!Instruction.Error.empty())
    appendSize(Result,
               fontSize(textSize(Instruction.Error)
                          + yield::Graph::Size(CommentIndicatorSize + 1, 0),
                        Configuration.CommentFontSize,
                        Configuration));

  // Instruction address.
  appendSize(Result,
             fontSize(textSize(Instruction.Address.toString())
                        + yield::Graph::Size(CommentIndicatorSize + 1, 0),
                      Configuration.InstructionAddressFontSize,
                      Configuration));

  // Raw instruction bytes.
  appendSize(Result,
             fontSize(yield::Graph::Size(Instruction.RawBytes.size() * 3
                                           + CommentIndicatorSize,
                                         1),
                      Configuration.InstructionBytesFontSize,
                      Configuration));

  // Account for the padding
  Result.W += Configuration.HorizontalInstructionMarginSize * 2;
  Result.H += Configuration.VerticalInstructionMarginSize * 2;

  return Result;
}

using ParsedSuccessorVector = llvm::SmallVector<efa::ParsedSuccessor, 4>;

class TargetSizeHelper {
private:
  const yield::BasicBlock &BasicBlock;
  const yield::Function &Function;
  const model::Binary &Binary;
  const yield::Graph::Dimension PrefixSize;

public:
  TargetSizeHelper(const yield::BasicBlock &BasicBlock,
                   const yield::Function &Function,
                   const model::Binary &Binary,
                   const yield::Graph::Dimension &PrefixSize) :
    BasicBlock(BasicBlock),
    Function(Function),
    Binary(Binary),
    PrefixSize(PrefixSize) {}

  yield::Graph::Size singleTarget(const efa::ParsedSuccessor &Target) {
    const auto &[NextAddress, CallAddress] = Target;
    if (NextAddress.isValid()) {
      if (CallAddress.isValid()) {
        yield::Graph::Size Result{ PrefixSize, 1 };
        Result.W += textSize("then goes to ").W + targetLinkWidth(NextAddress);
        return appendSize(Result, call({ CallAddress }));
      } else {
        yield::Graph::Size Result{ PrefixSize, 1 };
        Result.W += textSize("always goes to ").W
                    + targetLinkWidth(NextAddress);
        return Result;
      }
    } else {
      if (CallAddress.isValid()) {
        yield::Graph::Size Result{ PrefixSize, 1 };
        Result.W += textSize("and does not return").W;
        return appendSize(Result, call({ CallAddress }));
      } else {
        return { 0, 0 };
      }
    }
  }

  yield::Graph::Size twoTargets(const efa::ParsedSuccessor &First,
                                const efa::ParsedSuccessor &Second) {
    if (First.OptionalCallAddress.isValid()
        || Second.OptionalCallAddress.isValid()) {
      return multipleTargets({ First, Second });
    }

    MetaAddress FirstTarget = First.NextInstructionAddress;
    MetaAddress SecondTarget = Second.NextInstructionAddress;
    if (FirstTarget == SecondTarget)
      return singleTarget(First);

    if (FirstTarget == BasicBlock.End)
      std::swap(FirstTarget, SecondTarget);

    if (SecondTarget == BasicBlock.End) {
      yield::Graph::Size R1{ PrefixSize, 1 };
      R1.W += textSize("if taken, goes to ,").W + targetLinkWidth(FirstTarget);

      yield::Graph::Size R2{ PrefixSize, 1 };
      R2.W += textSize("otherwise, goes to ").W + targetLinkWidth(SecondTarget);

      return appendSize(R1, R2);
    } else {
      return multipleTargets({ First, Second });
    }
  }

  yield::Graph::Size multipleTargets(const ParsedSuccessorVector &Targets,
                                     bool HasUnknownTargets = false) {
    llvm::SmallVector<MetaAddress, 4> CallAddresses;
    for (const auto &[_, Target] : Targets)
      if (Target.isValid())
        CallAddresses.emplace_back(Target);

    yield::Graph::Size Result{ PrefixSize, 1 };
    if (!CallAddresses.empty()) {
      Result.W += textSize("then goes to one of: ").W;
      appendSize(Result, call(CallAddresses));
    } else {
      Result.W += textSize("known targets include: ").W;
    }

    size_t ValidTargetCount = 0;
    for (const auto &[Target, _] : Targets)
      if (Target.isValid())
        ++ValidTargetCount;
    revng_assert(ValidTargetCount != 0);

    for (size_t Counter = 0; const auto &[Target, _] : Targets) {
      if (Target.isValid()) {
        yield::Graph::Size Link{ PrefixSize, 1 };
        Link.W += textSize("- ").W + targetLinkWidth(Target);
        appendSize(Result, Link);
      }
    }

    if (HasUnknownTargets == true) {
      yield::Graph::Size AndMore{ PrefixSize, 1 };
      AndMore.W += textSize("and more").W;
      appendSize(Result, AndMore);
    }

    return Result;
  }

protected:
  size_t targetLinkWidth(const MetaAddress &Target) {
    return linkSize(Target, Function, Binary, 0, BasicBlock.End).W;
  }

  yield::Graph::Size
  call(const llvm::SmallVector<MetaAddress, 4> &CallAddresses) {
    revng_assert(!CallAddresses.empty());

    yield::Graph::Size Result{ PrefixSize + textSize("calls ").W, 1 };
    for (const MetaAddress &Address : CallAddresses)
      Result.W += targetLinkWidth(Address);

    Result.W += textSize(", ").W * (CallAddresses.size() - 1);

    return Result;
  }
};

static yield::Graph::Size
targetFooterSize(const yield::BasicBlock &BasicBlock,
                 const yield::Function &Function,
                 const model::Binary &Binary,
                 const yield::cfg::Configuration &Configuration) {
  static const efa::ParsedSuccessor UnknownTarget{
    .NextInstructionAddress = MetaAddress::invalid(),
    .OptionalCallAddress = MetaAddress::invalid()
  };

  bool HasUnknown = false;
  ParsedSuccessorVector Targets;
  for (const auto &Edge : BasicBlock.Successors) {
    auto TargetPair = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (TargetPair.NextInstructionAddress.isValid()
        || TargetPair.OptionalCallAddress.isValid()) {
      Targets.emplace_back(std::move(TargetPair));
    } else {
      HasUnknown = true;
    }
  }

  namespace A = model::Architecture;
  auto CommentIndicator = A::getAssemblyCommentIndicator(Binary.Architecture);

  yield::Graph::Size Result = fontSize(textSize(CommentIndicator),
                                       Configuration.InstructionFontSize,
                                       Configuration);
  auto LocalAppendHelper = [&](yield::Graph::Size &&Size) {
    return appendSize(Result,
                      fontSize(std::move(Size),
                               Configuration.CommentFontSize,
                               Configuration.InstructionFontSize,
                               Configuration));
  };

  yield::Graph::Dimension PrefixSize = CommentIndicator.size() + 1;
  TargetSizeHelper Helper(BasicBlock, Function, Binary, PrefixSize);
  if (Targets.size() == 0) {
    revng_assert(HasUnknown == true, "A basic block with no successors.");
    return LocalAppendHelper(Helper.singleTarget(UnknownTarget));
  } else if (Targets.size() == 1) {
    const auto &First = Targets.front();
    if (HasUnknown == false)
      return LocalAppendHelper(Helper.singleTarget(First));
    else
      return LocalAppendHelper(Helper.twoTargets(First, UnknownTarget));
  } else if (Targets.size() == 2 && HasUnknown == false) {
    const auto &First = Targets.front();
    const auto &Last = Targets.back();
    return LocalAppendHelper(Helper.twoTargets(First, Last));
  } else {
    return LocalAppendHelper(Helper.multipleTargets(Targets, HasUnknown));
  }
}

static yield::Graph::Size
basicBlockSize(const yield::BasicBlock &BasicBlock,
               const yield::Function &Function,
               const model::Binary &Binary,
               const yield::cfg::Configuration &Configuration) {
  // Account for the size of the label
  namespace A = model::Architecture;
  auto LabelIndicator = A::getAssemblyLabelIndicator(Binary.Architecture);
  yield::Graph::Size Result = fontSize(linkSize(BasicBlock.Start,
                                                Function,
                                                Binary,
                                                LabelIndicator.size()),
                                       Configuration.LabelFontSize,
                                       Configuration);

  namespace A = model::Architecture;
  auto CommentIndicator = A::getAssemblyCommentIndicator(Binary.Architecture);

  // Account for the sizes of each instruction.
  auto FromIterator = BasicBlock.Instructions.begin();
  auto ToIterator = std::prev(BasicBlock.Instructions.end());
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
                             BasicBlock.HasDelaySlot));
  revng_assert(ToIterator == BasicBlock.Instructions.end());

  // Account for the footer size.
  // The footer is where target information is printed in the vertical layout.
  appendSize(Result,
             targetFooterSize(BasicBlock, Function, Binary, Configuration));

  return Result;
}

void yield::cfg::calculateNodeSizes(Graph &Graph,
                                    const yield::Function &Function,
                                    const model::Binary &Binary,
                                    const Configuration &Configuration) {
  for (auto *Node : Graph.nodes()) {
    revng_assert(Node != nullptr);

    if (Node->Address.isValid()) {
      // A normal node.
      if (auto Iterator = Function.ControlFlowGraph.find(Node->Address);
          Iterator != Function.ControlFlowGraph.end()) {
        Node->Size = basicBlockSize(*Iterator, Function, Binary, Configuration);
      } else if (auto Iterator = Binary.Functions.find(Node->Address);
                 Iterator != Binary.Functions.end()) {
        Node->Size = singleLineSize(Iterator->name().str(),
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
