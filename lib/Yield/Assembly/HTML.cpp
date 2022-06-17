/// \file HTML.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/HTML.h"

namespace tags {

static constexpr auto Function = "function";
static constexpr auto LabeledBlock = "labeled-block";
static constexpr auto BasicBlock = "basic-block";
static constexpr auto FunctionLabel = "function-label";
static constexpr auto BasicBlockLabel = "basic-block-label";

static constexpr auto Instruction = "instruction";
static constexpr auto InstructionAddress = "instruction-address";
static constexpr auto InstructionBytes = "instruction-bytes";

static constexpr auto InstructionMnemonic = "mnemonic";
static constexpr auto InstructionMnemonicPrefix = "mnemonic-prefix";
static constexpr auto InstructionMnemonicSuffix = "mnemonic-suffix";
static constexpr auto InstructionOpcode = "instruction-opcode";

static constexpr auto Comment = "comment";
static constexpr auto Error = "error";
static constexpr auto ImmediateValue = "immediate-value";
static constexpr auto MemoryOperand = "memory-operand";
static constexpr auto Register = "register";

static constexpr auto CommentIndicator = "comment-indicator";
static constexpr auto LabelIndicator = "label-indicator";

static constexpr auto FunctionLink = "function-link";
static constexpr auto BasicBlockLink = "basic-block-link";
static constexpr auto InstructionLink = "instruction-link";

static constexpr auto BasicBlockOwner = "basic-block-owner";

static constexpr auto InstructionTarget = "instruction-target";
static constexpr auto InstructionTargets = "instruction-targets";

static constexpr auto Whitespace = "whitespace";
static constexpr auto Untagged = "untagged";

} // namespace tags

namespace templates {

static constexpr auto BlockDiv = R"(<div class="{0}" id="{1}">{2}</div>)";
static constexpr auto SimpleDiv = R"(<div class="{0}">{1}</div>)";

static constexpr auto Link = R"(<a class="{0}" href="{1}">{2}</a>)";
static constexpr auto Span = R"(<span class="{0}">{1}</span>)";

} // namespace templates

static std::string address(const MetaAddress &Address) {
  std::string Result = Address.toString();

  constexpr std::array ForbiddenCharacters = { ' ', ':', '!', '#',  '?',
                                               '<', '>', '/', '\\', '{',
                                               '}', '[', ']' };

  for (char &Character : Result)
    if (llvm::find(ForbiddenCharacters, Character) != ForbiddenCharacters.end())
      Character = '_';

  return Result;
}

static std::string basicBlockID(const MetaAddress &Address) {
  return "basic_block_at_" + address(Address);
}

static std::string instructionID(const MetaAddress &Address) {
  return "instruction_at_" + address(Address);
}

static std::string link(const MetaAddress &Target,
                        const yield::Function &Function,
                        const model::Binary &Binary,
                        llvm::StringRef CustomName = "") {
  if (auto Iterator = Binary.Functions.find(Target);
      Iterator != Binary.Functions.end()) {
    // The target is a function
    std::string FinalName = CustomName.str();
    if (FinalName.empty())
      FinalName = Iterator->name().str().str();
    return llvm::formatv(templates::Link,
                         tags::FunctionLink,
                         address(Target) + ".html#" + basicBlockID(Target),
                         std::move(FinalName));
  } else if (auto Iterator = Function.ControlFlowGraph.find(Target);
             Iterator != Function.ControlFlowGraph.end()) {
    // The target is a basic block
    std::string FinalName = CustomName.str();
    if (FinalName.empty()) {
      auto FunctionIterator = Binary.Functions.find(Function.Entry);
      revng_assert(FunctionIterator != Binary.Functions.end());

      std::string FunctionPrefix = FunctionIterator->name().str().str() + "_";
      std::string BlockOwnerName = llvm::formatv(templates::Span,
                                                 tags::BasicBlockOwner,
                                                 std::move(FunctionPrefix));

      std::string BlockName = "basic_block_at_" + address(Target);
      FinalName = std::move(BlockOwnerName) + std::move(BlockName);
    }
    return llvm::formatv(templates::Link,
                         tags::BasicBlockLink,
                         address(Function.Entry) + ".html#"
                           + basicBlockID(Target),
                         std::move(FinalName));
  } else if (Target.isValid()) {
    // The target is an instruction
    std::string FinalName = CustomName.str();
    if (FinalName.empty())
      FinalName = "instruction_at_" + Target.toString();
    return llvm::formatv(templates::Link,
                         tags::InstructionLink,
                         address(Function.Entry) + ".html#"
                           + instructionID(Target),
                         std::move(FinalName));
  } else {
    // The target is impossible to deduce, it's an indirect call or the like.
    return "unknown_target";
  }
}

static std::string commentIndicator(model::Architecture::Values Architecture) {
  namespace Arch = model::Architecture;
  return llvm::formatv(templates::Span,
                       tags::CommentIndicator,
                       Arch::getAssemblyCommentIndicator(Architecture));
}

static std::string labelIndicator(model::Architecture::Values Architecture) {
  namespace Arch = model::Architecture;
  return llvm::formatv(templates::Span,
                       tags::LabelIndicator,
                       Arch::getAssemblyLabelIndicator(Architecture));
}

static std::string label(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary) {
  std::string Link = link(BasicBlock.Start, Function, Binary);
  return llvm::formatv(templates::SimpleDiv,
                       Function.Entry == BasicBlock.Start ?
                         tags::FunctionLabel :
                         tags::BasicBlockLabel,
                       std::move(Link += labelIndicator(Binary.Architecture)));
}

static std::string whitespace(size_t Count) {
  if (Count == 0)
    return "";

  std::string Result;
  for (size_t Counter = 0; Counter < Count; ++Counter)
    Result += "&nbsp;";
  return llvm::formatv(templates::Span, tags::Whitespace, std::move(Result));
}

static std::string newLine() {
  return llvm::formatv(templates::Span, tags::Whitespace, "<br />");
}

static std::string commentImpl(const char *Template,
                               llvm::StringRef Tag,
                               const model::Binary &Binary,
                               std::string &&Body,
                               size_t Offset,
                               bool NeedsNewLine) {
  std::string Result = commentIndicator(Binary.Architecture) + whitespace(1)
                       + std::move(Body);
  Result = llvm::formatv(Template, Tag, std::move(Result));
  return (NeedsNewLine ? newLine() : "") + whitespace(Offset)
         + std::move(Result);
}

static std::string comment(const model::Binary &Binary,
                           std::string &&Body,
                           size_t Offset = 0,
                           bool NeedsNewLine = false) {
  return commentImpl(templates::Span,
                     tags::Comment,
                     Binary,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string error(const model::Binary &Binary,
                         std::string &&Body,
                         size_t Offset = 0,
                         bool NeedsNewLine = false) {
  return commentImpl(templates::Span,
                     tags::Error,
                     Binary,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string blockComment(llvm::StringRef Tag,
                                const model::Binary &Binary,
                                std::string &&Body,
                                size_t Offset = 0,
                                bool NeedsNewLine = false) {
  return commentImpl(templates::SimpleDiv,
                     Tag,
                     Binary,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string bytes(const model::Binary &Binary,
                         const yield::ByteContainer &Bytes,
                         size_t Limit = std::numeric_limits<size_t>::max()) {
  std::string Result;
  llvm::raw_string_ostream FormattingStream(Result);

  bool NeedsSpace = false;
  for (const auto &Byte : llvm::ArrayRef<uint8_t>{ Bytes }.take_front(Limit)) {
    if (NeedsSpace)
      FormattingStream << "&nbsp;";
    else
      NeedsSpace = true;

    llvm::write_hex(FormattingStream, Byte, llvm::HexPrintStyle::Upper, 2);
  }

  if (Bytes.size() > Limit)
    FormattingStream << "&nbsp;[...]";

  FormattingStream.flush();
  return blockComment(tags::InstructionBytes, Binary, std::move(Result));
}

using ParsedSuccessorVector = llvm::SmallVector<efa::ParsedSuccessor, 4>;

template<bool ShouldUseVerticalLayout = false>
class TargetPrintingHelper {
private:
  const yield::BasicBlock &BasicBlock;
  const yield::Function &Function;
  const model::Binary &Binary;
  size_t TailOffset;

public:
  TargetPrintingHelper(const yield::BasicBlock &BasicBlock,
                       const yield::Function &Function,
                       const model::Binary &Binary,
                       size_t TailOffset) :
    BasicBlock(BasicBlock),
    Function(Function),
    Binary(Binary),
    TailOffset(TailOffset) {}

  std::string singleTarget(const efa::ParsedSuccessor &Target) {
    const auto &[NextAddress, CallAddress] = Target;
    if (NextAddress.isValid()) {
      if (CallAddress.isValid()) {
        // Both are valid, it's a normal call.
        return call({ CallAddress })
               + comment(Binary,
                         "then goes to " + targetLink(NextAddress),
                         TailOffset,
                         true);
      } else {
        // Only jump address is valid, it's a normal jump.
        if (NextAddress == BasicBlock.End) {
          // The only target is the next instruction.
          // Don't emit these in horizontal layout.
          if constexpr (ShouldUseVerticalLayout == false)
            return "";
        }

        return comment(Binary, "always goes to " + targetLink(NextAddress));
      }
    } else {
      if (CallAddress.isValid()) {
        // Only call address is valid, it's a no-return call.
        return call({ CallAddress })
               + comment(Binary, "and does not return", TailOffset, true);
      } else {
        // Neither is valid, nothing is known about the target.
        return "";
      }
    }
  }

  std::string twoTargets(const efa::ParsedSuccessor &First,
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
      // One of the targets is the next instruction.
      std::string Result = comment(Binary,
                                   "if taken, goes to "
                                     + targetLink(FirstTarget) + ",");
      Result += comment(Binary,
                        "otherwise, goes to " + targetLink(SecondTarget),
                        TailOffset,
                        true);
      return Result;
    } else {
      return multipleTargets({ First, Second });
    }
  }

  std::string multipleTargets(const ParsedSuccessorVector &Targets,
                              bool HasUnknownTargets = false) {
    llvm::SmallVector<MetaAddress, 4> CallAddresses;
    for (const auto &[_, Target] : Targets)
      if (Target.isValid())
        CallAddresses.emplace_back(Target);

    std::string Result = !CallAddresses.empty() ? call(CallAddresses) : "";
    if (!Result.empty())
      Result += comment(Binary, "then goes to one of: ", TailOffset, true);
    else
      Result += comment(Binary, "known targets include: ");

    size_t ValidTargetCount = 0;
    for (const auto &[Target, _] : Targets)
      if (Target.isValid())
        ++ValidTargetCount;
    revng_assert(ValidTargetCount != 0);

    for (size_t Counter = 0; const auto &[Target, _] : Targets) {
      if (Target.isValid()) {
        std::string Link = targetLink(Target);
        if (++Counter < ValidTargetCount)
          Link += ",";
        Result += comment(Binary, "- " + std::move(Link), TailOffset, true);
      }
    }

    if (HasUnknownTargets == true)
      Result += comment(Binary, "and more", TailOffset, true);
    return Result;
  }

protected:
  std::string targetLink(const MetaAddress &Target) {
    if (Target.isInvalid())
      return "an unknown location";
    else if (Target == BasicBlock.End)
      return llvm::formatv(templates::Span,
                           tags::InstructionTarget,
                           link(Target,
                                Function,
                                Binary,
                                "the next instruction"));
    else
      return llvm::formatv(templates::Span,
                           tags::InstructionTarget,
                           link(Target, Function, Binary));
  }

  std::string call(const llvm::SmallVector<MetaAddress, 4> &CallAddresses) {
    revng_assert(!CallAddresses.empty());

    std::string Result = "calls ";
    for (size_t Counter = 0; const MetaAddress &Address : CallAddresses) {
      Result += targetLink(Address);
      if (++Counter != CallAddresses.size())
        Result += ", ";
    }

    return comment(Binary, std::move(Result));
  }
};

template<bool ShouldUseVerticalLayout = false>
static std::string targets(const yield::BasicBlock &BasicBlock,
                           const yield::Function &Function,
                           const model::Binary &Binary,
                           size_t TailOffset = 0) {
  static const efa::ParsedSuccessor UnknownTarget{
    .NextInstructionAddress = MetaAddress::invalid(),
    .OptionalCallAddress = MetaAddress::invalid()
  };

  bool HasUnknownTargets = false;
  ParsedSuccessorVector SuccessorTargets;
  for (const auto &Edge : BasicBlock.Successors) {
    auto TargetPair = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (TargetPair.NextInstructionAddress.isValid()
        || TargetPair.OptionalCallAddress.isValid()) {
      SuccessorTargets.emplace_back(std::move(TargetPair));
    } else {
      HasUnknownTargets = true;
    }
  }

  std::string Result;
  using LocalHelper = TargetPrintingHelper<ShouldUseVerticalLayout>;
  LocalHelper Helper(BasicBlock, Function, Binary, TailOffset);
  if (SuccessorTargets.size() == 0) {
    revng_assert(HasUnknownTargets == true,
                 "A basic block with no successors.");
    Result = Helper.singleTarget(UnknownTarget);
  } else if (SuccessorTargets.size() == 1) {
    if (HasUnknownTargets == false)
      Result = Helper.singleTarget(SuccessorTargets.front());
    else
      Result = Helper.twoTargets(SuccessorTargets.front(), UnknownTarget);
  } else if (SuccessorTargets.size() == 2 && HasUnknownTargets == false) {
    Result = Helper.twoTargets(SuccessorTargets.front(),
                               SuccessorTargets.back());
  } else {
    Result = Helper.multipleTargets(SuccessorTargets, HasUnknownTargets);
  }

  return llvm::formatv(templates::Span,
                       tags::InstructionTargets,
                       std::move(Result));
}

static std::string tagTypeAsString(yield::TagType::Values Type) {
  switch (Type) {
  case yield::TagType::Immediate:
    return tags::ImmediateValue;
  case yield::TagType::Memory:
    return tags::MemoryOperand;
  case yield::TagType::Mnemonic:
    return tags::InstructionMnemonic;
  case yield::TagType::MnemonicPrefix:
    return tags::InstructionMnemonicPrefix;
  case yield::TagType::MnemonicSuffix:
    return tags::InstructionMnemonicSuffix;
  case yield::TagType::Register:
    return tags::Register;
  case yield::TagType::Whitespace:
    return tags::Whitespace;
  case yield::TagType::Invalid:
  default:
    revng_abort("Unknown tag type");
  }
}

using LeafContainer = llvm::SmallVector<llvm::SmallVector<size_t, 4>, 16>;
static std::string tag(size_t Index,
                       const LeafContainer &Leaves,
                       const yield::Instruction &Instruction) {
  revng_assert(Index < Instruction.Tags.size());
  const yield::Tag &Tag = *std::next(Instruction.Tags.begin(), Index);
  llvm::StringRef TextView = Instruction.Disassembled;

  revng_assert(Index < Leaves.size());
  const auto &AdjacentLeaves = Leaves[Index];

  std::string Result;
  size_t CurrentIndex = Tag.From;
  for (const auto &LeafIndex : llvm::reverse(AdjacentLeaves)) {
    revng_assert(LeafIndex < Instruction.Tags.size());
    const auto &LeafTag = *std::next(Instruction.Tags.begin(), LeafIndex);

    revng_assert(CurrentIndex <= LeafTag.From);
    if (CurrentIndex < LeafTag.From)
      Result += TextView.slice(CurrentIndex, LeafTag.From);
    Result += tag(LeafIndex, Leaves, Instruction);
    CurrentIndex = LeafTag.To;
  }
  revng_assert(CurrentIndex <= Tag.To);
  if (CurrentIndex < Tag.To)
    Result += TextView.slice(CurrentIndex, Tag.To);

  std::string TagStr = tagTypeAsString(Tag.Type);

  if (Tag.Type != yield::TagType::Mnemonic)
    return llvm::formatv(templates::Span, std::move(TagStr), std::move(Result));
  else
    return llvm::formatv(templates::Link,
                         std::move(TagStr),
                         "#" + instructionID(Instruction.Address),
                         std::move(Result));
}

static std::string taggedText(const yield::Instruction &Instruction) {
  revng_assert(!Instruction.Tags.empty(),
               "Tagless instructions are not supported");

  // Convert the tag list into a tree to simplify working with nested tags.
  llvm::SmallVector<size_t> RootIndices;
  LeafContainer Leaves(Instruction.Tags.size());
  for (size_t Index = Instruction.Tags.size() - 1; Index > 0; --Index) {
    const auto &CurrentTag = *std::next(Instruction.Tags.begin(), Index);

    bool DependencyDetected = false;
    for (size_t PrevIndex = Index - 1; PrevIndex != size_t(-1); --PrevIndex) {
      const auto &PreviousTag = *std::next(Instruction.Tags.begin(), PrevIndex);
      if (CurrentTag.From >= PreviousTag.From
          && CurrentTag.To <= PreviousTag.To) {
        // Current tag is inside the previous one.
        // Add an edge corresponding to this relation.
        if (!DependencyDetected)
          Leaves[PrevIndex].emplace_back(Index);
        DependencyDetected = true;
      } else if (CurrentTag.From >= PreviousTag.To
                 && CurrentTag.To >= PreviousTag.To) {
        // Current tag is after (and outside) the previous one.
        // Do nothing.
      } else if (CurrentTag.From <= PreviousTag.From
                 && CurrentTag.To <= PreviousTag.From) {
        // Current tag is before (and outside) the previous one.
        revng_abort("Tag container must be sorted.");
      } else {
        revng_abort("Tags must not intersect");
      }
    }

    // The node is not depended on - add it as a root.
    if (!DependencyDetected)
      RootIndices.emplace_back(Index);
  }

  // Make sure there's at least one root.
  RootIndices.emplace_back(0);

  // Insert html-flavoured tags based on the tree.
  std::string Result;
  size_t CurrentIndex = 0;
  llvm::StringRef TextView = Instruction.Disassembled;
  for (size_t RootIndex : llvm::reverse(RootIndices)) {
    revng_assert(RootIndex < Instruction.Tags.size());
    const auto &RootTag = *std::next(Instruction.Tags.begin(), RootIndex);

    if (CurrentIndex < RootTag.From)
      Result += llvm::formatv(templates::Span,
                              tags::Untagged,
                              TextView.slice(CurrentIndex, RootTag.From));
    Result += tag(RootIndex, Leaves, Instruction);
    CurrentIndex = RootTag.To;
  }
  revng_assert(CurrentIndex <= TextView.size());
  if (CurrentIndex < TextView.size())
    Result += llvm::formatv(templates::Span,
                            tags::Untagged,
                            TextView.substr(CurrentIndex));

  return Result;
}

template<bool ShouldUseVerticalLayout>
static std::string instruction(const yield::Instruction &Instruction,
                               const yield::BasicBlock &BasicBlock,
                               const yield::Function &Function,
                               const model::Binary &Binary,
                               bool ShouldPrintTargets = false,
                               bool IsInDelayedSlot = false) {
  // MetaAddress of the instruction.
  std::string Result = blockComment(tags::InstructionAddress,
                                    Binary,
                                    Instruction.Address.toString());

  // Raw bytes of the instruction.
  //
  // \note the instructions disassembler failed on are limited to 16 bytes.
  if (Instruction.Error == "MCDisassembler failed")
    Result += bytes(Binary, Instruction.RawBytes, 16);
  else
    Result += bytes(Binary, Instruction.RawBytes);

  // LLVM's Opcode of the instruction.
  if (!Instruction.OpcodeIdentifier.empty())
    Result += blockComment(tags::InstructionOpcode,
                           Binary,
                           "llvm Opcode: " + Instruction.OpcodeIdentifier);

  // Error message (Vertical layout only).
  if (ShouldUseVerticalLayout == true && !Instruction.Error.empty())
    Result += error(Binary, "Error: " + Instruction.Error + "\n");

  // Tagged instruction body.
  Result += taggedText(Instruction);
  size_t Tail = Instruction.Disassembled.size() + 1;

  // The original comment if present.
  bool HasTailComments = false;
  if (!Instruction.Comment.empty()) {
    Result += comment(Binary, std::string(Instruction.Comment), 1);
    HasTailComments = true;
  }

  // Delayed slot notice if applicable.
  if (IsInDelayedSlot) {
    if (HasTailComments == true)
      Result += comment(Binary, "delayed", Tail, true);
    else
      Result += comment(Binary, "delayed", 1);
    HasTailComments = true;
  }

  // Horizontal layout only
  if constexpr (ShouldUseVerticalLayout == false) {
    // An error message.
    if (!Instruction.Error.empty()) {
      if (HasTailComments == true)
        Result += error(Binary, "Error: " + Instruction.Error, Tail, true);
      else
        Result += error(Binary, "Error: " + Instruction.Error, 1);
      HasTailComments = true;
    }

    // The list of targets.
    if (ShouldPrintTargets) {
      auto Targets = targets(BasicBlock, Function, Binary, Tail);
      if (!Targets.empty()) {
        if (HasTailComments == false)
          Result += whitespace(1) + std::move(Targets);
        else
          Result += newLine() + whitespace(Tail) + std::move(Targets);
      }
    }
  }

  return llvm::formatv(templates::BlockDiv,
                       tags::Instruction,
                       instructionID(Instruction.Address),
                       std::move(Result));
}

template<bool UseVerticalLayout>
static std::string basicBlock(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary,
                              bool HasLabel = true) {
  revng_assert(!BasicBlock.Instructions.empty());
  auto FromIterator = BasicBlock.Instructions.begin();
  auto ToIterator = std::prev(BasicBlock.Instructions.end());
  if (BasicBlock.HasDelaySlot) {
    revng_assert(BasicBlock.Instructions.size() > 1);
    --ToIterator;
  }

  std::string Result;
  for (auto Iterator = FromIterator; Iterator != ToIterator; ++Iterator) {
    // Print most of the instructions.
    Result += instruction<UseVerticalLayout>(*Iterator,
                                             BasicBlock,
                                             Function,
                                             Binary);
  }

  // Print the instruction with targets.
  Result += instruction<UseVerticalLayout>(*ToIterator++,
                                           BasicBlock,
                                           Function,
                                           Binary,
                                           true);

  if (BasicBlock.HasDelaySlot) {
    // Print the instruction in the delay slot, if applicable.
    Result += instruction<UseVerticalLayout>(*ToIterator++,
                                             BasicBlock,
                                             Function,
                                             Binary,
                                             false,
                                             true);
  }

  revng_assert(ToIterator == BasicBlock.Instructions.end());

  if (HasLabel)
    return llvm::formatv(templates::SimpleDiv,
                         tags::BasicBlock,
                         std::move(Result));
  else
    return llvm::formatv(templates::BlockDiv,
                         tags::BasicBlock,
                         basicBlockID(BasicBlock.Start),
                         std::move(Result));
}

template<bool ShouldMergeFallthroughTargets, bool UseVerticalTargetLayout>
static std::string labeledBlock(const yield::BasicBlock &FirstBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  std::string Result;
  Result += label(FirstBlock, Function, Binary);

  const yield::BasicBlock *LastBlock = nullptr;
  if constexpr (ShouldMergeFallthroughTargets == false) {
    Result += basicBlock<UseVerticalTargetLayout>(FirstBlock, Function, Binary);
    LastBlock = &FirstBlock;
  } else {
    auto BasicBlocks = yield::cfg::labeledBlock(FirstBlock, Function, Binary);
    if (BasicBlocks.empty())
      return "";

    bool IsFirst = true;
    for (const auto &BasicBlock : BasicBlocks) {
      Result += basicBlock<UseVerticalTargetLayout>(*BasicBlock,
                                                    Function,
                                                    Binary,
                                                    IsFirst);
      IsFirst = false;
    }

    LastBlock = BasicBlocks.back();
  }

  if constexpr (UseVerticalTargetLayout == true) {
    auto Targets = targets<UseVerticalTargetLayout>(*LastBlock,
                                                    Function,
                                                    Binary);
    if (!Targets.empty())
      Result += newLine() + std::move(Targets);
  }

  return llvm::formatv(templates::BlockDiv,
                       tags::LabeledBlock,
                       basicBlockID(FirstBlock.Start),
                       std::move(Result));
}

std::string yield::html::functionAssembly(const yield::Function &Function,
                                          const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.ControlFlowGraph)
    Result += labeledBlock<true, false>(BasicBlock, Function, Binary);

  return Result;
}

std::string yield::html::controlFlowNode(const MetaAddress &Address,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph.find(Address);
  revng_assert(Iterator != Function.ControlFlowGraph.end());

  auto Result = labeledBlock<false, true>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}
