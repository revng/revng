/// \file HTML.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/Assembly/Assembly.h"
#include "revng/Yield/HTML.h"

namespace yield::html {

namespace tags {

static constexpr auto NormalBasicBlockLabel = "basic-block-label";
static constexpr auto HiddenBasicBlockLabel = "hidden-basic-block-label";
static constexpr auto GraphOnlyBasicBlockLabel = "graph-only-basic-block-label";

static constexpr auto Function = "function";
static constexpr auto BasicBlock = "basic-block";

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

static std::string linkAddress(const MetaAddress &Address) {
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
  return "basic_block_at_" + linkAddress(Address);
}

static std::string instructionID(const MetaAddress &Address) {
  return "instruction_at_" + linkAddress(Address);
}

static std::string link(const MetaAddress &Target,
                        const efa::FunctionMetadata &Metadata,
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
                         linkAddress(Target) + ".html#" + basicBlockID(Target),
                         std::move(FinalName));
  } else if (auto Iterator = Metadata.ControlFlowGraph.find(Target);
             Iterator != Metadata.ControlFlowGraph.end()) {
    // The target is a basic block
    std::string FinalName = CustomName.str();
    if (FinalName.empty()) {
      // Deduce the function current basic block belongs to based on
      // the received metadata and use that as a part of its name
      std::string FunctionName = "unknown_function_";
      if (auto FunctionIterator = Binary.Functions.find(Metadata.Entry);
          FunctionIterator != Binary.Functions.end()) {
        FunctionName = FunctionIterator->name().str().str() + "_";
      }

      std::string BlockOwnerName = llvm::formatv(templates::Span,
                                                 tags::BasicBlockOwner,
                                                 std::move(FunctionName));
      std::string BlockName = "basic_block_at_" + linkAddress(Target);
      FinalName = std::move(BlockOwnerName) + std::move(BlockName);
    }
    return llvm::formatv(templates::Link,
                         tags::BasicBlockLink,
                         linkAddress(Metadata.Entry) + ".html#"
                           + basicBlockID(Target),
                         std::move(FinalName));
  } else if (Target.isValid()) {
    // The target is an instruction
    std::string FinalName = CustomName.str();
    if (FinalName.empty())
      FinalName = Target.toString();
    return llvm::formatv(templates::Link,
                         tags::InstructionLink,
                         linkAddress(Metadata.Entry) + ".html#"
                           + instructionID(Target),
                         std::move(FinalName));
  } else {
    // The target is impossible to deduce, it's an indirect call or the like.
    return "";
  }
}

static std::string commentIndicator(const assembly::BasicBlock &BasicBlock) {
  return llvm::formatv(templates::Span,
                       tags::CommentIndicator,
                       BasicBlock.CommentIndicator);
}

static std::string labelIndicator(const assembly::BasicBlock &BasicBlock) {
  return llvm::formatv(templates::Span,
                       tags::LabelIndicator,
                       BasicBlock.LabelIndicator);
}

static std::string label(const assembly::BasicBlock &BasicBlock,
                         const efa::FunctionMetadata &Metadata,
                         const model::Binary &Binary) {
  std::string_view LabelTag = tags::NormalBasicBlockLabel;
  if (BasicBlock.CanBeMergedWithPredecessor)
    LabelTag = tags::HiddenBasicBlockLabel;
  else if (BasicBlock.IsAFallthroughTarget)
    LabelTag = tags::GraphOnlyBasicBlockLabel;

  std::string Link = link(BasicBlock.Address, Metadata, Binary);
  return llvm::formatv(templates::SimpleDiv,
                       std::move(LabelTag),
                       std::move(Link += labelIndicator(BasicBlock)));
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
                               const assembly::BasicBlock &BasicBlock,
                               std::string &&Body,
                               size_t Offset,
                               bool NeedsNewLine) {
  std::string Result = commentIndicator(BasicBlock) + whitespace(1)
                       + std::move(Body);
  Result = llvm::formatv(Template, Tag, std::move(Result));
  return (NeedsNewLine ? newLine() : "") + whitespace(Offset)
         + std::move(Result);
}

static std::string comment(const assembly::BasicBlock &BasicBlock,
                           std::string &&Body,
                           size_t Offset = 0,
                           bool NeedsNewLine = false) {
  return commentImpl(templates::Span,
                     tags::Comment,
                     BasicBlock,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string error(const assembly::BasicBlock &BasicBlock,
                         std::string &&Body,
                         size_t Offset = 0,
                         bool NeedsNewLine = false) {
  return commentImpl(templates::Span,
                     tags::Error,
                     BasicBlock,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string blockComment(llvm::StringRef Tag,
                                const assembly::BasicBlock &BasicBlock,
                                std::string &&Body,
                                size_t Offset = 0,
                                bool NeedsNewLine = false) {
  return commentImpl(templates::SimpleDiv,
                     Tag,
                     BasicBlock,
                     std::move(Body),
                     Offset,
                     NeedsNewLine);
}

static std::string bytes(const assembly::BasicBlock &BasicBlock,
                         const assembly::Instruction::ByteContainer &Bytes,
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
  return blockComment(tags::InstructionBytes, BasicBlock, std::move(Result));
}

static size_t countTargets(const SortedVector<MetaAddress> &Targets) {
  return Targets.size() - Targets.count(MetaAddress::invalid());
}

static bool areTargetsAdjacent(const MetaAddress &CurrentAddress,
                               const MetaAddress &TargetAddress,
                               const efa::FunctionMetadata &Metadata) {
  auto CurrentIterator = Metadata.ControlFlowGraph.find(CurrentAddress);
  if (CurrentIterator == Metadata.ControlFlowGraph.end())
    return false;

  return CurrentIterator->End == TargetAddress;
}

static std::string targetLink(const MetaAddress &Target,
                              const assembly::BasicBlock &BasicBlock,
                              const efa::FunctionMetadata &Metadata,
                              const model::Binary &Binary) {
  if (areTargetsAdjacent(BasicBlock.Address, Target, Metadata))
    return llvm::formatv(templates::Span,
                         tags::InstructionTarget,
                         link(Target,
                              Metadata,
                              Binary,
                              "the next instruction"));
  else
    return llvm::formatv(templates::Span,
                         tags::InstructionTarget,
                         link(Target, Metadata, Binary));
}

static std::string targets(const assembly::BasicBlock &BasicBlock,
                           const efa::FunctionMetadata &Metadata,
                           const model::Binary &Binary,
                           size_t TailOffset) {
  size_t TargetCount = countTargets(BasicBlock.Targets);
  if (TargetCount == 0)
    return ""; // We know nothing about the targets.

  std::string Result;
  if (BasicBlock.Targets.size() == 1) {
    // There's only a single known target. This is probably a direct call.
    revng_assert(BasicBlock.Targets.begin()->isValid());
    Result += comment(BasicBlock,
                      "always goes to "
                        + targetLink(*BasicBlock.Targets.begin(),
                                     BasicBlock,
                                     Metadata,
                                     Binary));
  } else {
    Result += comment(BasicBlock, "known targets include: ");
    bool HasInvalidTargets = false;
    for (size_t TargetCounter = 0; const auto &Target : BasicBlock.Targets) {
      if (Target.isValid()) {
        std::string Link = targetLink(Target, BasicBlock, Metadata, Binary);
        if (BasicBlock.Targets.size() == TargetCount
            && TargetCounter != TargetCount - 1) {
          Link = std::move(Link) + ",";
        }
        ++TargetCounter;

        Result += comment(BasicBlock, "- " + std::move(Link), TailOffset, true);
      } else {
        HasInvalidTargets = true;
      }
    }

    if (HasInvalidTargets)
      Result += comment(BasicBlock, "and more", TailOffset, true);
  }

  return llvm::formatv(templates::Span,
                       tags::InstructionTargets,
                       std::move(Result));
}

static std::string tagTypeAsString(assembly::Instruction::TagType Type) {
  switch (Type) {
  case assembly::Instruction::TagType::Immediate:
    return tags::ImmediateValue;
  case assembly::Instruction::TagType::Memory:
    return tags::MemoryOperand;
  case assembly::Instruction::TagType::Mnemonic:
    return tags::InstructionMnemonic;
  case assembly::Instruction::TagType::MnemonicPrefix:
    return tags::InstructionMnemonicPrefix;
  case assembly::Instruction::TagType::MnemonicSuffix:
    return tags::InstructionMnemonicSuffix;
  case assembly::Instruction::TagType::Register:
    return tags::Register;
  case assembly::Instruction::TagType::Whitespace:
    return tags::Whitespace;
  case assembly::Instruction::TagType::Invalid:
  default:
    revng_abort("Unknown tag type");
  }
}

using LeafContainer = llvm::SmallVector<llvm::SmallVector<size_t, 4>, 16>;
static std::string tag(size_t Index,
                       const LeafContainer &Leaves,
                       const assembly::Instruction &Instruction) {
  revng_assert(Index < Instruction.Tags.size());
  const assembly::Instruction::Tag &Tag = Instruction.Tags[Index];
  llvm::StringRef TextView = Instruction.Text;

  revng_assert(Index < Leaves.size());
  const auto &AdjacentLeaves = Leaves[Index];

  std::string Result;
  size_t CurrentIndex = Tag.FromIndex;
  for (const auto &LeafIndex : llvm::reverse(AdjacentLeaves)) {
    revng_assert(LeafIndex < Instruction.Tags.size());
    const auto &LeafTag = Instruction.Tags[LeafIndex];

    revng_assert(CurrentIndex <= LeafTag.FromIndex);
    if (CurrentIndex < LeafTag.FromIndex)
      Result += TextView.slice(CurrentIndex, LeafTag.FromIndex);
    Result += tag(LeafIndex, Leaves, Instruction);
    CurrentIndex = LeafTag.ToIndex;
  }
  revng_assert(CurrentIndex <= Tag.ToIndex);
  if (CurrentIndex < Tag.ToIndex)
    Result += TextView.slice(CurrentIndex, Tag.ToIndex);

  std::string TagStr = tagTypeAsString(Tag.Type);

  if (Tag.Type != assembly::Instruction::TagType::Mnemonic)
    return llvm::formatv(templates::Span, std::move(TagStr), std::move(Result));
  else
    return llvm::formatv(templates::Link,
                         std::move(TagStr),
                         "#" + instructionID(Instruction.Address),
                         std::move(Result));
}

static std::string taggedText(const assembly::Instruction &Instruction) {
  revng_assert(!Instruction.Tags.empty(),
               "Tagless instructions are not supported");

  // Convert the tag list into a tree to simplify working with nested tags.
  llvm::SmallVector<size_t> RootIndices;
  LeafContainer Leaves(Instruction.Tags.size());
  for (size_t Index = Instruction.Tags.size() - 1; Index > 0; --Index) {
    const auto &CurrentTag = Instruction.Tags[Index];

    bool DependencyDetected = false;
    for (size_t PrevIndex = Index - 1; PrevIndex != size_t(-1); --PrevIndex) {
      const auto &PreviousTag = Instruction.Tags[PrevIndex];
      if (CurrentTag.FromIndex >= PreviousTag.FromIndex
          && CurrentTag.ToIndex <= PreviousTag.ToIndex) {
        // Current tag is inside the previous one.
        // Add an edge corresponding to this relation.
        if (!DependencyDetected)
          Leaves[PrevIndex].emplace_back(Index);
        DependencyDetected = true;
      } else if (CurrentTag.FromIndex >= PreviousTag.ToIndex
                 && CurrentTag.ToIndex >= PreviousTag.ToIndex) {
        // Current tag is after (and outside) the previous one.
        // Do nothing.
      } else if (CurrentTag.FromIndex <= PreviousTag.FromIndex
                 && CurrentTag.ToIndex <= PreviousTag.FromIndex) {
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
  llvm::StringRef TextView = Instruction.Text;
  for (size_t RootIndex : llvm::reverse(RootIndices)) {
    revng_assert(RootIndex < Instruction.Tags.size());
    const auto &RootTag = Instruction.Tags[RootIndex];

    if (CurrentIndex < RootTag.FromIndex)
      Result += llvm::formatv(templates::Span,
                              tags::Untagged,
                              TextView.slice(CurrentIndex, RootTag.FromIndex));
    Result += tag(RootIndex, Leaves, Instruction);
    CurrentIndex = RootTag.ToIndex;
  }
  revng_assert(CurrentIndex <= TextView.size());
  if (CurrentIndex < TextView.size())
    Result += llvm::formatv(templates::Span,
                            tags::Untagged,
                            TextView.substr(CurrentIndex));

  return Result;
}

static std::string instruction(const assembly::Instruction &Instruction,
                               bool IsInDelayedSlot,
                               bool NeedsToPrintTargets,
                               const assembly::BasicBlock &BasicBlock,
                               const efa::FunctionMetadata &Metadata,
                               const model::Binary &Binary) {
  // MetaAddress of the instruction.
  std::string Result = blockComment(tags::InstructionAddress,
                                    BasicBlock,
                                    Instruction.Address.toString());

  // Raw bytes of the instruction.
  //
  // \note the instructions disassembler failed on are limited to 16 bytes.
  if (Instruction.Error == "MCDisassembler failed")
    Result += bytes(BasicBlock, Instruction.Bytes, 16);
  else
    Result += bytes(BasicBlock, Instruction.Bytes);

  // LLVM's Opcode of the instruction.
  if (Instruction.Opcode.has_value())
    Result += llvm::formatv(templates::Span,
                            tags::InstructionOpcode,
                            Instruction.Opcode.value());

  // Tagged instruction body.
  Result += taggedText(Instruction);
  size_t Tail = Instruction.Text.size() + 1;

  // The original comment if present.
  bool HasTailComments = false;
  if (!Instruction.Comment.empty()) {
    Result += comment(BasicBlock, std::string(Instruction.Comment), 1);
    HasTailComments = true;
  }

  // Delayed slot notice if applicable.
  if (IsInDelayedSlot) {
    if (HasTailComments == true)
      Result += comment(BasicBlock, "delayed", Tail, true);
    else
      Result += comment(BasicBlock, "delayed", 1);
    HasTailComments = true;
  }

  // An error message if present.
  if (!Instruction.Error.empty()) {
    if (HasTailComments == true)
      Result += error(BasicBlock, "Error: " + Instruction.Error, Tail, true);
    else
      Result += error(BasicBlock, "Error: " + Instruction.Error, 1);
    HasTailComments = true;
  }

  // The list of targets if needed
  if (NeedsToPrintTargets) {
    if (HasTailComments == false) {
      Result += whitespace(1);
    } else if (countTargets(BasicBlock.Targets) != 0) {
      Result += newLine() + whitespace(Tail);
    }

    Result += targets(BasicBlock, Metadata, Binary, Tail);
  }

  return llvm::formatv(templates::BlockDiv,
                       tags::Instruction,
                       instructionID(Instruction.Address),
                       std::move(Result));
}

static std::string basicBlock(const assembly::BasicBlock &BasicBlock,
                              const efa::FunctionMetadata &Metadata,
                              const model::Binary &Binary) {
  std::string Result;

  llvm::SmallVector<MetaAddress, 2> DelayedList;
  bool IsNextInstructionDelayed = false;
  for (const auto &Instruction : BasicBlock.Instructions) {
    if (IsNextInstructionDelayed)
      DelayedList.emplace_back(Instruction.Address);
    IsNextInstructionDelayed = Instruction.HasDelayedSlot;
  }

  MetaAddress LastNotDelayedInstruction = MetaAddress::invalid();
  for (const auto &Instruction : llvm::reverse(BasicBlock.Instructions)) {
    if (!llvm::is_contained(DelayedList, Instruction.Address)) {
      LastNotDelayedInstruction = Instruction.Address;
      break;
    }
  }

  Result += label(BasicBlock, Metadata, Binary);
  for (const auto &Instruction : BasicBlock.Instructions) {
    Result += instruction(Instruction,
                          llvm::is_contained(DelayedList, Instruction.Address),
                          LastNotDelayedInstruction == Instruction.Address,
                          BasicBlock,
                          Metadata,
                          Binary);
  }

  if (IsNextInstructionDelayed == true) {
    std::string DelayedError = "Error: Last instruction has a delayed slot.";
    Result += error(BasicBlock, std::move(DelayedError), 2, true);
  }

  return llvm::formatv(templates::BlockDiv,
                       tags::BasicBlock,
                       basicBlockID(BasicBlock.Address),
                       std::move(Result));
}

std::string assembly(const assembly::BasicBlock &BasicBlock,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary) {
  return basicBlock(BasicBlock, Metadata, Binary);
}
std::string assembly(const assembly::Function &Function,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.BasicBlocks)
    Result += basicBlock(BasicBlock, Metadata, Binary);

  return Result;
}

} // namespace yield::html
