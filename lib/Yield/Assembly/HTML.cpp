/// \file HTML.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/Concepts.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/HTML.h"
#include "revng/Yield/Support/PTML.h"

using yield::ptml::Tag;
namespace attributes = yield::ptml::attributes;
namespace ptmlScopes = yield::ptml::scopes;
namespace tags = yield::ptml::tags;

namespace tokenTypes {

static constexpr auto Label = "asm.label";
static constexpr auto LabelIndicator = "asm.label-indicator";
static constexpr auto CommentIndicator = "asm.comment-indicator";
static constexpr auto Mnemonic = "asm.mnemonic";
static constexpr auto MnemonicPrefix = "asm.mnemonic-prefix";
static constexpr auto MnemonicSuffix = "asm.mnemonic-suffix";
static constexpr auto ImmediateValue = "asm.immediate-value";
static constexpr auto MemoryOperand = "asm.memory-operand";
static constexpr auto Register = "asm.register";

} // namespace tokenTypes

namespace scopes {

static constexpr auto Function = "asm.function";
static constexpr auto BasicBlock = "asm.basic-block";
static constexpr auto Instruction = "asm.instruction";

} // namespace scopes

static std::string labelAddress(const MetaAddress &Address) {
  std::string Result = Address.toString();

  constexpr std::array ForbiddenCharacters = { ' ', ':', '!', '#',  '?',
                                               '<', '>', '/', '\\', '{',
                                               '}', '[', ']' };

  for (char &Character : Result)
    if (llvm::find(ForbiddenCharacters, Character) != ForbiddenCharacters.end())
      Character = '_';

  return Result;
}

static std::string link(const MetaAddress &TargetAddress,
                        const yield::Function &Function,
                        const model::Binary &Binary) {
  if (auto Iterator = Binary.Functions.find(TargetAddress);
      Iterator != Binary.Functions.end()) {
    return Iterator->name().str().str();
  } else if (auto Iterator = Function.ControlFlowGraph.find(TargetAddress);
             Iterator != Function.ControlFlowGraph.end()) {
    return "basic_block_at_" + labelAddress(TargetAddress);
  } else if (TargetAddress.isValid()) {
    return "instruction_at_" + labelAddress(TargetAddress);
  } else {
    return "unknown location";
  }
}

static std::string label(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary) {
  using model::Architecture::getAssemblyLabelIndicator;
  auto LabelIndicator = getAssemblyLabelIndicator(Binary.Architecture);
  Tag LabelTag(tags::Span, link(BasicBlock.Start, Function, Binary));
  LabelTag.add(attributes::token, tokenTypes::Label);

  if (auto Iterator = Binary.Functions.find(BasicBlock.Start);
      Iterator != Binary.Functions.end()) {
    LabelTag.add(attributes::modelEditPath,
                 "/Functions/" + Iterator->Entry.toString() + "/CustomName");
  }

  return LabelTag.serialize()
         + Tag(tags::Span, LabelIndicator)
             .add(attributes::token, tokenTypes::LabelIndicator)
             .serialize();
}

static std::string indent() {
  using namespace std::string_literals;
  return Tag(tags::Span, "  "s)
    .add(attributes::scope, ptmlScopes::Indentation)
    .serialize();
}

static std::string targetPath(const MetaAddress &Target,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  if (auto Iterator = Binary.Functions.find(Target);
      Iterator != Binary.Functions.end()) {
    // The target is a function
    return "/function/" + Iterator->stringKey();
  } else if (auto Iterator = Function.ControlFlowGraph.find(Target);
             Iterator != Function.ControlFlowGraph.end()) {
    // The target is a basic block
    return "/basic-block/" + Function.stringKey() + "/"
           + Iterator->stringKey();
  } else if (Target.isValid()) {
    for (auto BasicBlock : Function.ControlFlowGraph) {
      for (auto Instruction : BasicBlock.Instructions) {
        if (Instruction.Address == Target) {
          return "/instruction/" + Function.stringKey() + "/"
                 + BasicBlock.stringKey() + "/" + Target.toString();
        }
      }
    }
  }
  return "";
}

static std::set<std::string> targets(const yield::BasicBlock &BasicBlock,
                                     const yield::Function &Function,
                                     const model::Binary &Binary) {
  std::set<std::string> Result;
  for (const auto &Edge : BasicBlock.Successors) {
    auto [Next, Call] = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (Next.has_value() && Next->isValid())
      if (auto Path = targetPath(*Next, Function, Binary); !Path.empty())
        Result.insert(std::move(Path));
    if (Call.has_value() && Call->isValid())
      if (auto Path = targetPath(*Call, Function, Binary); !Path.empty())
        Result.insert(std::move(Path));
  }

  return Result;
}

static std::string tagTypeAsString(const yield::TagType::Values &Type) {
  switch (Type) {
  case yield::TagType::Immediate:
    return tokenTypes::ImmediateValue;
  case yield::TagType::Memory:
    return tokenTypes::MemoryOperand;
  case yield::TagType::Mnemonic:
    return tokenTypes::Mnemonic;
  case yield::TagType::MnemonicPrefix:
    return tokenTypes::MnemonicPrefix;
  case yield::TagType::MnemonicSuffix:
    return tokenTypes::MnemonicSuffix;
  case yield::TagType::Register:
    return tokenTypes::Register;
  case yield::TagType::Whitespace:
  case yield::TagType::Invalid:
    return "";
  default:
    revng_abort("Unknown tag type");
  }
}

static std::string
tokenTag(llvm::StringRef Buffer, const yield::TagType::Values &Tag) {
  std::string TagStr = tagTypeAsString(Tag);
  if (!TagStr.empty()) {
    return ::Tag(tags::Span, Buffer).add(attributes::token, TagStr).serialize();
  } else {
    return Buffer.str();
  }
}

static std::string taggedText(const yield::Instruction &Instruction) {
  revng_assert(!Instruction.Tags.empty(),
               "Tagless instructions are not supported");
  revng_assert(!Instruction.Disassembled.empty(),
               "Empty disassembled instructions are not supported");

  std::vector TagMap(Instruction.Disassembled.size(), yield::TagType::Invalid);
  for (yield::Tag Tag : Instruction.Tags) {
    revng_assert(Tag.Type != yield::TagType::Invalid,
                 "\"Invalid\" TagType encountered");
    for (size_t Index = Tag.From; Index < Tag.To; Index++) {
      TagMap[Index] = Tag.Type;
    }
  }

  std::string Result;
  std::string Buffer;
  yield::TagType::Values Tag = yield::TagType::Invalid;
  for (size_t Index = 0; Index < Instruction.Disassembled.size(); Index++) {
    if (Tag != TagMap[Index]) {
      Result += tokenTag(Buffer, Tag);
      Tag = TagMap[Index];
      Buffer.clear();
    }
    Buffer += Instruction.Disassembled[Index];
  }
  Result += tokenTag(Buffer, Tag);

  return Result;
}

static std::string bytes(const yield::ByteContainer &Bytes,
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
  return Result;
}

static std::string targetTypes(const MetaAddress &NextAddress,
                               const yield::BasicBlock &BasicBlock,
                               const yield::Function &Function,
                               const model::Binary &Binary) {
  std::string Result;

  revng_assert(NextAddress.isValid());
  for (const auto &Edge : BasicBlock.Successors) {
    auto [Next, Call] = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (Call.has_value()) {
      // The target is a call.
      if (Call->isValid()) {
        // The call is direct.
        if (Next.has_value()) {
          revng_assert(Next->isValid());
          Result += "call:";
          if (Next.value() == NextAddress)
            Result += "the next instruction:";
          else
            Result += link(*Next, Function, Binary) + ":";
          Result += link(*Call, Function, Binary) + ",";
        } else {
          Result += "noreturn-call:" + link(*Call, Function, Binary) + ",";
        }
      } else {
        // The call is indirect.
        if (Next.has_value()) {
          revng_assert(Next->isValid());
          Result += "indirect-call:";
          if (Next.value() == NextAddress)
            Result += "the next instruction,";
          else
            Result += link(*Next, Function, Binary) + ",";
        } else {
          Result += "indirect-noreturn-call,";
        }
      }
    } else {
      // The target is not a call.
      if (Next.has_value()) {
        // The target is a branch.
        if (Next->isValid()) {
          if (Next.value() == NextAddress)
            Result += "jump:the next instruction,";
          else
            Result += "jump:" + link(*Next, Function, Binary) + ",";
        } else {
          Result += "indirect-jump,";
        }
      } else {
        Result += "indirect-jump,";
      }
    }
  }

  Result.erase(Result.size() - 1);
  return Result;
}

static std::string metadata(const yield::Instruction &Instruction,
                            const yield::BasicBlock &BasicBlock,
                            const yield::Function &Function,
                            const model::Binary &Binary,
                            bool AddTargets = false) {
  std::string Result = scopes::Instruction;
  MetaAddress NextAddress = MetaAddress::invalid();

  static constexpr auto Separator = "|";
  Result += Separator;

  // The next instruction address.
  auto Iterator = BasicBlock.Instructions.find(Instruction.Address);
  revng_assert(Iterator != BasicBlock.Instructions.end());
  if (std::next(Iterator) != BasicBlock.Instructions.end()) {
    if (!BasicBlock.HasDelaySlot
        || std::next(Iterator, 2) != BasicBlock.Instructions.end()) {
      NextAddress = std::next(Iterator)->Address;
    }
  }
  if (NextAddress.isInvalid())
    NextAddress = BasicBlock.End;

  // Raw instruction bytes.
  Result += bytes(Instruction.RawBytes);
  Result += Separator;

  // Comment indicator.
  using model::Architecture::getAssemblyCommentIndicator;
  Result += getAssemblyCommentIndicator(Binary.Architecture);
  Result += Separator;

  // Extra information.
  Result += Instruction.OpcodeIdentifier + Separator;
  Result += Instruction.Error + Separator;
  Result += Instruction.Comment + Separator;

  // "Delayed" notice if applicable.
  if (BasicBlock.HasDelaySlot
      && std::next(Iterator) == BasicBlock.Instructions.end())
    Result += "delayed";
  Result += Separator;

  // Target type annotation.
  if (AddTargets)
    Result += targetTypes(NextAddress, BasicBlock, Function, Binary);

  return Result;
}

static std::string instruction(const yield::Instruction &Instruction,
                               const yield::BasicBlock &BasicBlock,
                               const yield::Function &Function,
                               const model::Binary &Binary,
                               bool AddTargets = false) {

  // Tagged instruction body.
  std::string Result = taggedText(Instruction);
  size_t Tail = Instruction.Disassembled.size() + 1;

  Tag Out = Tag(tags::Div, std::move(Result))
              .add(attributes::scope, scopes::Instruction)
              .add(attributes::locationDefinition,
                   "/instruction/" + Function.stringKey() + "/"
                     + BasicBlock.stringKey() + "/"
                     + Instruction.Address.toString())
              .add(attributes::htmlExclusiveMetadata,
                   metadata(Instruction,
                            BasicBlock,
                            Function,
                            Binary,
                            AddTargets));

  if (AddTargets) {
    auto Targets = targets(BasicBlock, Function, Binary);
    Out.add(attributes::locationReferences, Targets);
  }

  return Out.serialize() + '\n';
}

static std::string basicBlock(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Bin,
                              std::string &&Label) {
  revng_assert(!BasicBlock.Instructions.empty());
  auto Begin = BasicBlock.Instructions.begin();
  auto End = std::prev(BasicBlock.Instructions.end());
  if (BasicBlock.HasDelaySlot) {
    revng_assert(BasicBlock.Instructions.size() > 1);
    std::advance(End, -1);
  }

  std::string Result;
  for (auto Iterator = Begin; Iterator != End; ++Iterator)
    Result += indent() + instruction(*Iterator, BasicBlock, Function, Bin);
  Result += indent() + instruction(*(End++), BasicBlock, Function, Bin, true);
  if (BasicBlock.HasDelaySlot)
    Result += indent() + instruction(*(End++), BasicBlock, Function, Bin);

  if (!Label.empty())
    Label += '\n';

  return Tag(tags::Div, std::move(Label) += std::move(Result))
    .add(attributes::scope, scopes::BasicBlock)
    .add(attributes::locationDefinition,
         "/basic-block/" + Function.stringKey() + "/"
           + BasicBlock.stringKey())
    .serialize();
}

template<bool ShouldMergeFallthroughTargets>
static std::string labeledBlock(const yield::BasicBlock &FirstBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  std::string Result;
  std::string Label = label(FirstBlock, Function, Binary);

  if constexpr (ShouldMergeFallthroughTargets == false) {
    Result = basicBlock(FirstBlock, Function, Binary, std::move(Label));
  } else {
    auto BasicBlocks = yield::cfg::labeledBlock(FirstBlock, Function, Binary);
    if (BasicBlocks.empty())
      return "";

    bool IsFirst = true;
    for (const auto &BasicBlock : BasicBlocks) {
      std::string L = IsFirst ? std::move(Label) : "";
      Result += basicBlock(*BasicBlock, Function, Binary, std::move(L));
      IsFirst = false;
    }
  }

  return Result += '\n';
}

std::string yield::html::functionAssembly(const yield::Function &Function,
                                          const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.ControlFlowGraph) {
    Result += labeledBlock<true>(BasicBlock, Function, Binary);
  }

  return ::Tag(tags::Div, Result)
    .add(attributes::scope, scopes::Function)
    .add(attributes::locationDefinition, "/function/" + Function.stringKey())
    .serialize();
}

std::string yield::html::controlFlowNode(const MetaAddress &Address,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph.find(Address);
  revng_assert(Iterator != Function.ControlFlowGraph.end());

  auto Result = labeledBlock<false>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}
