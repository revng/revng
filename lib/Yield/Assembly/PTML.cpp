//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/// \file PTML.cpp
/// \brief

#include "revng/ADT/Concepts.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"

using ptml::str;
using ptml::Tag;
namespace attributes = ptml::attributes;
namespace ptmlScopes = ptml::scopes;
namespace tags = ptml::tags;

namespace tokenTypes {

static constexpr auto Label = "asm.label";
static constexpr auto LabelIndicator = "asm.label-indicator";
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

static std::string label(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary) {
  std::string LabelName;
  std::string FunctionPath;
  if (auto Iterator = Binary.Functions.find(BasicBlock.Start);
      Iterator != Binary.Functions.end()) {
    LabelName = Iterator->name().str().str();
    FunctionPath = "/Functions/" + str(Iterator->key()) + "/CustomName";
  } else {
    LabelName = "basic_block_at_" + labelAddress(BasicBlock.Start);
  }
  using model::Architecture::getAssemblyLabelIndicator;
  auto LabelIndicator = getAssemblyLabelIndicator(Binary.Architecture);
  Tag LabelTag(tags::Span, LabelName);
  LabelTag.addAttribute(attributes::Token, tokenTypes::Label);
  if (!FunctionPath.empty())
    LabelTag.addAttribute(attributes::ModelEditPath, FunctionPath);

  return LabelTag.serialize()
         + Tag(tags::Span, LabelIndicator)
             .addAttribute(attributes::Token, tokenTypes::LabelIndicator)
             .serialize();
}

static std::string indent() {
  return Tag(tags::Span, "  ")
    .addAttribute(attributes::Token, ptml::tokens::Indentation)
    .serialize();
}

static std::string targetPath(const MetaAddress &Target,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  if (auto Iterator = Binary.Functions.find(Target);
      Iterator != Binary.Functions.end()) {
    // The target is a function
    return "/function/" + str(Iterator->key());
  } else if (auto Iterator = Function.ControlFlowGraph.find(Target);
             Iterator != Function.ControlFlowGraph.end()) {
    // The target is a basic block
    return "/basic-block/" + str(Function.key()) + "/"
           + Iterator->Start.toString();
  } else if (Target.isValid()) {
    for (auto BasicBlock : Function.ControlFlowGraph) {
      for (auto Instruction : BasicBlock.Instructions) {
        if (Instruction.Address == Target) {
          return "/instruction/" + str(Function.key()) + "/"
                 + BasicBlock.Start.toString() + "/" + Target.toString();
        }
      }
    }
  }
  return "";
}

static std::set<std::string> targets(const yield::BasicBlock &BasicBlock,
                                     const yield::Function &Function,
                                     const model::Binary &Binary) {

  static const efa::ParsedSuccessor UnknownTarget{
    .NextInstructionAddress = MetaAddress::invalid(),
    .OptionalCallAddress = MetaAddress::invalid()
  };

  std::set<std::string> Result;
  for (const auto &Edge : BasicBlock.Successors) {
    auto TargetPair = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
    if (TargetPair.NextInstructionAddress.isValid()) {
      std::string Path = targetPath(TargetPair.NextInstructionAddress,
                                    Function,
                                    Binary);
      if (!Path.empty()) {
        Result.insert(Path);
      }
    }
    if (TargetPair.OptionalCallAddress.isValid()) {
      std::string Path = targetPath(TargetPair.OptionalCallAddress,
                                    Function,
                                    Binary);
      if (!Path.empty()) {
        Result.insert(Path);
      }
    }
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
    return ::Tag(tags::Span, Buffer)
      .addAttribute(attributes::Token, TagStr)
      .serialize();
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

static std::string instruction(const yield::Instruction &Instruction,
                               const yield::BasicBlock &BasicBlock,
                               const yield::Function &Function,
                               const model::Binary &Binary,
                               bool AddTargets = false) {

  // Tagged instruction body.
  std::string Result = taggedText(Instruction);
  size_t Tail = Instruction.Disassembled.size() + 1;

  Tag Out = Tag(tags::Div, std::move(Result))
              .addAttribute(attributes::Scope, scopes::Instruction)
              .addAttribute(attributes::LocationDefinition,
                            "/instruction/" + str(Function.key()) + "/"
                              + BasicBlock.Start.toString() + "/"
                              + Instruction.Address.toString());

  if (AddTargets) {
    auto Targets = targets(BasicBlock, Function, Binary);
    Out.addListAttribute(attributes::LocationReferences, Targets);
  }

  return Out.serialize();
}

static std::string basicBlock(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary,
                              std::string Label) {
  revng_assert(!BasicBlock.Instructions.empty());
  auto FromIterator = BasicBlock.Instructions.begin();
  auto ToIterator = std::prev(BasicBlock.Instructions.end());
  if (BasicBlock.HasDelaySlot) {
    revng_assert(BasicBlock.Instructions.size() > 1);
    --ToIterator;
  }

  std::string Result;
  for (auto Iterator = FromIterator; Iterator != ToIterator; ++Iterator) {
    Result += indent() + instruction(*Iterator, BasicBlock, Function, Binary)
              + "\n";
  }
  Result += indent()
            + instruction(*(ToIterator++), BasicBlock, Function, Binary, true)
            + "\n";

  return Tag(tags::Div, Label + (Label.empty() ? "" : "\n") + Result)
    .addAttribute(attributes::Scope, scopes::BasicBlock)
    .addAttribute(attributes::LocationDefinition,
                  "/basic-block/" + str(Function.key()) + "/"
                    + BasicBlock.Start.toString())
    .serialize();
}

template<bool ShouldMergeFallthroughTargets>
static std::string labeledBlock(const yield::BasicBlock &FirstBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  std::string Result;
  std::string Label = label(FirstBlock, Function, Binary);

  if constexpr (ShouldMergeFallthroughTargets == false) {
    Result = basicBlock(FirstBlock, Function, Binary, std::move(Label)) + "\n";
  } else {
    auto BasicBlocks = yield::cfg::labeledBlock(FirstBlock, Function, Binary);
    if (BasicBlocks.empty())
      return "";

    bool IsFirst = true;
    for (const auto &BasicBlock : BasicBlocks) {
      Result += basicBlock(*BasicBlock, Function, Binary, IsFirst ? Label : "");
      IsFirst = false;
    }
    Result += "\n";
  }

  return Result;
}

std::string yield::ptml::functionAssembly(const yield::Function &Function,
                                          const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.ControlFlowGraph) {
    Result += labeledBlock<true>(BasicBlock, Function, Binary);
  }

  return ::Tag(tags::Div, Result)
    .addAttribute(attributes::Scope, scopes::Function)
    .addAttribute(attributes::LocationDefinition,
                  "/function/" + str(Function.key()))
    .serialize();
}

std::string yield::ptml::controlFlowNode(const MetaAddress &Address,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph.find(Address);
  revng_assert(Iterator != Function.ControlFlowGraph.end());

  auto Result = labeledBlock<false>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}
