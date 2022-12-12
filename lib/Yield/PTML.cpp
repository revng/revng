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
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/PTML.h"

using pipeline::serializedLocation;
using ptml::str;
using ptml::Tag;
namespace attributes = ptml::attributes;
namespace ptmlScopes = ptml::scopes;
namespace tags = ptml::tags;
namespace ranks = revng::ranks;

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
  std::string Location;
  if (auto Iterator = Binary.Functions().find(BasicBlock.Start());
      Iterator != Binary.Functions().end()) {
    LabelName = Iterator->name().str().str();
    FunctionPath = "/Functions/" + str(Iterator->key()) + "/CustomName";
    Location = serializedLocation(ranks::Function, Iterator->key());
  } else {
    LabelName = "basic_block_at_" + labelAddress(BasicBlock.Start());
    Location = serializedLocation(ranks::BasicBlock,
                                  model::Function(Function.Entry()).key(),
                                  BasicBlock.Start());
  }
  using model::Architecture::getAssemblyLabelIndicator;
  auto LabelIndicator = getAssemblyLabelIndicator(Binary.Architecture());
  Tag LabelTag(tags::Span, LabelName);
  LabelTag.addAttribute(attributes::Token, tokenTypes::Label)
    .addAttribute(attributes::LocationDefinition, Location);
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
  if (auto Iterator = Binary.Functions().find(Target);
      Iterator != Binary.Functions().end()) {
    // The target is a function
    return serializedLocation(ranks::Function, Iterator->Entry());
  } else if (auto Iterator = Function.ControlFlowGraph().find(Target);
             Iterator != Function.ControlFlowGraph().end()) {
    // The target is a basic block
    return serializedLocation(ranks::BasicBlock,
                              Function.Entry(),
                              Iterator->Start());
  } else if (Target.isValid()) {
    for (const auto &Block : Function.ControlFlowGraph()) {
      if (Block.Instructions().find(Target) != Block.Instructions().end()) {
        // The target is an instruction
        return serializedLocation(ranks::Instruction,
                                  Function.Entry(),
                                  Block.Start(),
                                  Target);
      }
    }
  }

  // The target is not known
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
  for (const auto &Edge : BasicBlock.Successors()) {
    auto TargetPair = efa::parseSuccessor(*Edge, BasicBlock.End(), Binary);
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
  revng_assert(!Instruction.Tags().empty(),
               "Tagless instructions are not supported");
  revng_assert(!Instruction.Disassembled().empty(),
               "Empty disassembled instructions are not supported");

  std::vector TagMap(Instruction.Disassembled().size(),
                     yield::TagType::Invalid);
  for (yield::Tag Tag : Instruction.Tags()) {
    revng_assert(Tag.Type() != yield::TagType::Invalid,
                 "\"Invalid\" TagType encountered");
    for (size_t Index = Tag.From(); Index < Tag.To(); Index++) {
      TagMap[Index] = Tag.Type();
    }
  }

  std::string Result;
  std::string Buffer;
  yield::TagType::Values Tag = yield::TagType::Invalid;
  for (size_t Index = 0; Index < Instruction.Disassembled().size(); Index++) {
    if (Tag != TagMap[Index]) {
      Result += tokenTag(Buffer, Tag);
      Tag = TagMap[Index];
      Buffer.clear();
    }
    Buffer += Instruction.Disassembled()[Index];
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
  size_t Tail = Instruction.Disassembled().size() + 1;

  Tag Location = Tag(tags::Span)
                   .addAttribute(attributes::LocationDefinition,
                                 serializedLocation(ranks::Instruction,
                                                    Function.Entry(),
                                                    BasicBlock.Start(),
                                                    Instruction.Address()));
  Tag Out = Tag(tags::Div, std::move(Result))
              .addAttribute(attributes::Scope, scopes::Instruction);

  if (AddTargets) {
    auto Targets = targets(BasicBlock, Function, Binary);
    Out.addListAttribute(attributes::LocationReferences, Targets);
  }

  return Location + Out;
}

static std::string basicBlock(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary,
                              std::string Label) {
  revng_assert(!BasicBlock.Instructions().empty());
  auto FromIterator = BasicBlock.Instructions().begin();
  auto ToIterator = std::prev(BasicBlock.Instructions().end());
  if (BasicBlock.HasDelaySlot()) {
    revng_assert(BasicBlock.Instructions().size() > 1);
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

  std::string LabelString;
  if (!Label.empty()) {
    LabelString = Label + "\n";
  } else {
    std::string Location = serializedLocation(ranks::BasicBlock,
                                              model::Function(Function.Entry())
                                                .key(),
                                              BasicBlock.Start());
    LabelString = Tag(tags::Span)
                    .addAttribute(attributes::LocationDefinition, Location)
                    .serialize();
  }

  return Tag(tags::Div, LabelString + Result)
    .addAttribute(attributes::Scope, scopes::BasicBlock)
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

  for (const auto &BasicBlock : Function.ControlFlowGraph()) {
    Result += labeledBlock<true>(BasicBlock, Function, Binary);
  }

  using pipeline::serializedLocation;
  namespace ranks = revng::ranks;
  return ::Tag(tags::Div, Result)
    .addAttribute(attributes::Scope, scopes::Function)
    .serialize();
}

std::string yield::ptml::controlFlowNode(const MetaAddress &Address,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph().find(Address);
  revng_assert(Iterator != Function.ControlFlowGraph().end());

  auto Result = labeledBlock<false>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}

namespace callGraphTokens {

static constexpr auto NodeLabel = "call-graph.node-label";
static constexpr auto ShallowNodeLabel = "call-graph.shallow-node-label";

} // namespace callGraphTokens

static Tag functionLinkHelper(const model::Function &Function,
                              llvm::StringRef TokenAttributeValue) {
  Tag Result(tags::Div, Function.name());
  Result.addAttribute(attributes::Token, TokenAttributeValue);

  return Result;
}

using pipeline::serializedLocation;

std::string
yield::ptml::functionNameDefinition(const MetaAddress &FunctionEntryPoint,
                                    const model::Binary &Binary) {
  if (FunctionEntryPoint.isInvalid())
    return "";

  return functionLinkHelper(Binary.Functions().at(FunctionEntryPoint),
                            callGraphTokens::NodeLabel)
    .addAttribute(attributes::LocationDefinition,
                  serializedLocation(revng::ranks::Function,
                                     FunctionEntryPoint))
    .serialize();
}

std::string yield::ptml::functionLink(const MetaAddress &FunctionEntryPoint,
                                      const model::Binary &Binary) {
  if (FunctionEntryPoint.isInvalid())
    return "";

  return functionLinkHelper(Binary.Functions().at(FunctionEntryPoint),
                            callGraphTokens::NodeLabel)
    .addListAttribute(attributes::LocationReferences,
                      serializedLocation(revng::ranks::Function,
                                         FunctionEntryPoint))
    .serialize();
}

std::string
yield::ptml::shallowFunctionLink(const MetaAddress &FunctionEntryPoint,
                                 const model::Binary &Binary) {
  if (FunctionEntryPoint.isInvalid())
    return "";

  return functionLinkHelper(Binary.Functions().at(FunctionEntryPoint),
                            callGraphTokens::ShallowNodeLabel)
    .addListAttribute(attributes::LocationReferences,
                      serializedLocation(revng::ranks::Function,
                                         FunctionEntryPoint))
    .serialize();
}
