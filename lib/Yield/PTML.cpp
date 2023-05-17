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

static std::string labelAddress(const BasicBlockID &Address) {
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
  if (auto *F = yield::tryGetFunction(Binary, BasicBlock.ID())) {
    LabelName = F->name().str().str();
    FunctionPath = "/Functions/" + str(F->key()) + "/CustomName";
    Location = serializedLocation(ranks::Function, F->key());
  } else {
    LabelName = "basic_block_at_" + labelAddress(BasicBlock.ID());
    Location = serializedLocation(ranks::BasicBlock,
                                  model::Function(Function.Entry()).key(),
                                  BasicBlock.ID());
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

static std::string targetPath(const BasicBlockID &Target,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  if (const auto *F = yield::tryGetFunction(Binary, Target)) {
    // The target is a function
    return serializedLocation(ranks::Function, F->Entry());
  } else if (auto Iterator = Function.ControlFlowGraph().find(Target);
             Iterator != Function.ControlFlowGraph().end()) {
    // The target is a basic block
    return serializedLocation(ranks::BasicBlock,
                              Function.Entry(),
                              Iterator->ID());
  } else if (Target.isValid()) {
    for (const auto &Block : Function.ControlFlowGraph()) {
      if (Block.Instructions().count(Target.start()) != 0) {
        // The target is an instruction
        return serializedLocation(ranks::Instruction,
                                  Function.Entry(),
                                  Block.ID(),
                                  Target.start());
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
    .NextInstructionAddress = BasicBlockID::invalid(),
    .OptionalCallAddress = MetaAddress::invalid()
  };

  std::set<std::string> Result;
  for (const auto &Edge : BasicBlock.Successors()) {
    auto TargetPair = efa::parseSuccessor(*Edge,
                                          BasicBlock.nextBlock(),
                                          Binary);
    if (TargetPair.NextInstructionAddress.isValid()) {
      std::string Path = targetPath(TargetPair.NextInstructionAddress,
                                    Function,
                                    Binary);
      if (!Path.empty()) {
        Result.insert(Path);
      }
    }
    if (TargetPair.OptionalCallAddress.isValid()) {
      BasicBlockID ID = BasicBlockID(TargetPair.OptionalCallAddress);
      std::string Path = targetPath(ID, Function, Binary);
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
                                                    BasicBlock.ID(),
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
                                              BasicBlock.ID());
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

std::string yield::ptml::controlFlowNode(const BasicBlockID &BasicBlock,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph().find(BasicBlock);
  revng_assert(Iterator != Function.ControlFlowGraph().end());

  auto Result = labeledBlock<false>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}

namespace callGraphTokens {

static constexpr auto NodeLabel = "call-graph.node-label";
static constexpr auto ShallowNodeLabel = "call-graph.shallow-node-label";

} // namespace callGraphTokens

static model::Identifier
functionNameHelper(std::string_view Location, const model::Binary &Binary) {
  if (auto L = pipeline::locationFromString(revng::ranks::DynamicFunction,
                                            Location)) {
    auto Key = std::get<0>(L->at(revng::ranks::DynamicFunction));
    auto Iterator = Binary.ImportedDynamicFunctions().find(Key);
    revng_assert(Iterator != Binary.ImportedDynamicFunctions().end());
    return Iterator->name();
  } else if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                                   Location)) {
    auto Key = std::get<0>(L->at(revng::ranks::Function));
    auto Iterator = Binary.Functions().find(Key);
    revng_assert(Iterator != Binary.Functions().end());
    return Iterator->name();
  } else {
    revng_abort("Unsupported function type.");
  }
}

std::string yield::ptml::functionNameDefinition(std::string_view Location,
                                                const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result(tags::Div, functionNameHelper(Location, Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::NodeLabel);
  Result.addAttribute(attributes::LocationDefinition, Location);
  return Result.serialize();
}

std::string yield::ptml::functionLink(std::string_view Location,
                                      const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result(tags::Div, functionNameHelper(Location, Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::NodeLabel);
  Result.addAttribute(attributes::LocationReferences, Location);
  return Result.serialize();
}

std::string yield::ptml::shallowFunctionLink(std::string_view Location,
                                             const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result(tags::Div, functionNameHelper(Location, Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::ShallowNodeLabel);
  Result.addAttribute(attributes::LocationReferences, Location);
  return Result.serialize();
}
