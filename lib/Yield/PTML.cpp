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

namespace options {

enum class AddressStyles {
  /// Look for the addresses among basic blocks and functions.
  /// When a match is found, replace the addresses with relevant labels.
  /// Otherwise, prints an absolute address instead.
  Smart,
  // TODO: extend to support segment lookup as well.

  /// Same as \ref Smart, except when unable to single out the target,
  /// print a PC-relative address instead.
  SmartWithPCRelativeFallback,

  /// Same as \ref Smart, except when unable to single out the target,
  /// print an error token.
  Strict,

  /// Convert PC relative addresses into global representation.
  Global,

  /// Print all the addresses exactly how disassembler emitted them
  /// in PC-relative mode.
  PCRelative
};
static AddressStyles AddressStyle = AddressStyles::Smart;

} // namespace options

using pipeline::serializedLocation;
using ptml::str;
using ptml::Tag;
namespace attributes = ptml::attributes;
namespace ptmlScopes = ptml::scopes;
namespace tags = ptml::tags;
namespace ranks = revng::ranks;

namespace tokenTypes {

static constexpr auto Helper = "asm.helper";
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

struct LabelDescription {
  std::string Name;
  std::string Location;
  std::string Path = "";
};
static LabelDescription labelImpl(const BasicBlockID &BasicBlock,
                                  const yield::Function &Function,
                                  const model::Binary &Binary) {
  const auto &CFG = Function.ControlFlowGraph();
  if (auto *ModelFunction = yield::tryGetFunction(Binary, BasicBlock)) {
    return LabelDescription{
      .Name = ModelFunction->name().str().str(),
      .Location = serializedLocation(ranks::Function, ModelFunction->key()),
      .Path = "/Functions/" + str(ModelFunction->key()) + "/CustomName"
    };
  } else if (CFG.find(BasicBlock) != CFG.end()) {
    return LabelDescription{
      .Name = "basic_block_at_" + labelAddress(BasicBlock),
      .Location = serializedLocation(ranks::BasicBlock,
                                     model::Function(Function.Entry()).key(),
                                     BasicBlock)
    };
  } else {
    revng_abort("Unable to emit a label for an object that does not exist.");
  }
}

static std::string labelDefinition(const BasicBlockID &BasicBlock,
                                   const yield::Function &Function,
                                   const model::Binary &Binary) {
  auto [Name, Location, Path] = labelImpl(BasicBlock, Function, Binary);

  Tag LabelTag(tags::Span, std::move(Name));
  LabelTag.addAttribute(attributes::Token, tokenTypes::Label)
    .addAttribute(attributes::LocationDefinition, Location);
  if (!Path.empty())
    LabelTag.addAttribute(attributes::ModelEditPath, Path);

  using model::Architecture::getAssemblyLabelIndicator;
  std::string Indicator(getAssemblyLabelIndicator(Binary.Architecture()));
  return LabelTag.serialize()
         + Tag(tags::Span, std::move(Indicator))
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

  revng_abort(("Unknown target:\n" + serializeToString(Target)).c_str());
}

static std::set<std::string> targets(const yield::BasicBlock &BasicBlock,
                                     const yield::Function &Function,
                                     const model::Binary &Binary) {
  std::set<std::string> Result;
  for (const auto &Edge : BasicBlock.Successors()) {
    auto [NextAddress, MaybeCall] = efa::parseSuccessor(*Edge,
                                                        BasicBlock.nextBlock(),
                                                        Binary);
    if (NextAddress.isValid())
      Result.emplace(targetPath(NextAddress, Function, Binary));

    if (MaybeCall.isValid())
      Result.emplace(targetPath(BasicBlockID(MaybeCall), Function, Binary));
  }

  // Explicitly remove the next target if there is only a single other target,
  // i.e. it's a conditional jump, call, etc.
  if (Result.size() == 2) {
    auto NextBlock = targetPath(BasicBlock.nextBlock(), Function, Binary);
    if (auto Iterator = llvm::find(Result, NextBlock); Iterator != Result.end())
      Result.erase(Iterator);
  }

  return Result;
}

static std::string tagTypeAsString(const yield::TagType::Values &Type) {
  switch (Type) {
  case yield::TagType::Address:
  case yield::TagType::PCRelativeAddress:
  case yield::TagType::AbsoluteAddress:
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
  case yield::TagType::Helper:
    return tokenTypes::Helper;
  case yield::TagType::Whitespace:
  case yield::TagType::Untagged:
    return "";
  default:
    revng_abort("Unknown tag type");
  }
}

static std::string
tokenTag(std::string &&Buffer, const yield::TagType::Values &Tag) {
  std::string TagStr = tagTypeAsString(Tag);
  if (!TagStr.empty()) {
    return ::Tag(tags::Span, std::move(Buffer))
      .addAttribute(attributes::Token, TagStr)
      .serialize();
  } else {
    return std::move(Buffer);
  }
}

static std::string taggedText(const yield::Instruction &Instruction) {
  revng_assert(!Instruction.Tags().empty(),
               "Tag-less instructions are not supported");
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
      Result += tokenTag(std::move(Buffer), Tag);
      Buffer.clear();

      Tag = TagMap[Index];
    }
    Buffer += Instruction.Disassembled()[Index];
  }
  Result += tokenTag(std::move(Buffer), Tag);

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
  std::string Label = labelDefinition(FirstBlock.ID(), Function, Binary);

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
  return ::Tag(tags::Div, std::move(Result))
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
