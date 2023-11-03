/// \file PTML.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ADT/Concepts.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
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
using ptml::PTMLBuilder;
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
};

static LabelDescription labelImpl(const BasicBlockID &BasicBlock,
                                  const yield::Function &Function,
                                  const model::Binary &Binary) {
  const auto &CFG = Function.ControlFlowGraph();
  if (auto *ModelFunction = yield::tryGetFunction(Binary, BasicBlock)) {
    return LabelDescription{
      .Name = ModelFunction->name().str().str(),
      .Location = serializedLocation(ranks::Function, ModelFunction->key()),
    };
  } else if (CFG.contains(BasicBlock)) {
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

static std::string labelDefinition(const PTMLBuilder &ThePTMLBuilder,
                                   const BasicBlockID &BasicBlock,
                                   const yield::Function &Function,
                                   const model::Binary &Binary) {
  auto [Name, Location] = labelImpl(BasicBlock, Function, Binary);

  Tag LabelTag = ThePTMLBuilder.getTag(tags::Span, Name);
  LabelTag.addAttribute(attributes::Token, tokenTypes::Label)
    .addAttribute(attributes::LocationDefinition, Location)
    .addAttribute(attributes::ActionContextLocation, Location);

  using model::Architecture::getAssemblyLabelIndicator;
  std::string Indicator(getAssemblyLabelIndicator(Binary.Architecture()));
  return LabelTag.serialize()
         + ThePTMLBuilder.getTag(tags::Span, Indicator)
             .addAttribute(attributes::Token, tokenTypes::LabelIndicator)
             .serialize();
}

static std::string labelReference(const PTMLBuilder &ThePTMLBuilder,
                                  const BasicBlockID &BasicBlock,
                                  const yield::Function &Function,
                                  const model::Binary &Binary) {
  auto [Name, Location] = labelImpl(BasicBlock, Function, Binary);

  Tag LabelTag = ThePTMLBuilder.getTag(tags::Span, std::move(Name));
  LabelTag.addAttribute(attributes::Token, tokenTypes::Label)
    .addAttribute(attributes::LocationReferences, Location);

  return LabelTag.serialize();
}

static std::string indent(const PTMLBuilder &ThePTMLBuilder) {
  return ThePTMLBuilder.getTag(tags::Span, "  ")
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
      if (Block.Instructions().contains(Target.start())) {
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

struct TaggedString {
  yield::TagType::Values Type;
  std::variant<std::string, std::string_view> Content;

public:
  TaggedString(yield::TagType::Values Type, std::string &&Content) :
    Type(Type), Content(std::move(Content)) {}
  TaggedString(yield::TagType::Values Type, std::string_view Content) :
    Type(Type), Content(Content) {}

  /// Exports the tag as PTML.
  ///
  /// \note this consumes \ref Content, so the tag is not usable after this
  ///       has been called.
  std::string emit(const PTMLBuilder &ThePTMLBuilder) {
    std::string TagStr = tagTypeAsString(Type);
    if (TagStr.empty())
      return moveContent();

    return ThePTMLBuilder.getTag(tags::Span, moveContent())
      .addAttribute(attributes::Token, TagStr)
      .serialize();
  }

  std::string_view content() const {
    return std::visit([](auto &S) -> std::string_view { return S; }, Content);
  }

private:
  /// Consume the tag to export its contents.
  std::string moveContent() {
    if (std::holds_alternative<std::string>(Content))
      return std::move(std::get<std::string>(Content));
    else if (std::holds_alternative<std::string_view>(Content))
      return std::string(std::get<std::string_view>(Content));
    else
      revng_abort("Unknown content type");
  }
};
using TaggedStrings = llvm::SmallVector<TaggedString, 16u>;

static std::vector<yield::Tag> sortTags(const SortedVector<yield::Tag> &Tags) {
  std::vector<yield::Tag> Result(Tags.begin(), Tags.end());
  std::sort(Result.begin(),
            Result.end(),
            [](const yield::Tag &LHS, const yield::Tag &RHS) {
              if (LHS.From() != RHS.From())
                return LHS.From() < RHS.From();
              else if (LHS.To() != RHS.To())
                return LHS.To() > RHS.To(); // reverse order
              else
                return LHS.Type() < RHS.Type();
            });
  return Result;
}

static TaggedStrings embedContentIntoTags(const std::vector<yield::Tag> &Tags,
                                          llvm::StringRef RawText) {
  TaggedStrings Result;

  for (const yield::Tag &Tag : Tags)
    Result.emplace_back(Tag.Type(), RawText.slice(Tag.From(), Tag.To()));

  return Result;
}

static TaggedStrings flattenTags(const SortedVector<yield::Tag> &Tags,
                                 llvm::StringRef RawText) {
  std::vector<yield::Tag> Result = sortTags(Tags);
  Result.emplace(Result.begin(), yield::TagType::Untagged, 0, RawText.size());
  for (std::ptrdiff_t Index = Result.size() - 1; Index >= 0; --Index) {
    yield::Tag &Current = Result[Index];
    auto IsParentOf = [&Current](const yield::Tag &Next) {
      if (Current.From() >= Next.From() && Current.To() <= Next.To())
        return true;
      else
        return false;
    };

    auto It = std::find_if(std::next(Result.rbegin(), Result.size() - Index),
                           Result.rend(),
                           IsParentOf);
    while (It != Result.rend()) {
      auto [ParentType, ParentFrom, ParentTo] = *It;
      auto [CurrentType, CurrentFrom, CurrentTo] = Result[Index];

      std::ptrdiff_t ParentIndex = std::distance(It, Result.rend()) - 1;
      if (ParentFrom == CurrentFrom) {
        Result.erase(std::next(It).base());
        It = Result.rend();
        --Index;
      } else {
        Result[ParentIndex].To() = CurrentFrom;
        It = std::find_if(std::next(Result.rbegin(), Result.size() - Index),
                          Result.rend(),
                          IsParentOf);
      }

      if (ParentTo != CurrentTo) {
        yield::Tag New(ParentType, CurrentTo, ParentTo);
        Result.insert(std::next(Result.begin(), Index + 1), std::move(New));
        Index += 2;
        break;
      }
    }
  }

  return embedContentIntoTags(Result, RawText);
}

static int64_t parseImmediate(llvm::StringRef String) {
  revng_assert(String.size() > 0);
  if (String[0] == '#')
    String = String.drop_front();

  revng_assert(String.size() > 0);
  bool IsNegative = String[0] == '-';
  if (IsNegative)
    String = String.drop_front();

  uint64_t Value;
  bool Failure = String.getAsInteger(0, Value);
  if (Failure || static_cast<int64_t>(Value) < 0) {
    std::string Error = "Unsupported immediate: " + String.str();
    revng_abort(Error.c_str());
  }

  if (IsNegative)
    return -static_cast<int64_t>(Value);
  else
    return +static_cast<int64_t>(Value);
}

static MetaAddress
absoluteAddressFromAbsoluteImmediate(const TaggedString &Input,
                                     const yield::Instruction &Instruction) {
  MetaAddress Result = Instruction.getRelativeAddressBase().toGeneric();
  return Result.replaceAddress(parseImmediate(Input.content()));
}

static MetaAddress
absoluteAddressFromPCRelativeImmediate(const TaggedString &Input,
                                       const yield::Instruction &Instruction) {
  MetaAddress Result = Instruction.getRelativeAddressBase().toGeneric();
  return Result += parseImmediate(Input.content());
}

static TaggedString toGlobal(const TaggedString &Input,
                             const MetaAddress &Address) {
  if (Address.isInvalid())
    return TaggedString{ yield::TagType::Immediate, std::string("invalid") };

  std::string_view Content = Input.content();
  std::string Prefix = (!Content.empty() && Content[0] == '#' ? "#0x" : "0x");
  std::string Body = llvm::utohexstr(Address.address(), true);
  return TaggedString{ yield::TagType::Immediate, std::move(Prefix += Body) };
}

static std::optional<TaggedString>
tryEmitLabel(const PTMLBuilder &ThePTMLBuilder,
             const MetaAddress &Address,
             const yield::BasicBlock &BasicBlock,
             const yield::Function &Function,
             const model::Binary &Binary) {
  if (Address.isInvalid())
    return std::nullopt;

  for (const auto &Successor : BasicBlock.Successors()) {
    // Ignore address spaces and epochs for now.
    // TODO: see what can be done about it.
    if (Successor->Destination().start().isValid()
        && Successor->Destination().start().address() == Address.address()) {
      // Since we have no easy way to decide which one of the successors
      // is better, stop looking after the first match.
      return TaggedString{ yield::TagType::Untagged,
                           labelReference(ThePTMLBuilder,
                                          Successor->Destination(),
                                          Function,
                                          Binary) };
    }
  }

  return std::nullopt;
}

static TaggedString emitAddress(const PTMLBuilder &ThePTMLBuilder,
                                TaggedString &&Input,
                                const MetaAddress &Address,
                                const yield::BasicBlock &BasicBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  using Styles = options::AddressStyles;
  if (options::AddressStyle == Styles::SmartWithPCRelativeFallback
      || options::AddressStyle == Styles::Smart
      || options::AddressStyle == Styles::Strict) {
    // "Smart" style selected, try to emit the label.
    if (std::optional MaybeLabel = tryEmitLabel(ThePTMLBuilder,
                                                Address,
                                                BasicBlock,
                                                Function,
                                                Binary)) {
      return std::move(*MaybeLabel);
    }
  }

  // "Simple" style selected OR "Smart" detection failed.
  if (options::AddressStyle == Styles::SmartWithPCRelativeFallback
      || options::AddressStyle == Styles::PCRelative) {
    // Emit a relative address.
    Input.Type = yield::TagType::Immediate;
    return std::move(Input);
  } else if (options::AddressStyle == Styles::Smart
             || options::AddressStyle == Styles::Global) {
    // Emit an absolute address.
    return toGlobal(Input, Address);
  } else if (options::AddressStyle == Styles::Strict) {
    // Emit an `invalid` marker.
    return TaggedString{ yield::TagType::Immediate, std::string("invalid") };
  } else {
    revng_abort("Unsupported addressing style.");
  }
}

static TaggedStrings handleSpecialCases(const PTMLBuilder &ThePTMLBuilder,
                                        TaggedStrings &&Input,
                                        const yield::Instruction &Instruction,
                                        const yield::BasicBlock &BasicBlock,
                                        const yield::Function &Function,
                                        const model::Binary &Binary) {
  TaggedStrings Result(std::move(Input));

  for (auto Iterator = Result.begin(); Iterator != Result.end(); ++Iterator) {
    if (Iterator->Type == yield::TagType::Address) {
      auto Address = absoluteAddressFromPCRelativeImmediate(*Iterator,
                                                            Instruction);
      *Iterator = emitAddress(ThePTMLBuilder,
                              std::move(*Iterator),
                              Address,
                              BasicBlock,
                              Function,
                              Binary);
    } else if (Iterator->Type == yield::TagType::AbsoluteAddress) {
      auto Address = absoluteAddressFromAbsoluteImmediate(*Iterator,
                                                          Instruction);
      *Iterator = emitAddress(ThePTMLBuilder,
                              std::move(*Iterator),
                              Address,
                              BasicBlock,
                              Function,
                              Binary);
    } else if (Iterator->Type == yield::TagType::PCRelativeAddress) {
      auto Address = absoluteAddressFromPCRelativeImmediate(*Iterator,
                                                            Instruction);
      TaggedStrings NewTags{ TaggedString{ yield::TagType::Helper,
                                           "offset_to("s },
                             emitAddress(ThePTMLBuilder,
                                         std::move(*Iterator),
                                         Address,
                                         BasicBlock,
                                         Function,
                                         Binary),
                             TaggedString{ yield::TagType::Helper, ")"s } };
      Iterator = Result.erase(Iterator);
      Iterator = Result.insert(Iterator, NewTags.begin(), NewTags.end());
      std::advance(Iterator, NewTags.size() - 1);
    } else {
      // TODO: handle other interesting tag types.
    }
  }

  return Result;
}

static std::string taggedText(const PTMLBuilder &ThePTMLBuilder,
                              const yield::Instruction &Instruction,
                              const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  revng_assert(!Instruction.Tags().empty(),
               "Tag-less instructions are not supported");
  revng_assert(!Instruction.Disassembled().empty(),
               "Empty disassembled instructions are not supported");

  TaggedStrings Flattened = flattenTags(Instruction.Tags(),
                                        Instruction.Disassembled());
  TaggedStrings Processed = handleSpecialCases(ThePTMLBuilder,
                                               std::move(Flattened),
                                               Instruction,
                                               BasicBlock,
                                               Function,
                                               Binary);

  std::string Result;
  for (auto &TaggedString : Processed)
    Result += TaggedString.emit(ThePTMLBuilder);

  return Result;
}

static std::string instruction(const PTMLBuilder &ThePTMLBuilder,
                               const yield::Instruction &Instruction,
                               const yield::BasicBlock &BasicBlock,
                               const yield::Function &Function,
                               const model::Binary &Binary,
                               bool AddTargets = false) {
  // Tagged instruction body.
  std::string Result = taggedText(ThePTMLBuilder,
                                  Instruction,
                                  BasicBlock,
                                  Function,
                                  Binary);
  size_t Tail = Instruction.Disassembled().size() + 1;

  std::string InstructionLocation = serializedLocation(ranks::Instruction,
                                                       Function.Entry(),
                                                       BasicBlock.ID(),
                                                       Instruction.Address());
  Tag Location = ThePTMLBuilder.getTag(tags::Span)
                   .addAttribute(attributes::LocationDefinition,
                                 InstructionLocation);
  Tag Out = ThePTMLBuilder.getTag(tags::Div, std::move(Result))
              .addAttribute(attributes::Scope, scopes::Instruction)
              .addAttribute(attributes::ActionContextLocation,
                            InstructionLocation);

  if (AddTargets) {
    auto Targets = targets(BasicBlock, Function, Binary);
    Out.addListAttribute(attributes::LocationReferences, Targets);
  }

  return Location + Out;
}

static std::string basicBlock(const PTMLBuilder &ThePTMLBuilder,
                              const yield::BasicBlock &BasicBlock,
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
    Result += indent(ThePTMLBuilder)
              + instruction(ThePTMLBuilder,
                            *Iterator,
                            BasicBlock,
                            Function,
                            Binary)
              + "\n";
  }
  Result += indent(ThePTMLBuilder)
            + instruction(ThePTMLBuilder,
                          *(ToIterator++),
                          BasicBlock,
                          Function,
                          Binary,
                          true)
            + "\n";

  std::string LabelString;
  if (!Label.empty()) {
    LabelString = Label + "\n";
  } else {
    std::string Location = serializedLocation(ranks::BasicBlock,
                                              model::Function(Function.Entry())
                                                .key(),
                                              BasicBlock.ID());
    LabelString = ThePTMLBuilder.getTag(tags::Span)
                    .addAttribute(attributes::LocationDefinition, Location)
                    .serialize();
  }

  return ThePTMLBuilder.getTag(tags::Div, LabelString + Result)
    .addAttribute(attributes::Scope, scopes::BasicBlock)
    .serialize();
}

template<bool ShouldMergeFallthroughTargets>
static std::string labeledBlock(const PTMLBuilder &ThePTMLBuilder,
                                const yield::BasicBlock &FirstBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  std::string Result;
  std::string Label = labelDefinition(ThePTMLBuilder,
                                      FirstBlock.ID(),
                                      Function,
                                      Binary);

  if constexpr (ShouldMergeFallthroughTargets == false) {
    Result = basicBlock(ThePTMLBuilder,
                        FirstBlock,
                        Function,
                        Binary,
                        std::move(Label))
             + "\n";
  } else {
    auto BasicBlocks = yield::cfg::labeledBlock(FirstBlock, Function, Binary);
    if (BasicBlocks.empty())
      return "";

    bool IsFirst = true;
    for (const auto &BasicBlock : BasicBlocks) {
      Result += basicBlock(ThePTMLBuilder,
                           *BasicBlock,
                           Function,
                           Binary,
                           IsFirst ? Label : "");
      IsFirst = false;
    }
    Result += "\n";
  }

  return Result;
}

std::string yield::ptml::functionAssembly(const PTMLBuilder &ThePTMLBuilder,
                                          const yield::Function &Function,
                                          const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.ControlFlowGraph()) {
    Result += labeledBlock<true>(ThePTMLBuilder, BasicBlock, Function, Binary);
  }

  return ThePTMLBuilder.getTag(tags::Div, Result)
    .addAttribute(attributes::Scope, scopes::Function)
    .serialize();
}

std::string yield::ptml::controlFlowNode(const PTMLBuilder &ThePTMLBuilder,
                                         const BasicBlockID &BasicBlock,
                                         const yield::Function &Function,
                                         const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph().find(BasicBlock);
  revng_assert(Iterator != Function.ControlFlowGraph().end());

  auto Result = labeledBlock<false>(ThePTMLBuilder,
                                    *Iterator,
                                    Function,
                                    Binary);
  revng_assert(!Result.empty());

  return Result;
}

namespace callGraphTokens {

static constexpr auto NodeLabel = "call-graph.node-label";
static constexpr auto ShallowNodeLabel = "call-graph.shallow-node-label";

} // namespace callGraphTokens

static model::Identifier functionNameHelper(std::string_view Location,
                                            const model::Binary &Binary) {
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

std::string
yield::ptml::functionNameDefinition(const PTMLBuilder &ThePTMLBuilder,
                                    std::string_view Location,
                                    const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result = ThePTMLBuilder.getTag(tags::Div,
                                             functionNameHelper(Location,
                                                                Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::NodeLabel);
  Result.addAttribute(attributes::LocationDefinition, Location);
  return Result.serialize();
}

std::string yield::ptml::functionLink(const PTMLBuilder &ThePTMLBuilder,
                                      std::string_view Location,
                                      const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result = ThePTMLBuilder.getTag(tags::Div,
                                             functionNameHelper(Location,
                                                                Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::NodeLabel);
  Result.addAttribute(attributes::LocationReferences, Location);
  return Result.serialize();
}

std::string yield::ptml::shallowFunctionLink(const PTMLBuilder &ThePTMLBuilder,
                                             std::string_view Location,
                                             const model::Binary &Binary) {
  if (Location.empty())
    return "";

  ::ptml::Tag Result = ThePTMLBuilder.getTag(tags::Div,
                                             functionNameHelper(Location,
                                                                Binary));
  Result.addAttribute(attributes::Token, callGraphTokens::ShallowNodeLabel);
  Result.addAttribute(attributes::LocationReferences, Location);
  return Result.serialize();
}
