/// \file DisassemblyHelper.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Yield/Function.h"

//
// Address serialization helper
//

template<yield::MetaAddressOrBasicBlockID T>
std::string
yield::sanitizedAddress(const T &Target, const model::Binary &Binary) {
  const auto &Configuration = Binary.Configuration().Disassembly();
  std::optional<llvm::Triple::ArchType> SerializationStyle = std::nullopt;
  if (!Configuration.PrintFullMetaAddress()) {
    namespace Arch = model::Architecture;
    SerializationStyle = Arch::toLLVMArchitecture(Binary.Architecture());
  }

  std::string Result = Target.toString(std::move(SerializationStyle));

  constexpr std::array ForbiddenCharacters = { ' ', ':', '!', '#',  '?',
                                               '<', '>', '/', '\\', '{',
                                               '}', '[', ']' };

  for (char &Character : Result)
    if (llvm::is_contained(ForbiddenCharacters, Character))
      Character = '_';

  return Result;
}

template std::string yield::sanitizedAddress(const MetaAddress &Target,
                                             const model::Binary &Binary);
template std::string yield::sanitizedAddress(const BasicBlockID &Target,
                                             const model::Binary &Binary);

//
// Tag to tag converter
//

static std::vector<yield::Instruction::RawTag>
sortTags(std::vector<yield::Instruction::RawTag> &&Tags) {
  std::vector<yield::Instruction::RawTag> Result = std::move(Tags);
  std::sort(Result.begin(),
            Result.end(),
            [](const yield::Instruction::RawTag &LHS,
               const yield::Instruction::RawTag &RHS) {
              if (LHS.From != RHS.From)
                return LHS.From < RHS.From;
              else if (LHS.To != RHS.To)
                return LHS.To > RHS.To; // reverse order
              else
                return LHS.Type < RHS.Type;
            });
  return Result;
}

static SortedVector<yield::TaggedString>
embedContentIntoTags(const std::vector<yield::Instruction::RawTag> &Tags,
                     llvm::StringRef Text) {
  SortedVector<yield::TaggedString> Result;

  uint64_t Index = 0;
  for (const yield::Instruction::RawTag &T : Tags) {
    auto &&[New, S] = Result.emplace(Index++, T.Type, Text.slice(T.From, T.To));
    revng_assert(S);

    if (New->Type() == yield::TagType::Whitespace)
      if (New->Content().find("\n") != std::string::npos)
        New->Content() = "\n";
  }

  return Result;
}

static auto flattenTags(std::vector<yield::Instruction::RawTag> &&Tags,
                        llvm::StringRef RawText) {
  std::vector<yield::Instruction::RawTag> Result = sortTags(std::move(Tags));
  Result.emplace(Result.begin(), yield::TagType::Untagged, 0, RawText.size());
  for (std::ptrdiff_t Index = Result.size() - 1; Index >= 0; --Index) {
    yield::Instruction::RawTag &Current = Result[Index];
    auto IsParentOf = [&Current](const yield::Instruction::RawTag &Next) {
      if (Current.From >= Next.From && Current.To <= Next.To)
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
        Result[ParentIndex].To = CurrentFrom;
        It = std::find_if(std::next(Result.rbegin(), Result.size() - Index),
                          Result.rend(),
                          IsParentOf);
      }

      if (ParentTo != CurrentTo) {
        yield::Instruction::RawTag New(ParentType, CurrentTo, ParentTo);
        Result.insert(std::next(Result.begin(), Index + 1), std::move(New));
        Index += 2;
        break;
      }
    }
  }

  return embedContentIntoTags(Result, RawText);
}

using Directives = std::tuple<SortedVector<yield::TaggedString>,
                              SortedVector<yield::TaggedLine>,
                              SortedVector<yield::TaggedLine>>;
static Directives detectDirectives(SortedVector<yield::TaggedString> &&Tagged) {
  // Split the input into separate lines.
  uint64_t Index = 0;
  auto ParsedUntil = Tagged.begin();
  std::vector<yield::TaggedLine> Lines;
  for (auto Iterator = Tagged.begin(); Iterator != Tagged.end(); ++Iterator) {
    if (Iterator->Type() == yield::TagType::Whitespace) {
      if (Iterator->Content() == "\n") {
        revng_assert(ParsedUntil != Iterator);
        Lines.emplace_back(Index++,
                           std::make_move_iterator(ParsedUntil),
                           std::make_move_iterator(Iterator));
        ParsedUntil = Iterator;
        ++ParsedUntil; // Skip the whitespace tag that triggered the split.
      }
    }
  }

  if (Lines.empty()) {
    // This is a single-line instruction, so no directives.
    return Directives{ std::move(Tagged), {}, {} };
  } else {
    Lines.emplace_back(Index++,
                       std::make_move_iterator(ParsedUntil),
                       std::make_move_iterator(Tagged.end()));
  }

  // Look at each line and determine whether it's a directive or not.
  Directives Result;
  auto &[Instruction, Preceding, Following] = Result;
  for (auto &Line : Lines) {
    revng_assert(!Line.Tags().empty());
    revng_assert(!Line.Tags().begin()->Content().empty());
    if (Line.Tags().begin()->Content()[0] == '.') {
      if (Instruction.empty()) {
        // No instruction yet, this is a preceding directive.
        auto &&[_, Success] = Preceding.emplace(std::move(Line));
        revng_assert(Success);
      } else {
        // Instruction is set already, this is a following directive.
        auto &&[_, Success] = Following.emplace(std::move(Line));
        revng_assert(Success);
      }
    } else {
      // This is not a directive, set it as the main instruction.
      revng_assert(Instruction.empty(),
                   "Multi-line instructions are not supported.");
      Instruction = std::move(Line.Tags());
    }
  }

  // Fix indexing and mark appropriate tags as `Directive`s.
  for (auto &&[Index, Tag] : llvm::enumerate(Instruction))
    Tag.Index() = Index;
  for (auto &&[Index, String] : llvm::enumerate(Preceding)) {
    String.Index() = Index;
    for (auto &&[Index, Tag] : llvm::enumerate(String.Tags())) {
      Tag.Index() = Index;
      if (Tag.Type() == yield::TagType::Untagged && Tag.Content()[0] == '.')
        Tag.Type() = yield::TagType::Directive;
    }
  }
  for (auto &&[Index, String] : llvm::enumerate(Following)) {
    String.Index() = Index;
    for (auto &&[Index, Tag] : llvm::enumerate(String.Tags())) {
      Tag.Index() = Index;
      if (Tag.Type() == yield::TagType::Untagged && Tag.Content()[0] == '.')
        Tag.Type() = yield::TagType::Directive;
    }
  }

  return Result;
}

void yield::Instruction::importTags(std::vector<RawTag> &&Tags,
                                    std::string &&Content) {
  Disassembled() = flattenTags(std::move(Tags), std::move(Content));
  std::tie(Disassembled(),
           PrecedingDirectives(),
           FollowingDirectives()) = detectDirectives(std::move(Disassembled()));
}

//
// Logic for replacing immediates with labels.
//

struct LabelDescription {
  std::string Name;
  std::string Location;
};

static LabelDescription labelImpl(const BasicBlockID &BasicBlock,
                                  const yield::Function &Function,
                                  const model::Binary &Binary,
                                  model::NameBuilder &NameBuilder) {
  if (auto *ModelFunction = yield::tryGetFunction(Binary, BasicBlock)) {
    return LabelDescription{
      .Name = NameBuilder.name(*ModelFunction),
      .Location = locationString(revng::ranks::Function, ModelFunction->key()),
    };
  } else if (Function.Blocks().contains(BasicBlock)) {
    std::string BBPr = Binary.Configuration().Disassembly().BasicBlockPrefix();
    if (BBPr.empty()) {
      // TODO: introduce a better way to handle default configuration values.
      BBPr = "bb_";
    }

    return LabelDescription{
      .Name = BBPr + yield::sanitizedAddress(BasicBlock, Binary),
      .Location = locationString(revng::ranks::BasicBlock,
                                 model::Function(Function.Entry()).key(),
                                 BasicBlock)
    };
  } else {
    revng_abort("Unable to emit a label for an object that does not exist.");
  }
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
absoluteAddressFromAbsoluteImmediate(const yield::TaggedString &Input,
                                     const yield::Instruction &Instruction) {
  MetaAddress Result = Instruction.getRelativeAddressBase().toGeneric();
  return Result.replaceAddress(parseImmediate(Input.Content()));
}

static MetaAddress
absoluteAddressFromPCRelativeImmediate(const yield::TaggedString &Input,
                                       const yield::Instruction &Instruction) {
  MetaAddress Result = Instruction.getRelativeAddressBase().toGeneric();
  return Result += parseImmediate(Input.Content());
}

static bool tryToTurnIntoALabel(yield::TaggedString &Input,
                                const MetaAddress &Address,
                                const yield::BasicBlock &BasicBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary,
                                model::NameBuilder &NameBuilder) {
  if (Address.isInvalid())
    return false;

  for (const auto &Successor : BasicBlock.Successors()) {
    // Ignore address spaces and epochs for now.
    // TODO: see what can be done about it.
    if (Successor->Destination().start().isValid()
        && Successor->Destination().start().address() == Address.address()) {
      // Since we have no easy way to decide which one of the successors
      // is better, stop looking after the first match.
      auto &&[Name, Loc] = labelImpl(Successor->Destination(),
                                     Function,
                                     Binary,
                                     NameBuilder);

      Input.Type() = yield::TagType::Label;
      Input.Content() = std::move(Name);
      Input.Attributes().emplace(ptml::attributes::LocationReferences,
                                 std::move(Loc));
      return true;
    }
  }

  return false;
}

static yield::TaggedString emitAddress(yield::TaggedString &&Input,
                                       const MetaAddress &Address,
                                       const yield::BasicBlock &BasicBlock,
                                       const yield::Function &Function,
                                       const model::Binary &Binary,
                                       model::NameBuilder &NameBuilder) {
  namespace Style = model::DisassemblyConfigurationAddressStyle;
  const auto &Configuration = Binary.Configuration().Disassembly();
  Style::Values AddressStyle = Configuration.AddressStyle();
  if (AddressStyle == Style::Invalid) {
    // TODO: introduce a better way to handle default configuration values.
    AddressStyle = Style::Smart;
  }

  if (AddressStyle == Style::SmartWithPCRelativeFallback
      || AddressStyle == Style::Smart || AddressStyle == Style::Strict) {
    // "Smart" style selected, try to emit the label.
    if (tryToTurnIntoALabel(Input,
                            Address,
                            BasicBlock,
                            Function,
                            Binary,
                            NameBuilder)) {
      return std::move(Input);
    }
  }

  // "Simple" style selected OR "Smart" detection failed.
  if (AddressStyle == Style::SmartWithPCRelativeFallback
      || AddressStyle == Style::PCRelative) {
    // Emit a relative address.
    Input.Type() = yield::TagType::Immediate;
    return std::move(Input);

  } else if (AddressStyle == Style::Smart || AddressStyle == Style::Global) {
    // Emit an absolute address.
    if (Address.isInvalid()) {
      Input.Type() = yield::TagType::Immediate;
      Input.Content() = "invalid";
    } else {
      Input.Type() = yield::TagType::Immediate;

      std::string Prefix;
      if (!Input.Content().empty() && Input.Content()[0] == '#')
        Prefix += '#';
      Prefix += "0x";
      Input.Content() = Prefix + llvm::utohexstr(Address.address(), true);
    }
    return std::move(Input);

  } else if (AddressStyle == Style::Strict) {
    // Turn it into an `invalid` marker
    Input.Type() = yield::TagType::Immediate;
    Input.Content() = "invalid";
    return std::move(Input);

  } else {
    revng_abort("Unsupported addressing style.");
  }
}

static SortedVector<yield::TaggedString>
handleSpecialCases(SortedVector<yield::TaggedString> &&Input,
                   const yield::Instruction &Instruction,
                   const yield::BasicBlock &BasicBlock,
                   const yield::Function &Function,
                   const model::Binary &Binary,
                   model::NameBuilder &NameBuilder) {
  SortedVector<yield::TaggedString> Result;

  uint64_t IndexOffset = 0;
  for (auto Iterator = Input.begin(); Iterator != Input.end(); ++Iterator) {
    Iterator->Index() += IndexOffset;

    if (Iterator->Type() == yield::TagType::Address) {
      auto Address = absoluteAddressFromPCRelativeImmediate(*Iterator,
                                                            Instruction);
      Result.emplace(emitAddress(std::move(*Iterator),
                                 Address,
                                 BasicBlock,
                                 Function,
                                 Binary,
                                 NameBuilder));
    } else if (Iterator->Type() == yield::TagType::AbsoluteAddress) {
      auto Address = absoluteAddressFromAbsoluteImmediate(*Iterator,
                                                          Instruction);
      Result.emplace(emitAddress(std::move(*Iterator),
                                 Address,
                                 BasicBlock,
                                 Function,
                                 Binary,
                                 NameBuilder));
    } else if (Iterator->Type() == yield::TagType::PCRelativeAddress) {
      auto Address = absoluteAddressFromPCRelativeImmediate(*Iterator,
                                                            Instruction);

      uint64_t CurrentIndex = ++Iterator->Index() - 1;
      Result.emplace(CurrentIndex, yield::TagType::Helper, "offset_to("s);
      Result.emplace(emitAddress(std::move(*Iterator),
                                 Address,
                                 BasicBlock,
                                 Function,
                                 Binary,
                                 NameBuilder));
      Result.emplace(CurrentIndex + 2, yield::TagType::Helper, ")"s);
      IndexOffset += 2;
    } else {
      // TODO: handle other interesting tag types.

      Result.emplace(std::move(*Iterator));
    }
  }

  return Result;
}

void yield::Instruction::handleSpecialTags(const yield::BasicBlock &BasicBlock,
                                           const yield::Function &Function,
                                           const model::Binary &Binary,
                                           model::NameBuilder &NameBuilder) {
  Disassembled() = handleSpecialCases(std::move(Disassembled()),
                                      *this,
                                      BasicBlock,
                                      Function,
                                      Binary,
                                      NameBuilder);
}

void yield::BasicBlock::setLabel(const yield::Function &Function,
                                 const model::Binary &Binary,
                                 model::NameBuilder &NameBuilder) {
  auto &&[N, Location] = labelImpl(ID(), Function, Binary, NameBuilder);

  SortedVector<TagAttribute> Attributes;
  Attributes.emplace(ptml::attributes::LocationDefinition, Location);
  Attributes.emplace(ptml::attributes::ActionContextLocation, Location);
  Label() = { 0, yield::TagType::Label, std::move(N), std::move(Attributes) };
}
