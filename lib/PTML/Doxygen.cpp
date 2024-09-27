/// \file Doxygen.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Doxygen.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

namespace ptml::tokens {

static constexpr auto Keyword = "doxygen.keyword";
static constexpr auto Identifier = "doxygen.identifier";

} // namespace ptml::tokens

struct Attribute {
  std::string Name, Value;
};

struct DoxygenToken {
  enum class Types {
    /// Indicates that no special formatting is needed.
    Untagged,

    /// Wraps spaces together so they can be easier to work with by the UI.
    Indentation,

    /// Indicates that the contained text is a keyword, so it should be
    /// formatted as such.
    Keyword,

    /// Indicates that the contained text is a name, so it should be
    /// formatted as such.
    Identifier
  };

  Types Type;
  std::string Value;

  llvm::SmallVector<Attribute, 2> ExtraAttributes = {};

public:
  /// \note Consumes the value: must only be used once per object.
  std::string emit(const ptml::PTMLBuilder &B) {
    auto Result = B.getTag(ptml::tags::Span, std::move(Value));

    switch (Type) {
    case DoxygenToken::Types::Untagged:
      Result.addAttribute(ptml::attributes::Token, ptml::tokens::Comment);
      break;
    case DoxygenToken::Types::Indentation:
      Result.addAttribute(ptml::attributes::Token, ptml::tokens::Indentation);
      break;
    case DoxygenToken::Types::Keyword:
      Result.addAttribute(ptml::attributes::Token, ptml::tokens::Keyword);
      break;
    case DoxygenToken::Types::Identifier:
      Result.addAttribute(ptml::attributes::Token, ptml::tokens::Identifier);
      break;

    default:
      revng_abort("Unsupported doxygen tag.");
    }

    for (const auto &Attribute : ExtraAttributes)
      Result.addAttribute(Attribute.Name, Attribute.Value);

    return Result.toString();
  }
};

struct DoxygenLine {
  llvm::SmallVector<DoxygenToken, 8> Tags;
  size_t InternalIndentation;

  llvm::SmallVector<DoxygenToken, 8> *operator->() { return &Tags; }
};

struct CommentBuilder {
  const ptml::PTMLBuilder &PTML;
  llvm::StringRef Indicator;
  size_t Indentation;
  size_t WrapAt;

public:
  std::string emit(llvm::SmallVector<DoxygenLine, 16> &&Lines) {
    std::string Result;

    for (DoxygenLine &&Line : as_rvalue(Lines)) {
      auto [ResultLine, CurrentSize] = line();

      if (Line.Tags.empty())
        Result += toString(std::move(ResultLine));

      bool WereTagsEmittedSinceLastBreak = false;
      for (auto Iterator = Line->begin(); Iterator != Line->end(); ++Iterator) {
        Result += token(*Iterator,
                        Line.InternalIndentation,
                        ResultLine,
                        CurrentSize,
                        WereTagsEmittedSinceLastBreak);
      }

      if (WereTagsEmittedSinceLastBreak)
        Result += toString(std::move(ResultLine));
    }

    return Result;
  }

  std::string emit(llvm::StringRef Text) {
    DoxygenLine Argument{ .Tags = { DoxygenToken{
                            .Type = DoxygenToken::Types::Untagged,
                            .Value = Text.str() } },
                          .InternalIndentation = 0 };
    return emit({ std::move(Argument) });
  }

private:
  std::string token(const DoxygenToken &Token,
                    size_t InternalIndentation,
                    DoxygenLine &ResultLine,
                    size_t &CurrentSize,
                    bool &WereTagsEmittedSinceLastBreak) {
    std::string Result;

    llvm::StringRef TagText = Token.Value;
    while (!TagText.empty()) {
      if (size_t NewLinePosition = TagText.find('\n');
          NewLinePosition != TagText.npos
          && NewLinePosition + CurrentSize < WrapAt) {
        revng_assert(Token.Type == DoxygenToken::Types::Untagged,
                     "Line breaks are only allowed in untagged sections.");

        // There's a new line character in the next tag, use it as
        // the break point
        if (NewLinePosition != 0)
          ResultLine->emplace_back(DoxygenToken::Types::Untagged,
                                   TagText.substr(0, NewLinePosition).str(),
                                   Token.ExtraAttributes);

        auto NextCharacter = std::min(NewLinePosition + 1, TagText.size());
        TagText = TagText.drop_front(NextCharacter);

        Result += toString(std::move(ResultLine));
        std::tie(ResultLine, CurrentSize) = line(InternalIndentation);
        WereTagsEmittedSinceLastBreak = false;
      } else if (TagText.size() + CurrentSize < WrapAt) {
        // The rest of the tag fits into this line, just append it as is.
        ResultLine->emplace_back(DoxygenToken::Types::Untagged,
                                 TagText.str(),
                                 Token.ExtraAttributes);
        CurrentSize += TagText.size();
        TagText = "";
        WereTagsEmittedSinceLastBreak = true;
      } else if (auto LastSpace = TagText.rfind(' ', WrapAt - CurrentSize);
                 LastSpace != llvm::StringRef::npos) {
        // Tag doesn't fit, break on the last space that still does.
        DoxygenToken Tag{ .Type = Token.Type,
                          .Value = TagText.substr(0, LastSpace).str(),
                          .ExtraAttributes = Token.ExtraAttributes };
        ResultLine->emplace_back(std::move(Tag));
        Result += toString(std::move(ResultLine));
        std::tie(ResultLine, CurrentSize) = line(InternalIndentation);
        WereTagsEmittedSinceLastBreak = false;

        auto NextCharacter = std::min(LastSpace + 1, TagText.size());
        TagText = (LastSpace != llvm::StringRef::npos) ?
                    TagText.drop_front(NextCharacter) :
                    "";
      } else if (WereTagsEmittedSinceLastBreak) {
        // Tag doesn't fit and there's no good breaking point, but there
        // are already tags on this line: insert a break and try again.
        Result += toString(std::move(ResultLine));
        std::tie(ResultLine, CurrentSize) = line(InternalIndentation);
        WereTagsEmittedSinceLastBreak = false;
      } else {
        // No viable breaking point, make this line longer than expected.
        size_t FirstSpace = TagText.find(' ');
        if (FirstSpace == TagText.npos)
          FirstSpace = TagText.size();
        DoxygenToken Tag{ .Type = Token.Type,
                          .Value = TagText.substr(0, FirstSpace).str(),
                          .ExtraAttributes = Token.ExtraAttributes };
        ResultLine->emplace_back(std::move(Tag));
        Result += toString(std::move(ResultLine));
        std::tie(ResultLine, CurrentSize) = line(InternalIndentation);
        WereTagsEmittedSinceLastBreak = false;

        auto NextCharacter = std::min(FirstSpace + 1, TagText.size());
        TagText = (FirstSpace != llvm::StringRef::npos) ?
                    TagText.drop_front(NextCharacter) :
                    "";
      }
    }

    return Result;
  }

  std::tuple<DoxygenLine, size_t> firstLine() {
    DoxygenLine Result;

    if (Indentation != 0)
      Result->emplace_back(DoxygenToken::Types::Indentation,
                           std::string(Indentation, ' '));
    if (!Indicator.empty())
      Result->emplace_back(DoxygenToken::Types::Untagged,
                           Indicator.str() + ' ');

    Result.InternalIndentation = 0;

    return { std::move(Result), Indentation + Indicator.size() };
  }

  std::tuple<DoxygenLine, size_t> line(size_t IndentationSize = 0) {
    std::tuple<DoxygenLine, size_t> Result = firstLine();

    if (IndentationSize != 0) {
      auto &[Line, ResultSize] = Result;
      Line->emplace_back(DoxygenToken::Types::Indentation,
                         std::string(IndentationSize, ' '));
      Line.InternalIndentation = IndentationSize;
      ResultSize += IndentationSize;
    }

    return Result;
  }

  std::string toString(DoxygenLine &&Line) {
    std::string Result;

    revng_assert(!Line.Tags.empty());
    for (DoxygenToken &Tag : Line.Tags)
      Result += Tag.emit(PTML);

    return PTML.getTag(ptml::tags::Div, std::move(Result)).toString() += '\n';
  }
};

std::string ptml::freeFormComment(const ::ptml::PTMLBuilder &PTML,
                                  llvm::StringRef Text,
                                  llvm::StringRef CommentIndicator,
                                  size_t Indentation,
                                  size_t WrapAt) {
  CommentBuilder Builder(PTML, CommentIndicator, Indentation, WrapAt);
  auto Result = Builder.emit(Text.str());
  return Result.empty() ? Result : '\n' + Result;
}

using pipeline::toString;
namespace ranks = revng::ranks;

static llvm::SmallVector<DoxygenLine, 16>
gatherArgumentComments(const model::Binary &Binary,
                       const model::Function &Function) {
  if (Function.Prototype().isEmpty())
    return {};

  static constexpr std::string_view Keyword = "\\param ";

  llvm::SmallVector<DoxygenLine, 16> Result;
  if (auto *FT = Function.cabiPrototype()) {
    abi::FunctionType::Layout Layout(*FT);
    // Using layout here lets us put detailed register/stack information into
    // the comments. Consider that the same comments are used for disassembly
    // view, so seeing something like
    //
    // ```
    //   // A pretty cool function
    //   // \param i that takes in an int
    //   // \param s and a struct
    //   // \returns to return the sum of \ref i with one of \ref s fields.
    //   function:
    //       mov     eax, DWORD PTR [rsp+28]
    //       add     eax, edi
    //       ret
    // ```
    //
    // instead of (current version):
    //
    // ```
    //   // A pretty cool function
    //   // \param i (in rdi) that takes in an int
    //   // \param s (24 bytes at rsp+8) and a struct
    //   // \returns to return the sum of \ref i with one of \ref s fields.
    //   function:
    //       mov     eax, DWORD PTR [rsp+28]
    //       add     eax, edi
    //       ret
    // ```
    //
    // is pretty confusing, isn't it?

    std::size_t IndOffset = Layout.hasSPTAR() ? 1 : 0;
    revng_assert(FT->Arguments().size() + IndOffset == Layout.Arguments.size());
    for (size_t Index = 0; Index < FT->Arguments().size(); ++Index) {
      const std::string &Comment = FT->Arguments().at(Index).Comment();
      if (!Comment.empty()) {
        DoxygenLine &Line = Result.emplace_back();
        Line->emplace_back(DoxygenToken::Types::Keyword, std::string(Keyword));
        Line.InternalIndentation = Keyword.size();

        const model::Argument &Argument = FT->Arguments().at(Index);
        auto &N = Line->emplace_back(DoxygenToken::Types::Identifier,
                                     Argument.name().str().str());
        std::string Location = toString(ranks::CABIArgument,
                                        FT->key(),
                                        Argument.key());
        N.ExtraAttributes.emplace_back(ptml::attributes::ActionContextLocation,
                                       Location);
        N.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                       ptml::actions::Comment);
        N.ExtraAttributes.emplace_back(ptml::attributes::LocationReferences,
                                       Location);

        const auto CurrentArgument = Layout.Arguments[Index + IndOffset];
        if (!CurrentArgument.Registers.empty()) {
          Line->emplace_back(DoxygenToken::Types::Untagged, " (in ");
          for (auto Reg : skip_back(CurrentArgument.Registers)) {
            Line->emplace_back(DoxygenToken::Types::Identifier,
                               model::Register::getRegisterName(Reg).str());
            Line->emplace_back(DoxygenToken::Types::Untagged, ", ");
          }
          auto Last = CurrentArgument.Registers.back();
          Line->emplace_back(DoxygenToken::Types::Identifier,
                             model::Register::getRegisterName(Last).str());
          Line->emplace_back(DoxygenToken::Types::Untagged, ")");
        } else {
          revng_assert(CurrentArgument.Stack
                       && CurrentArgument.Stack->Size != 0);
          Line->emplace_back(DoxygenToken::Types::Untagged,
                             " (" + std::to_string(CurrentArgument.Stack->Size)
                               + " bytes at ");

          namespace Arch = model::Architecture;
          auto SP = Arch::getStackPointer(Binary.Architecture());
          Line->emplace_back(DoxygenToken::Types::Identifier,
                             model::Register::getRegisterName(SP).str());
          uint64_t Adjustment = Arch::getCallPushSize(Binary.Architecture());
          uint64_t Adjusted = CurrentArgument.Stack->Offset + Adjustment;
          Line->emplace_back(DoxygenToken::Types::Untagged,
                             "+" + std::to_string(Adjusted) + ")");
        }

        Line->emplace_back(DoxygenToken::Types::Untagged, " ");
        auto &Tag = Line->emplace_back(DoxygenToken::Types::Untagged, Comment);
        Tag.ExtraAttributes
          .emplace_back(ptml::attributes::ActionContextLocation,
                        toString(ranks::CABIArgument,
                                 FT->key(),
                                 Argument.key()));
        Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                         ptml::actions::Comment);
      }
    }
  } else if (auto *FT = Function.rawPrototype()) {
    for (const model::NamedTypedRegister &Argument : FT->Arguments()) {
      if (!Argument.Comment().empty()) {
        model::Register::Values Register = Argument.Location();

        DoxygenLine &Line = Result.emplace_back();
        Line->emplace_back(DoxygenToken::Types::Keyword, std::string(Keyword));
        Line.InternalIndentation = Keyword.size();

        auto &N = Line->emplace_back(DoxygenToken::Types::Identifier,
                                     Argument.name().str().str());
        std::string Location = toString(ranks::RawArgument,
                                        FT->key(),
                                        Argument.key());
        N.ExtraAttributes.emplace_back(ptml::attributes::ActionContextLocation,
                                       Location);
        N.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                       ptml::actions::Comment);
        N.ExtraAttributes.emplace_back(ptml::attributes::LocationReferences,
                                       Location);
        Line->emplace_back(DoxygenToken::Types::Untagged, " (in ");
        Line->emplace_back(DoxygenToken::Types::Identifier,
                           model::Register::getRegisterName(Register).str());
        Line->emplace_back(DoxygenToken::Types::Untagged, ")");
        auto &Tag = Line->emplace_back(DoxygenToken::Types::Untagged,
                                       Argument.Comment());
        Tag.ExtraAttributes
          .emplace_back(ptml::attributes::ActionContextLocation,
                        toString(ranks::RawArgument,
                                 FT->key(),
                                 Argument.key()));
        Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                         ptml::actions::Comment);
      }
    }

    if (const model::StructDefinition *Stack = FT->stackArgumentsType()) {
      struct FieldMapEntry {
        std::string Name;
        llvm::StringRef Comment;
        const model::StructField &Field;
      };
      llvm::SmallVector<FieldMapEntry> Comments;
      for (const model::StructField &Field : Stack->Fields())
        if (!Field.Comment().empty())
          Comments.emplace_back(Field.name().str().str(),
                                Field.Comment(),
                                Field);

      if (!Comments.empty()) {
        static constexpr std::string_view FirstLine = "stack_args ";

        bool IsFirst = true;
        for (auto [Name, Comment, Field] : Comments) {
          DoxygenLine &Line = Result.emplace_back();
          if (IsFirst) {
            Line->emplace_back(DoxygenToken::Types::Keyword,
                               std::string(Keyword));
            Line.InternalIndentation = Keyword.size();

            Line->emplace_back(DoxygenToken::Types::Untagged,
                               std::string(FirstLine));
            Line.InternalIndentation += FirstLine.size();

            IsFirst = false;
          } else {
            Line.InternalIndentation = Keyword.size() + FirstLine.size();
            Line->emplace_back(DoxygenToken::Types::Indentation,
                               std::string(Line.InternalIndentation, ' '));
          }

          auto &L = Line->emplace_back(DoxygenToken::Types::Identifier, Name);
          std::string Location = toString(ranks::StructField,
                                          Stack->key(),
                                          Field.key());
          L.ExtraAttributes
            .emplace_back(ptml::attributes::ActionContextLocation, Location);
          L.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                         ptml::actions::Comment);
          L.ExtraAttributes.emplace_back(ptml::attributes::LocationReferences,
                                         Location);

          auto &Tag = Line->emplace_back(DoxygenToken::Types::Untagged,
                                         ": " + Comment.str());
          Tag.ExtraAttributes
            .emplace_back(ptml::attributes::ActionContextLocation, Location);
          Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                           ptml::actions::Comment);
        }
      }
    }
  } else {
    revng_abort("Function prototype must be a function type.");
  }

  return Result;
}

static llvm::SmallVector<DoxygenLine, 16>
gatherReturnValueComments(const model::Binary &Binary,
                          const model::Function &Function) {
  if (Function.Prototype().isEmpty())
    return {};

  static constexpr std::string_view Keyword = "\\returns ";

  llvm::SmallVector<DoxygenLine, 16> Result;
  if (auto *FT = Function.cabiPrototype()) {
    if (!FT->ReturnValueComment().empty()) {
      DoxygenLine &Line = Result.emplace_back();
      Line->emplace_back(DoxygenToken::Types::Keyword, std::string(Keyword));
      Line.InternalIndentation = Keyword.size();

      DoxygenToken &Tag = Line->emplace_back(DoxygenToken::Types::Untagged,
                                             FT->ReturnValueComment());
      Tag.ExtraAttributes.emplace_back(ptml::attributes::ActionContextLocation,
                                       toString(ranks::ReturnValue, FT->key()));
      Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                       ptml::actions::Comment);
    }
  } else if (auto *FT = Function.rawPrototype()) {
    if (!FT->ReturnValueComment().empty()) {
      DoxygenLine &Line = Result.emplace_back();
      Line->emplace_back(DoxygenToken::Types::Keyword, std::string(Keyword));
      Line.InternalIndentation = Keyword.size();

      DoxygenToken &T = Line->emplace_back(DoxygenToken::Types::Untagged,
                                           FT->ReturnValueComment());
      T.ExtraAttributes.emplace_back(ptml::attributes::ActionContextLocation,
                                     toString(ranks::ReturnValue, FT->key()));
      T.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                     ptml::actions::Comment);
    }

    for (const model::NamedTypedRegister &ReturnValue : FT->ReturnValues()) {
      model::Register::Values Register = ReturnValue.Location();

      if (!ReturnValue.Comment().empty()) {
        if (Result.empty())
          Result.emplace_back();

        auto &Line = Result.emplace_back();
        Line.InternalIndentation = Keyword.size();

        if (Result.size() == 1) {
          Line->emplace_back(DoxygenToken::Types::Keyword,
                             std::string(Keyword));
        } else {
          Line->emplace_back(DoxygenToken::Types::Indentation,
                             std::string(Keyword.size(), ' '));
        }

        Line->emplace_back(DoxygenToken::Types::Identifier,
                           model::Register::getRegisterName(Register).str());
        auto &Tag = Line->emplace_back(DoxygenToken::Types::Untagged,
                                       ": " + ReturnValue.Comment());
        Tag.ExtraAttributes
          .emplace_back(ptml::attributes::ActionContextLocation,
                        toString(ranks::ReturnRegister,
                                 FT->key(),
                                 ReturnValue.key()));
        Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                         ptml::actions::Comment);
      }
    }

  } else {
    revng_abort("Function prototype must be a function type.");
  }

  return Result;
}

std::string ptml::functionComment(const ::ptml::PTMLBuilder &PTML,
                                  const model::Function &Function,
                                  const model::Binary &Binary,
                                  llvm::StringRef CommentIndicator,
                                  size_t Indentation,
                                  size_t WrapAt) {
  llvm::SmallVector<DoxygenLine, 16> Result;
  if (!Function.Comment().empty()) {
    DoxygenToken Tag{ .Type = DoxygenToken::Types::Untagged,
                      .Value = Function.Comment() };
    Tag.ExtraAttributes.emplace_back(ptml::attributes::ActionContextLocation,
                                     toString(ranks::Function, Function.key()));
    Tag.ExtraAttributes.emplace_back(ptml::attributes::AllowedActions,
                                     ptml::actions::Comment);
    Result.emplace_back(DoxygenLine{ .Tags = { std::move(Tag) } });
  }

  auto ArgumentComments = gatherArgumentComments(Binary, Function);
  if (!ArgumentComments.empty()) {
    if (!Result.empty())
      Result.emplace_back();

    Result.reserve(Result.size() + ArgumentComments.size());
    std::ranges::move(ArgumentComments, std::back_inserter(Result));
  }

  auto ReturnValueComments = gatherReturnValueComments(Binary, Function);
  if (!ReturnValueComments.empty()) {
    if (!Result.empty())
      Result.emplace_back();

    Result.reserve(Result.size() + ReturnValueComments.size());
    std::ranges::move(ReturnValueComments, std::back_inserter(Result));
  }

  CommentBuilder Builder(PTML, CommentIndicator, Indentation, WrapAt);
  return Builder.emit(std::move(Result));
}
