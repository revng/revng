#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/LineRange.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentingEmitter.h"

namespace ptml {

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  std::optional<llvm::StringRef> CommentHeader;
  std::optional<llvm::StringRef> CommentFooter;
  llvm::StringRef LinePrefix;
};

/// An Emitter capable of emitting Doxygen comments. While at the time of
/// creation only C comments are emitted (using CDoxygenEmitter via
/// CTokenEmitter::CommentEmitter), future use for assembly comments is planned.
template<PTMLEmitter EmitterT>
class DoxygenEmitter : IndentingEmitter<EmitterT> {
private:
  using BaseEmitter = IndentingEmitter<EmitterT>;

private:
  DoxygenCommentConfiguration Configuration;
  bool IsAtBeginningOfLine = true;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitter(const DoxygenCommentConfiguration &Configuration,
                          ArgsT &&...Args) :
    BaseEmitter(IndentString(" "), std::forward<ArgsT>(Args)...),
    Configuration(Configuration) {

    if (Configuration.CommentHeader) {
      EmitterT::emit(*Configuration.CommentHeader);
      EmitterT::emit(llvm::StringRef("\n"));
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    // TODO: emitting the signifier before the tag leads to it not being
    //       highlighted as a part of the keyword, but it's still better than
    //       the alternative of the tag extending over the line comment
    //       introducer (which is a bug).
    //
    // Here is an illustration of what I mean:
    //
    // With signifier *after* the tag, we emit (old behavior, bug):
    // ```xml
    //   <keyword-start-tag>/// \\param<keyword-end-tag> . . .
    // ```
    //
    // With signifier *before* the tag, we emit (current behavior, workaround):
    // ```xml
    //   /// \\<keyword-start-tag>param<keyword-end-tag> . . .
    // ```
    //
    // Ideally we want:
    // ```xml
    //   /// <keyword-start-tag>\\param<keyword-end-tag> . . .
    // ```
    // but that requires non-trivial changes to the emitter hierarchy.
    // Hence the TODO.
    DoxygenEmitter::emit(llvm::StringRef(&Configuration.KeywordSignifier, 1));

    auto Tag = BaseEmitter::initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::doxygen::tokens::Keyword);
    Tag.finalizeOpenTag();

    DoxygenEmitter::emit(Keyword);
  }

  DoxygenEmitter(const DoxygenEmitter &) = delete;
  DoxygenEmitter &operator=(const DoxygenEmitter &) = delete;

  ~DoxygenEmitter() {
    if (Configuration.CommentFooter) {
      if (not IsAtBeginningOfLine)
        EmitterT::emit(llvm::StringRef("\n"));
      EmitterT::emit(*Configuration.CommentFooter);
    }
  }

  void emit(llvm::StringRef Content) {
    if (not Content.empty()) {
      bool EmitLinePrefix = IsAtBeginningOfLine;

      for (auto Line : LineRange(Content)) {
        if (std::exchange(EmitLinePrefix, true))
          emitLinePrefix(Line == "\n");

        BaseEmitter::emit(Line);
      }

      IsAtBeginningOfLine = Content.back() == '\n';
    }
  }

  using BaseEmitter::indent;

private:
  void emitLinePrefix(bool IsEmptyLine) {
    llvm::StringRef Prefix = Configuration.LinePrefix;
    EmitterT::emit(IsEmptyLine ? Prefix.rtrim() : Prefix);
  }
};

} // namespace ptml
