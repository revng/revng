#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/LineRange.h"
#include "revng/PTML/Constants.h"

namespace ptml {

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  std::optional<llvm::StringRef> CommentHeader;
  std::optional<llvm::StringRef> CommentFooter;
  llvm::StringRef LinePrefix;
};

template<PTMLEmitter EmitterT>
class DoxygenEmitter : EmitterT {
  DoxygenCommentConfiguration Configuration;
  bool IsAtBeginningOfLine = true;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitter(const DoxygenCommentConfiguration &Configuration,
                          ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...), Configuration(Configuration) {
    if (Configuration.CommentHeader) {
      EmitterT::emit(*Configuration.CommentHeader);
      EmitterT::emit(llvm::StringRef("\n"));
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    auto Tag = EmitterT::initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::doxygen::tokens::Keyword);
    Tag.finalizeOpenTag();

    emit(llvm::StringRef(&Configuration.KeywordSignifier, 1));
    emit(Keyword);
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

        EmitterT::emit(Line);
      }

      IsAtBeginningOfLine = Content.back() == '\n';
    }
  }

private:
  void emitLinePrefix(bool IsEmptyLine) {
    llvm::StringRef Prefix = Configuration.LinePrefix;
    EmitterT::emit(IsEmptyLine ? Prefix.rtrim() : Prefix);
  }
};

} // namespace ptml
