#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"

namespace ptml {

struct DoxygenCommentConfiguration {
  char KeywordSignifier = '\\';
  std::optional<llvm::StringRef> CommentHeader;
  std::optional<llvm::StringRef> CommentFooter;
  llvm::StringRef LinePrefix;
};

namespace detail {

template<typename EmitterT>
class DoxygenEmitterBase : public EmitterT {
protected:
  DoxygenCommentConfiguration Configuration;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitterBase(const DoxygenCommentConfiguration &Configuration,
                              ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...), Configuration(Configuration) {}

  void emitIndentation(unsigned Indentation) {
    static constexpr llvm::StringRef IndentString = "  ";

    EmitterT::emit(Configuration.LinePrefix);
    for (unsigned I = 0; I < Indentation; ++I)
      EmitterT::emit(IndentString);
  }
};

} // namespace detail

template<PTMLEmitter EmitterT>
class DoxygenEmitter : IndentingEmitter<detail::DoxygenEmitterBase<EmitterT>> {
  using Base = detail::DoxygenEmitterBase<EmitterT>;
  using IndentingEmitter = IndentingEmitter<Base>;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit DoxygenEmitter(const DoxygenCommentConfiguration &Configuration,
                          ArgsT &&...Args) :
    IndentingEmitter(Configuration, std::forward<ArgsT>(Args)...) {
    if (Base::Configuration.CommentHeader) {
      EmitterT::emit(*Base::Configuration.CommentHeader);
      IndentingEmitter::emit("\n");
    }
  }

  void emitKeyword(llvm::StringRef Keyword) {
    auto Tag = EmitterT::initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::doxygen::tokens::Keyword);
    Tag.finalizeOpenTag();

    char Signifier = Base::Configuration.KeywordSignifier;
    IndentingEmitter::emit(llvm::StringRef(&Signifier, 1));
    IndentingEmitter::emit(Keyword);
  }

  DoxygenEmitter(const DoxygenEmitter &) = delete;
  DoxygenEmitter &operator=(const DoxygenEmitter &) = delete;

  ~DoxygenEmitter() {
    if (Base::Configuration.CommentFooter) {
      if (not IndentingEmitter::isAtBeginningOfLine())
        IndentingEmitter::emit("\n");
      EmitterT::emit(*Base::Configuration.CommentFooter);
    }
  }

  using IndentingEmitter::emit;
};

} // namespace ptml
