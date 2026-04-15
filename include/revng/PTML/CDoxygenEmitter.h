#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/DoxygenEmitter.h"

namespace ptml {

class CDoxygenEmitter : public DoxygenEmitter<CTokenEmitter::CommentEmitter> {
  using Base = DoxygenEmitter<CTokenEmitter::CommentEmitter>;

private:
  static constexpr DoxygenCommentConfiguration LineCommentConfiguration = {
    .LinePrefix = "/ "
  };

  static constexpr DoxygenCommentConfiguration BlockCommentConfiguration = {
    .CommentHeader = "*",
    .CommentFooter = " ",
    .LinePrefix = " * ",
  };

public:
  using Base::Base;

public:
  [[nodiscard]] static CDoxygenEmitter emitLineComment(CTokenEmitter &CE) {
    return CDoxygenEmitter(LineCommentConfiguration,
                           CE,
                           CTokenEmitter::CommentKind::Line);
  }

  [[nodiscard]] static CDoxygenEmitter emitBlockComment(CTokenEmitter &CE) {
    return CDoxygenEmitter(BlockCommentConfiguration,
                           CE,
                           CTokenEmitter::CommentKind::Block);
  }

public:
  // Taking target optional to emplace into is pretty awkward, but, to allow
  // conditional construction (as the constructor has non-trivial side-effects),
  // we have to resort to either having a dedicated factory OR exposing
  // a move-constructor.
  //
  // Which is to say, if in the future, we need something like this for more
  // than an `std::optional`, it's better to switch to something more general,
  // like a "super elider" instead of spawning more methods like this.
  static void emitLineComment(std::optional<CDoxygenEmitter> &EmplaceInto,
                              CTokenEmitter &CE) {
    EmplaceInto.emplace(LineCommentConfiguration,
                        CE,
                        CTokenEmitter::CommentKind::Line);
  }

  static void emitBlockComment(std::optional<CDoxygenEmitter> &EmplaceInto,
                               CTokenEmitter &CE) {
    EmplaceInto.emplace(BlockCommentConfiguration,
                        CE,
                        CTokenEmitter::CommentKind::Block);
  }
};

} // namespace ptml
