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
};

} // namespace ptml
