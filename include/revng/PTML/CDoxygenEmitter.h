#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/DoxygenEmitter.h"

namespace ptml {

using CDoxygenEmitter = DoxygenEmitter<CTokenEmitter::CommentEmitter>;

[[nodiscard]] inline CDoxygenEmitter emitDoxygenLineComment(CTokenEmitter &CE) {
  return CDoxygenEmitter({ .LinePrefix = "/ " },
                         CE,
                         CTokenEmitter::CommentKind::Line);
}

[[nodiscard]] inline CDoxygenEmitter
emitDoxygenBlockComment(CTokenEmitter &CE) {
  DoxygenCommentConfiguration Configuration = {
    .CommentHeader = "*",
    .CommentFooter = " ",
    .LinePrefix = " * ",
  };

  return CDoxygenEmitter(Configuration, CE, CTokenEmitter::CommentKind::Block);
}

} // namespace ptml
