#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/DoxygenEmitter.h"

namespace ptml {

using CDoxygenEmitter = DoxygenEmitter<CTokenEmitter::CommentEmitter>;

namespace detail {

inline constexpr DoxygenCommentConfiguration LineCommentConfiguration = {
  .LinePrefix = "/ "
};

inline constexpr DoxygenCommentConfiguration BlockCommentConfiguration = {
  .CommentHeader = "*",
  .CommentFooter = " ",
  .LinePrefix = " * ",
};

} // namespace detail

[[nodiscard]] inline CDoxygenEmitter emitDoxygenLineComment(CTokenEmitter &CE) {
  return CDoxygenEmitter(detail::LineCommentConfiguration,
                         CE,
                         CTokenEmitter::CommentKind::Line);
}

[[nodiscard]] inline CDoxygenEmitter
emitDoxygenBlockComment(CTokenEmitter &CE) {
  return CDoxygenEmitter(detail::BlockCommentConfiguration,
                         CE,
                         CTokenEmitter::CommentKind::Block);
}

} // namespace ptml
