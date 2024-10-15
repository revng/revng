//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/ErrorHandling.h"

#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"

namespace ptml {

uint64_t IndentedOstream::current_pos() const {
  return OS.tell();
}

void IndentedOstream::write_impl(const char *Ptr, size_t Size) {
  llvm::StringRef Str(Ptr, Size);
  llvm::SmallVector<llvm::StringRef> Lines;
  Str.split(Lines, '\n');

  bool EndsInNewLine = Str.endswith("\n");
  if (EndsInNewLine)
    Lines.pop_back();

  if (TrailingNewline)
    writeIndent();

  for (auto &Line : llvm::make_range(Lines.begin(), std::prev(Lines.end()))) {
    OS << Line << '\n';
    writeIndent();
  }

  OS << Lines.pop_back_val();
  if (EndsInNewLine)
    OS << '\n';

  TrailingNewline = EndsInNewLine;
}

void IndentedOstream::writeIndent() {
  if (IndentDepth > 0) {
    Tag IndentTag = B.getTag(tags::Span,
                             std::string(IndentSize * IndentDepth, ' '));

    if (not B.IsInTaglessMode)
      IndentTag.addAttribute(attributes::Token, ptml::tokens::Indentation);

    OS << IndentTag.toString();
  }
  TrailingNewline = false;
}

} // namespace ptml
