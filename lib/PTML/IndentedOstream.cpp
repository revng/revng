//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/ErrorHandling.h"

#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"

namespace ptml {

uint64_t PTMLIndentedOstream::current_pos() const {
  return OS.tell();
}

void PTMLIndentedOstream::write_impl(const char *Ptr, size_t Size) {
  llvm::StringRef Str(Ptr, Size);

  bool BufferEndsNewline = Str.endswith("\n");

  std::pair<llvm::StringRef, llvm::StringRef> Pair;
  while (Pair = Str.split('\n'), Pair.first != "") {
    if (TrailingNewline)
      writeIndent();
    OS << Pair.first;
    if (Pair.second != "") {
      OS << '\n';
      writeIndent();
    }
    Str = Pair.second;
  }

  if (BufferEndsNewline)
    OS << '\n';
  TrailingNewline = BufferEndsNewline;
}

void PTMLIndentedOstream::writeIndent() {
  if (IndentDepth > 0) {
    OS << Tag(tags::Span, std::string(IndentSize * IndentDepth, ' '))
            .addAttribute(attributes::Token, ptml::tokens::Indentation)
            .serialize();
  }
  TrailingNewline = false;
}

} // namespace ptml
