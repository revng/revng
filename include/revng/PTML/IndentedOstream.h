#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

namespace ptml {

class PTMLIndentedOstream : public llvm::raw_ostream {
private:
  int IndentSize;
  int IndentDepth;
  // If the buffer ends with a newline, we want to delay emitting indentation on
  // the next character, so that we can account for (de)indentations that could
  // happen in the meantime. This boolean tracks if we last read a newline
  // character.
  bool TrailingNewline;
  raw_ostream &OS;

public:
  explicit PTMLIndentedOstream(llvm::raw_ostream &OS, int IndentSize = 2) :
    IndentSize(IndentSize), IndentDepth(0), TrailingNewline(false), OS(OS) {
    SetUnbuffered();
  }

  struct Scope {
  private:
    PTMLIndentedOstream &OS;

  public:
    Scope(PTMLIndentedOstream &OS) : OS(OS) { OS.indent(); }

    ~Scope() { OS.unindent(); }
  };

  Scope scope() { return Scope(*this); }

  void indent() { IndentDepth = std::min(INT_MAX, IndentDepth + 1); }
  void unindent() { IndentDepth = std::max(0, IndentDepth - 1); }

private:
  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override;
  void writeIndent();
};

} // namespace ptml
