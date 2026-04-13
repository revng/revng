#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/LineRange.h"
#include "revng/PTML/Emitter.h"
#include "revng/Support/Assert.h"

namespace ptml {

struct IndentString : llvm::StringRef {
  explicit IndentString(llvm::StringRef String) : llvm::StringRef(String) {}
};

template<Emitter EmitterT>
class IndentingEmitter : protected EmitterT {
  llvm::StringRef IndentationString;
  unsigned Indentation = 0;
  bool IsAtBeginningOfLine = true;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit IndentingEmitter(ArgsT &&...Args) :
    IndentingEmitter(IndentString("  "), std::forward<ArgsT>(Args)...) {}

  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit IndentingEmitter(IndentString Indent, ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...), IndentationString(Indent) {}

  void indent(int Offset) {
    revng_assert(Offset >= 0 or static_cast<unsigned>(-Offset) <= Indentation,
                 "Offset would result in negative indentation.");

    Indentation += static_cast<unsigned>(Offset);
  }

  [[nodiscard]] unsigned indentation() const { return Indentation; }

  void emit(llvm::StringRef String) {
    if (not String.empty()) {
      bool EmitIndent = IsAtBeginningOfLine;

      for (auto Line : LineRange(String)) {
        if (std::exchange(EmitIndent, true))
          emitIndentation();

        if (not Line.empty())
          EmitterT::emit(Line);
      }

      IsAtBeginningOfLine = String.back() == '\n';
    }
  }

  void emitNewline() {
    EmitterT::emit(llvm::StringRef("\n"));
    IsAtBeginningOfLine = true;
  }

protected:
  void emitIndentationIfNeeded() {
    if (IsAtBeginningOfLine) {
      IsAtBeginningOfLine = false;
      emitIndentation();
    }
  }

private:
  void emitIndentation() {
    for (unsigned I = 0; I < Indentation; ++I)
      EmitterT::emit(IndentationString);
  }
};

} // namespace ptml
