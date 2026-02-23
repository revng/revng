#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/STLExtras.h"

#include "revng/PTML/Emitter.h"
#include "revng/Support/Assert.h"

namespace ptml {

template<typename EmitterT>
concept IndentableEmitter = //
  Emitter<EmitterT> and requires(EmitterT &Emitter) {
    Emitter.emitIndentation(static_cast<unsigned>(0));
    Emitter.emitEmptyLine();
  };

template<IndentableEmitter EmitterT>
class IndentingEmitter : protected EmitterT {
  unsigned Indentation = 0;
  bool IsAtBeginningOfLine = true;

public:
  template<typename... ArgsT>
    requires std::constructible_from<EmitterT, ArgsT...>
  explicit IndentingEmitter(ArgsT &&...Args) :
    EmitterT(std::forward<ArgsT>(Args)...) {}

  void indent(int Offset) {
    revng_assert(Offset >= 0 or static_cast<unsigned>(-Offset) <= Indentation,
                 "Offset would result in negative indentation.");

    Indentation += static_cast<unsigned>(Offset);
  }

  [[nodiscard]] unsigned indentation() const { return Indentation; }

  [[nodiscard]] bool isAtBeginningOfLine() const { return IsAtBeginningOfLine; }

  void emit(llvm::StringRef String) {
    if (not String.empty()) {
      for (auto [I, R] : llvm::enumerate(std::views::split(String, '\n'))) {
        llvm::StringRef Line = std::string_view(R.begin(), R.end());

        if (I != 0)
          emitNewline();

        if (not Line.empty()) {
          emitIndentationIfNeeded();
          EmitterT::emit(Line);
        }
      }

      IsAtBeginningOfLine = String.back() == '\n';
    }
  }

  void emitSpace() { EmitterT::emit(llvm::StringRef(" ")); }

  void emitNewline() {
    if (IsAtBeginningOfLine)
      EmitterT::emitEmptyLine();
    EmitterT::emit(llvm::StringRef("\n"));
    IsAtBeginningOfLine = true;
  }

protected:
  void emitIndentationIfNeeded() {
    if (IsAtBeginningOfLine) {
      IsAtBeginningOfLine = false;
      EmitterT::emitIndentation(Indentation);
    }
  }
};

} // namespace ptml
