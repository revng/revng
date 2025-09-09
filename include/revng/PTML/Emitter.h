#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"

namespace ptml {

enum class Tagging : bool {
  Disabled,
  Enabled,
};

class Emitter {
public:
  explicit Emitter(llvm::raw_ostream &OS, Tagging Tags) :
    OS(OS), EmitTags(Tags == Tagging::Enabled) {}

  [[nodiscard]] bool isTagged() const { return EmitTags; }

  // Emit a literal string. The string may not contain newlines or characters
  // requiring escape sequences.
  void emitLiteral(llvm::StringRef String);

  void emitNewline() {
    revng_assert(CurrentTagEmitter == nullptr);
    OS << '\n';
    HasTrailingNewline = true;
  }

  void emit(llvm::StringRef String);

  void indent(int Offset) {
    if (Offset < 0)
      revng_assert(Indent >= static_cast<unsigned>(-Offset));

    Indent += static_cast<unsigned>(Offset);
  }

  class TagEmitter {
  public:
    TagEmitter(decltype(nullptr)) : Emitter(nullptr) {}

    explicit TagEmitter(Emitter &Emitter, llvm::StringRef Tag) {
      initializeImpl(Emitter, Tag);
    }

    void initialize(Emitter &Emitter, llvm::StringRef Tag) {
      revng_assert(this->Emitter == nullptr);
      initializeImpl(Emitter, Tag);
    }

    TagEmitter(const TagEmitter &) = delete;
    TagEmitter &operator=(const TagEmitter &) = delete;

    ~TagEmitter() { close(); }

    void emitAttribute(llvm::StringRef Name, llvm::StringRef Value) {
      revng_assert(Emitter != nullptr);
      revng_assert(Emitter->CurrentTagEmitter == this);
      revng_assert(not IsFinalized);

      if (Emitter->EmitTags) {
        Emitter->OS << ' ' << Name << '=' << '"';
        Emitter->emitAttributeValue(Value);
        Emitter->OS << '"';
      }
    }

    void finalize() {
      revng_assert(Emitter != nullptr);

      if (not IsFinalized) {
        revng_assert(Emitter->CurrentTagEmitter == this);

        if (Emitter->EmitTags)
          Emitter->OS << '>';
        IsFinalized = true;

        Emitter->CurrentTagEmitter = nullptr;
      }
    }

    void close() {
      if (Emitter != nullptr) {
        finalize();

        if (Emitter->EmitTags)
          Emitter->OS << '<' << '/' << Tag << '>';
      }
      Emitter = nullptr;
    }

  private:
    void initializeImpl(Emitter &Emitter, llvm::StringRef Tag) {
      revng_assert(Emitter.CurrentTagEmitter == nullptr);

      this->Emitter = &Emitter;
      this->Tag = Tag;
      this->IsFinalized = false;

      if (Emitter.EmitTags) {
        if (Emitter.HasTrailingNewline)
          Emitter.emitIndent();
        Emitter.OS << '<' << Tag;
      }

      Emitter.CurrentTagEmitter = this;
    }

    Emitter *Emitter;
    llvm::StringRef Tag;
    bool IsFinalized;
  };

  TagEmitter enterTag(llvm::StringRef Tag) { return TagEmitter(*this, Tag); }

private:
  void emitLiteralImpl(llvm::StringRef String);

  void emitWithIndents(llvm::StringRef String);

  template<bool EscapeQuotes = false>
  void emitWithEscapes(llvm::StringRef String);

  void emitIndentWhitespace();
  void emitIndentWithTags();
  void emitIndent();

  void emitAttributeValue(llvm::StringRef String);

  llvm::raw_ostream &OS;
  const bool EmitTags;

  unsigned Indent = 0;
  bool HasTrailingNewline = true;

  const TagEmitter *CurrentTagEmitter = nullptr;
};

} // namespace ptml
