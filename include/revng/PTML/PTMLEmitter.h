#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/IndentingEmitter.h"

namespace ptml {

enum class Tagging : bool {
  Disabled,
  Enabled,
};

namespace detail {

class PTMLTagEmitterBase {};

class PTMLEmitterBase : public StreamEmitter {
protected:
  bool EmitTags = false;
  const PTMLTagEmitterBase *CurrentOpenTagEmitter = nullptr;

public:
  explicit PTMLEmitterBase(llvm::raw_ostream &OS, Tagging Tags) :
    StreamEmitter(OS), EmitTags(Tags == Tagging::Enabled) {}

  void emitIndentation(unsigned Indentation);
};

} // namespace detail

/// \brief Provides a streaming interface for emitting PTML tags and content.
///
/// Underlying byte-IO is done via the provided llvm::raw_ostream reference.
///
/// PTML tag emission is performed using PTMLEmitter::TagEmitter, which is an
/// RAII object guaranteeing emission of well-formed PTML. Tag content is
/// emitted using the PTMLEmitter interface. See the documentation of
/// PTMLEmitter::TagEmitter for more information.
///
/// PTML tag emission can be toggled using the ptml::Tagging parameter. Note
/// that valid usage of the PTML tag emission interface is checked regardless
/// of whether PTML tag emission is enabled.
class PTMLEmitter : IndentingEmitter<detail::PTMLEmitterBase> {
public:
  class TagEmitter;

  explicit PTMLEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    IndentingEmitter(OS, Tags) {}

  [[nodiscard]] bool isTagged() const { return EmitTags; }

  // Emit the specified content literally. The string shall not contain newlines
  // or characters requiring HTML escape sequences (<, >, &).
  void emitLiteralContent(llvm::StringRef String);

  void emitContentNewline() {
    revng_assert(CurrentOpenTagEmitter == nullptr,
                 "Content shall not emitted while an opening tag is "
                 "unfinalized.");
    return emitNewline();
  }

  void emitContent(llvm::StringRef String);

  using IndentingEmitter::indent;
  using IndentingEmitter::indentation;
  using IndentingEmitter::isAtBeginningOfLine;

  [[nodiscard]] TagEmitter initializeOpenTag(llvm::StringRef Tag);

private:
  template<bool EscapeQuotes = false>
  void emitEscapedContent(llvm::StringRef String);

  void emitAttributeValue(llvm::StringRef String);
};

/// \brief RAII object used for emitting PTML tags.
///
/// TagEmitter has three states:
/// 1. Uninitialized:
///    The TagEmitter has been default-constructed or initialised and
///    subsequently closed.
///
/// 2. Initialized:
///    Emission of the opening tag has been started, but not yet completed. It
///    is only in this state that emission of attributes is possible, while at
///    the same time emission of content via the associated ptml::PTMLEmitter is
///    disallowed.
///
///    This state can be entered using the initializeOpenTag member functions
///    of either PTMLEmitter or TagEmitter, or by constructing a TagEmitter
///    using its non-default constructor.
///
///    PTMLEmitter::initializeOpenTag is a convenience function which returns a
///    TagEmitter in the initialized state.
///
/// 3. Finalized:
///    Emission of the opening tag has been completed, but the closing tag has
///    not yet been emitted. In this state tag content can be emitted via the
///    associated ptml::PTMLEmitter.
///
///    This state is entered using the finalizeOpenTag member function.
///
/// The closing tag is emitted by explicitly calling the close member function,
/// or implicitly by the destructor, which also takes care of finalizing the
/// open tag if necessary.
///
/// At any given time, the emitter may be associated with multiple TagEmitters
/// but only the innermost can have an unfinalized open tag.
class PTMLEmitter::TagEmitter : public detail::PTMLTagEmitterBase {
  PTMLEmitter *ParentEmitter;
  llvm::StringRef Tag;
  bool IsOpenTagFinalized = false;

public:
  TagEmitter() : ParentEmitter(nullptr) {}

  explicit TagEmitter(PTMLEmitter &ParentEmitter, llvm::StringRef Tag) {
    revng_assert(ParentEmitter.CurrentOpenTagEmitter == nullptr,
                 "The parent PTMLEmitter is already associated with an "
                 "unfinalized TagEmitter.");

    initializeOpenTagImpl(ParentEmitter, Tag);
  }

  TagEmitter(const TagEmitter &) = delete;
  TagEmitter &operator=(const TagEmitter &) = delete;

  ~TagEmitter() { closeImpl(); }

  TagEmitter &initializeOpenTag(PTMLEmitter &ParentEmitter,
                                llvm::StringRef Tag) & {
    revng_assert(this->ParentEmitter == nullptr,
                 "This TagEmitter was already initialized.");
    revng_assert(ParentEmitter.CurrentOpenTagEmitter == nullptr,
                 "The parent PTMLEmitter is already associated with an "
                 "unfinalized TagEmitter.");

    initializeOpenTagImpl(ParentEmitter, Tag);
    return *this;
  }

  TagEmitter &emitAttribute(llvm::StringRef Name, llvm::StringRef Value) & {
    revng_assert(ParentEmitter != nullptr,
                 "This TagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This TagEmitter has already been finalized.");

    emitAttributeImpl(Name, Value);
    return *this;
  }

  TagEmitter &emitListAttribute(llvm::StringRef Name,
                                llvm::ArrayRef<llvm::StringRef> Values) & {
    revng_assert(ParentEmitter != nullptr,
                 "This TagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This TagEmitter has already been finalized.");

    emitListAttributeImpl(Name, Values);
    return *this;
  }

  [[nodiscard]] bool isOpenTagFinalized() const { return IsOpenTagFinalized; }

  void finalizeOpenTag() {
    revng_assert(ParentEmitter != nullptr,
                 "This TagEmitter has not been initialized.");
    revng_assert(not IsOpenTagFinalized,
                 "This TagEmitter has already been finalized.");
    finalizeOpenTagImpl();
  }

  [[nodiscard]] bool isOpen() const { return ParentEmitter != nullptr; }

  void close() {
    revng_assert(ParentEmitter != nullptr,
                 "This TagEmitter has not been initialized.");
    closeImpl();
  }

private:
  void initializeOpenTagImpl(PTMLEmitter &ParentEmitter, llvm::StringRef Tag);

  void emitAttributeImpl(llvm::StringRef Name, llvm::StringRef Value);
  void emitListAttributeImpl(llvm::StringRef Name,
                             llvm::ArrayRef<llvm::StringRef> Values);

  void finalizeOpenTagImpl();
  void closeImpl();
};

inline PTMLEmitter::TagEmitter
PTMLEmitter::initializeOpenTag(llvm::StringRef Tag) {
  return TagEmitter(*this, Tag);
}

using PTMLTagEmitter = PTMLEmitter::TagEmitter;

} // namespace ptml
