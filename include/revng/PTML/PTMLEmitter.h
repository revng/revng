#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"

#include "revng/PTML/Emitter.h"
#include "revng/PTML/IndentingEmitter.h"

namespace ptml {

enum class Tagging : bool {
  Disabled,
  Enabled,
};

class PTMLTagEmitter;

namespace detail {

class PTMLEmitterBase : public StreamEmitter {
  friend PTMLTagEmitter;

protected:
  bool EmitTags = false;
  const PTMLTagEmitter *CurrentOpenTagEmitter = nullptr;

public:
  explicit PTMLEmitterBase(llvm::raw_ostream &OS, Tagging Tags) :
    StreamEmitter(OS), EmitTags(Tags == Tagging::Enabled) {}
};

} // namespace detail

/// Provides a streaming interface for emitting PTML tags and content.
///
/// PTML tag emission is performed using a PTMLTagEmitter, which is an RAII type
/// guaranteeing emission of well-formed PTML tags. Tag content is emitted using
// the Emitter interface. See the documentation of Emitter for more information.
///
/// PTML tag emission can be toggled using the ptml::Tagging parameter. Note
/// that valid usage of the PTML tag emission interface is checked regardless
/// of whether PTML tag emission is enabled.
template<typename EmitterT>
concept PTMLEmitter = //
  Emitter<EmitterT> and requires(EmitterT &Emitter, llvm::StringRef String) {
    // auto makeTagInitializer(llvm::String String);
    {
      Emitter.makeTagInitializer(String)
    } -> std::convertible_to<PTMLTagEmitter>;

    // PTMLTagEmitter initializeOpenTag(llvm::StringRef String);
    { Emitter.initializeOpenTag(String) } -> std::same_as<PTMLTagEmitter>;
  };

/// RAII type used for emitting PTML tags.
///
/// PTMLTagEmitter has two states:
/// 1. Emitting the open tag:
///    Emission of the opening tag has been started, but not yet completed. It
///    is only in this state that emission of attributes is possible, while at
///    the same time emission of content via the associated emitter is
///    disallowed. This state is entered upon construction.
///
/// 2. Emitting tag content:
///    Emission of the opening tag has been completed, but the closing tag has
///    not yet been emitted. In this state tag content can be emitted via the
///    associated emitter. This state is entered using the finalizeOpenTag
///    member function.
///
/// The closing tag is emitted implicitly by the destructor, which also takes
/// care of finalizing the open tag if necessary.
///
/// At any given time, a PTMLEmitter may be associated with multiple tag
/// emitters but only the innermost can have an unfinalized open tag.
class PTMLTagEmitter {
  detail::PTMLEmitterBase &ParentEmitter;
  llvm::StringRef Tag;
  bool IsEmittingOpenTag = true;

public:
  explicit PTMLTagEmitter(detail::PTMLEmitterBase &ParentEmitter,
                          llvm::StringRef Tag);

  PTMLTagEmitter(const PTMLTagEmitter &) = delete;
  PTMLTagEmitter &operator=(const PTMLTagEmitter &) = delete;

  ~PTMLTagEmitter();

  PTMLTagEmitter &emitAttribute(llvm::StringRef Name, llvm::StringRef Value);
  PTMLTagEmitter &emitListAttribute(llvm::StringRef Name,
                                    llvm::ArrayRef<llvm::StringRef> Values);

  [[nodiscard]] bool isEmittingOpenTag() const { return IsEmittingOpenTag; }

  void finalizeOpenTag();

private:
  void emitAttributeValue(llvm::StringRef Value);
};

/// Concrete PTMLEmitter using an underlying llvm::raw_ostream.
class PTMLStreamEmitter : IndentingEmitter<detail::PTMLEmitterBase> {
  class TagInitializer {
    PTMLStreamEmitter &Emitter;
    llvm::StringRef Tag;

  public:
    explicit TagInitializer(PTMLStreamEmitter &Emitter, llvm::StringRef Tag) :
      Emitter(Emitter), Tag(Tag) {}

    [[nodiscard]] operator PTMLTagEmitter() const {
      Emitter.emitIndentationIfNeeded();
      return PTMLTagEmitter(Emitter, Tag);
    }
  };

public:
  explicit PTMLStreamEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    IndentingEmitter(OS, Tags) {}

  PTMLStreamEmitter(const PTMLStreamEmitter &) = delete;
  PTMLStreamEmitter &operator=(const PTMLStreamEmitter &) = delete;

  using IndentingEmitter::indent;
  using IndentingEmitter::indentation;

  /// Returns an initializer object which can be used for delayed initialization
  /// of a PTMLTagEmitter.
  [[nodiscard]] TagInitializer makeTagInitializer(llvm::StringRef Tag) {
    return TagInitializer(*this, Tag);
  }

  /// Convenience function for PTMLTagEmitter initialization.
  ///
  /// \note It is important that this function remains a simple wrapper around
  ///       the makeTagInitializer and does nothing more than that. This ensures
  ///       that delayed PTMLTagEmitter initialization remains valid.
  [[nodiscard]] PTMLTagEmitter initializeOpenTag(llvm::StringRef Tag) {
    return makeTagInitializer(Tag);
  }

  void emit(llvm::StringRef Content);
};
static_assert(PTMLEmitter<PTMLStreamEmitter>);

} // namespace ptml
