#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PTML/Emitter.h"
#include "revng/PTMLCFormat/Format.h"

namespace ptml {

enum class Tagging : bool {
  Disabled,
  Enabled,
};

class PTMLStreamEmitter;

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
/// care of finalizing the open tag if necessary, or early using closeTag().
///
/// At any given time, a PTMLEmitter may be associated with multiple tag
/// emitters but only the innermost can have an unfinalized open tag.
class PTMLTagEmitter {
  PTMLStreamEmitter &ParentEmitter;
  llvm::StringRef Tag;
  bool IsEmittingOpenTag = true;
  bool IsClosed = false;

  size_t StartOffset = 0;

public:
  explicit PTMLTagEmitter(PTMLStreamEmitter &ParentEmitter,
                          llvm::StringRef Tag);

  PTMLTagEmitter(const PTMLTagEmitter &) = delete;
  PTMLTagEmitter &operator=(const PTMLTagEmitter &) = delete;

  ~PTMLTagEmitter();

  PTMLTagEmitter &emitAttribute(llvm::StringRef Name, llvm::StringRef Value);
  PTMLTagEmitter &emitListAttribute(llvm::StringRef Name,
                                    llvm::ArrayRef<llvm::StringRef> Values);

  [[nodiscard]] bool isEmittingOpenTag() const { return IsEmittingOpenTag; }

  [[nodiscard]] bool isClosed() const { return IsClosed; }

  void finalizeOpenTag();

  /// Emits the close tag early, instead of leaving it to the destructor. The
  /// tag has to be the innermost one still open.
  void closeTag();

private:
  void emitAttributeValue(llvm::StringRef Value);
};

/// Provides a streaming interface for emitting PTML tags and content.
///
/// PTML tag emission is performed using a PTMLTagEmitter, which is an RAII type
/// guaranteeing emission of well-formed PTML tags. Tag content is emitted using
/// the Emitter interface. See the documentation of Emitter for more
/// information.
///
/// PTML tag emission can be toggled using the ptml::Tagging parameter. Note
/// that valid usage of the PTML tag emission interface is checked regardless
/// of whether PTML tag emission is enabled.
template<typename EmitterT>
concept PTMLEmitter = //
  Emitter<EmitterT> and requires(EmitterT &Emitter, llvm::StringRef String) {
    {
      Emitter.makeTagInitializer(String)
    } -> std::convertible_to<PTMLTagEmitter>;

    { Emitter.initializeOpenTag(String) } -> std::same_as<PTMLTagEmitter>;
  };

/// What a PTMLStreamEmitter writes to its stream.
enum class EmissionMode : uint8_t {
  /// Content only, with no tags.
  PlainText,

  /// Content wrapped in PTML tags.
  Tags,

  /// PTML tags, plus the metadata a downstream reformatter needs to reposition
  /// whitespace without re-lexing the emitted PTML.
  TagsAndMetadata,
};

/// Concrete PTMLEmitter using an underlying llvm::raw_ostream.
class PTMLStreamEmitter : StreamEmitter {
  friend PTMLTagEmitter;

  /// Lets a PTMLTagEmitter be constructed in place, so it need not be movable.
  class TagInitializer {
    PTMLStreamEmitter &Emitter;
    llvm::StringRef Tag;

  public:
    explicit TagInitializer(PTMLStreamEmitter &Emitter, llvm::StringRef Tag) :
      Emitter(Emitter), Tag(Tag) {}

    [[nodiscard]] operator PTMLTagEmitter() const {
      return PTMLTagEmitter(Emitter, Tag);
    }
  };

  EmissionMode Mode;
  const PTMLTagEmitter *CurrentOpenTagEmitter = nullptr;

  // In TagsAndMetadata mode, the metadata describing the emitted document is
  // built as it is written, so that reformatting can run without re-lexing it.
  PTMLCReformattableDocument DocumentMetadata;

public:
  explicit PTMLStreamEmitter(llvm::raw_ostream &OS, EmissionMode Mode) :
    StreamEmitter(OS), Mode(Mode) {}

  PTMLStreamEmitter(const PTMLStreamEmitter &) = delete;
  PTMLStreamEmitter &operator=(const PTMLStreamEmitter &) = delete;

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

  /// The metadata describing the document emitted so far. Only populated in
  /// TagsAndMetadata mode.
  const PTMLCReformattableDocument &metadata() const {
    return DocumentMetadata;
  }

private:
  [[nodiscard]] bool emitsTags() const {
    return Mode != EmissionMode::PlainText;
  }

  [[nodiscard]] bool buildsMetadata() const {
    return Mode == EmissionMode::TagsAndMetadata;
  }

  // Both record a tag spanning from Begin to the current position, and are
  // no-ops unless metadata is being built.
  void recordOpenTag(size_t Begin) {
    if (buildsMetadata()) {
      DocumentMetadata.Tags.push_back({ Begin,
                                        OS.tell(),
                                        /*IsClosing=*/false });
    }
  }

  void recordCloseTag(size_t Begin) {
    if (buildsMetadata()) {
      DocumentMetadata.Tags.push_back({ Begin,
                                        OS.tell(),
                                        /*IsClosing=*/true });
    }
  }

  /// Records a source-to-PTML sync point at the current position, unless it
  /// merely continues the previous run one-to-one.
  void recordSyncPoint();
};
static_assert(PTMLEmitter<PTMLStreamEmitter>);

} // namespace ptml
