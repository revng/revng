//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/Annotations.h"

ptml::CTokenEmitter::Scope
mlir::clift::emitHeaderPrologue(ptml::CTokenEmitter &PTML) {
  ptml::CTokenEmitter::Scope
    Scope = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::Header,
                            ptml::CTokenEmitter::Delimiter::None,
                            0);

  PTML.emitPragmaOnceDirective();
  PTML.emitNewline();

  PTML.emitComment("This header has been generated using rev.ng.",
                   ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();

  // TODO: emit the license information, revng version information, etc.

  return Scope;
}

ptml::CTokenEmitter::Scope
mlir::clift::emitHeaderPrologue(mlir::clift::CEmitter &Emitter) {
  return emitHeaderPrologue(Emitter.tokenEmitter());
}

struct AttributeScopeHelper {
  ptml::CTokenEmitter &PTML;
  bool IsReal = false;

  AttributeScopeHelper(ptml::CTokenEmitter &PTML, bool IsReal = false) :
    PTML(PTML), IsReal(IsReal) {

    PTML.emitSpace();
    PTML.emitMacro("__attribute__");
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);

    if (!IsReal) {
      PTML.emitMacro("annotate");
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
    }
  }
  ~AttributeScopeHelper() {
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
    if (!IsReal)
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);

    PTML.emitNewline();
  }
};

// TODO: ensure name doesn't collide with anything.
static constexpr llvm::StringRef STR_HELPER_MACRO = "REVNG_STR";

void mlir::clift::emitAttributes(ptml::CTokenEmitter &PTML) {
  // Provide a way to disable attributes
  {
    auto Guard = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::MacroIfDef,
                                 ptml::CTokenEmitter::Delimiter::None,
                                 0);

    PTML.emitMacro("DISABLE_ATTRIBUTES");
    PTML.emitNewline();

    // Intentionally keeping this as a reference, despite it being a definition.
    PTML.emitDefineDirective("__attribute__",
                             ptml::CTokenEmitter::IdentifierKind::Reference);

    PTML.emitNewline();
  }

  // Emit a helper macro
  PTML.emitNewline();
  PTML.emitDefineDirective(STR_HELPER_MACRO);
  PTML.emitSpace();
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument(STR_HELPER_MACRO,
                         "Value",
                         ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitOperator(ptml::CTokenEmitter::Operator::Hash);
  PTML.emitMacroArgument(STR_HELPER_MACRO, "Value");
  PTML.emitNewline();
  PTML.emitNewline();

  // Emit attributes and annotations
  PTML.emitComment("clang-format off");
  PTML.emitNewline();

  ptml::Attributes.forEachAttribute([&PTML](const auto &Attribute) {
    PTML.emitDefineDirective(Attribute.Macro);
    auto Scope = AttributeScopeHelper(PTML, Attribute.IsReal);
    if (!Attribute.IsReal)
      PTML.emitStringLiteral(Attribute.Value);
    else
      PTML.emitStringLiteralImpl(Attribute.Value, "");
  });

  PTML.emitNewline();

  ptml::Attributes.forEachAnnotation([&PTML](const auto &Annotation) {
    PTML.emitDefineDirective(Annotation.Macro);
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
    PTML.emitMacroArgument(Annotation.Macro,
                           "Value",
                           ptml::CTokenEmitter::IdentifierKind::Definition);
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
    auto Scope = AttributeScopeHelper(PTML);
    PTML.emitMacro(STR_HELPER_MACRO);
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
    PTML.emitStringLiteralImpl(Annotation.Prefix, "");
    PTML.emitMacroArgument(Annotation.Macro, "Value");
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  });

  PTML.emitNewline();

  PTML.emitComment("clang-format on");
  PTML.emitNewline();
}
