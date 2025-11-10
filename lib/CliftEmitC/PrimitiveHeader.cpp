//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftEmitC/Headers.h"
#include "revng/PTML/CTokenEmitter.h"

static void emitSanityAssert(ptml::CTokenEmitter &PTML,
                             llvm::StringRef LeftMacro,
                             llvm::StringRef RightMacro) {
  PTML.emitMacro("_Static_assert");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacro(LeftMacro);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitMacro(RightMacro);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Comma);
  PTML.emitStringLiteral(LeftMacro.str() + " != " + RightMacro.str());
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitPrimitiveTypedefImpl(ptml::CTokenEmitter &PTML,
                                     llvm::StringRef TypedefFrom,
                                     llvm::StringRef TypedefTo) {
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();
  PTML.emitPrimitive(TypedefFrom);
  PTML.emitSpace();
  PTML.emitPrimitive(TypedefTo);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emit16ByteGeneric(ptml::CTokenEmitter &PTML,
                              model::PrimitiveKind::Values Kind) {
  auto Guard = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::MacroIfDef,
                               ptml::CTokenEmitter::Delimiter::None,
                               0);

  PTML.emitMacro("__SIZEOF_INT128__");
  PTML.emitNewline();
  llvm::StringRef TypedefFrom = Kind == model::PrimitiveKind::Signed ?
                                  "__int128" :
                                  "unsigned __int128";
  emitPrimitiveTypedefImpl(PTML,
                           TypedefFrom,
                           model::PrimitiveType::getCName(Kind, 16));
}

static void emitFakePrimitiveStruct(ptml::CTokenEmitter &PTML, size_t Size) {
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Struct);
  PTML.emitSpace();

  auto Guard = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::StructDefinition,
                               ptml::CTokenEmitter::Delimiter::Braces);
  PTML.emitNewline();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Char);
  PTML.emitSpace();
  PTML.emitIdentifier("data",
                      "",
                      ptml::CTokenEmitter::EntityKind::Field,
                      ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftBracket);
  PTML.emitSignedIntegerLiteral(Size);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightBracket);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emit10Or12ByteGeneric(ptml::CTokenEmitter &PTML, size_t Size) {
  auto Guard = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::MacroIf,
                               ptml::CTokenEmitter::Delimiter::None,
                               0);

  PTML.emitMacro("__SIZEOF_LONG_DOUBLE__");
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(Size);
  PTML.emitNewline();

  namespace PK = model::PrimitiveKind;
  emitPrimitiveTypedefImpl(PTML,
                           "long double",
                           model::PrimitiveType::getCName(PK::Generic, Size));

  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Else);
  PTML.emitNewline();
  emitFakePrimitiveStruct(PTML, Size);
  PTML.emitSpace();
  PTML.emitPrimitive(model::PrimitiveType::getCName(PK::Generic, Size));
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitPrimitiveTypedef(ptml::CTokenEmitter &PTML,
                                 model::PrimitiveKind::Values Kind,
                                 size_t Size) {
  if (Size == 16) {
    PTML.emitNewline();
    emit16ByteGeneric(PTML, Kind);

  } else if (Size == 10 || Size == 12) {
    revng_assert(Kind == model::PrimitiveKind::Generic);
    PTML.emitNewline();
    emit10Or12ByteGeneric(PTML, Size);

  } else {
    namespace PK = model::PrimitiveKind;
    emitPrimitiveTypedefImpl(PTML,
                             model::PrimitiveType::getCName(PK::Unsigned, Size),
                             model::PrimitiveType::getCName(Kind, Size));
  }
}

static void emit16ByteFloatImpl(ptml::CTokenEmitter &PTML) {
  PTML.emitMacro("__SIZEOF_LONG_DOUBLE__");
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(16);
  PTML.emitNewline();

  namespace PK = model::PrimitiveKind;
  emitPrimitiveTypedefImpl(PTML,
                           "long double",
                           model::PrimitiveType::getCName(PK::Float, 16));

  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Else);
  PTML.emitNewline();

  {
    auto InnerG = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::MacroIfDef,
                                  ptml::CTokenEmitter::Delimiter::None,
                                  0);

    PTML.emitMacro("__FLT128_MIN__");
    PTML.emitNewline();

    emitPrimitiveTypedefImpl(PTML,
                             "_Float128",
                             model::PrimitiveType::getCName(PK::Float, 16));

    PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Else);
    PTML.emitNewline();

    emitFakePrimitiveStruct(PTML, 16);
    PTML.emitSpace();
    PTML.emitPrimitive(model::PrimitiveType::getCName(PK::Float, 16));
    PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
    PTML.emitNewline();
  }
}

static void emit2ByteFloatImpl(ptml::CTokenEmitter &PTML) {
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacro("__ARM_FP16_ARGS");
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(1);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::PipePipe);
  PTML.emitSpace();
  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Defined);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacro("__FLT16_MIN__");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitBackslash();
  PTML.indent(1);
  PTML.emitNewline();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::AmpersandAmpersand);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::Exclaim);
  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Defined);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacro("DISABLE_FLOAT16");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.indent(-1);
  PTML.emitNewline();

  namespace PK = model::PrimitiveKind;
  emitPrimitiveTypedefImpl(PTML,
                           "_Float16",
                           model::PrimitiveType::getCName(PK::Float, 2));

  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Else);
  PTML.emitNewline();

  emitFakePrimitiveStruct(PTML, 2);
  PTML.emitSpace();
  PTML.emitPrimitive(model::PrimitiveType::getCName(PK::Float, 2));
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitFloatImpl(ptml::CTokenEmitter &PTML,
                          llvm::StringRef MacroToCheck,
                          llvm::StringRef TypeToUse,
                          uint64_t Size) {
  PTML.emitMacro(MacroToCheck);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(Size);
  PTML.emitNewline();

  namespace PK = model::PrimitiveKind;
  emitPrimitiveTypedefImpl(PTML,
                           TypeToUse,
                           model::PrimitiveType::getCName(PK::Float, Size));

  PTML.emitDirective(ptml::CTokenEmitter::PreprocessorDirective::Else);
  PTML.emitNewline();
  emitFakePrimitiveStruct(PTML, Size);
  PTML.emitSpace();
  PTML.emitPrimitive(model::PrimitiveType::getCName(PK::Float, Size));
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitFloatTypedef(ptml::CTokenEmitter &PTML, size_t Size) {
  PTML.emitNewline();
  auto Guard = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::MacroIf,
                               ptml::CTokenEmitter::Delimiter::None,
                               0);

  if (Size == 16)
    emit16ByteFloatImpl(PTML);

  else if (Size == 12)
    emitFloatImpl(PTML, "__SIZEOF_LONG_DOUBLE__", "long double", Size);

  else if (Size == 10)
    emitFloatImpl(PTML, "__SIZEOF_LONG_DOUBLE__", "long double", Size);

  else if (Size == 8)
    emitFloatImpl(PTML, "__SIZEOF_DOUBLE__", "double", Size);

  else if (Size == 4)
    emitFloatImpl(PTML, "__SIZEOF_FLOAT__", "float", Size);

  else if (Size == 2)
    emit2ByteFloatImpl(PTML);

  else
    revng_abort("Unsupported float size.");
}

static void emitPrimitiveType(ptml::CTokenEmitter &PTML,
                              model::PrimitiveKind::Values Kind,
                              uint64_t Size) {
  switch (Kind) {
  case model::PrimitiveKind::Unsigned:
    if (Size == 16) {
      PTML.emitComment("Smaller sizes are already present in "
                       "`stdint.h`");
      emitPrimitiveTypedef(PTML, Kind, Size);
    }
    break;

  case model::PrimitiveKind::Signed:
    if (Size == 16) {
      PTML.emitComment("Smaller sizes are already present in "
                       "`stdint.h`");
      emitPrimitiveTypedef(PTML, Kind, Size);
    }
    break;

  case model::PrimitiveKind::Generic:
    emitPrimitiveTypedef(PTML, Kind, Size);
    break;
  case model::PrimitiveKind::PointerOrNumber:
    emitPrimitiveTypedef(PTML, Kind, Size);
    break;
  case model::PrimitiveKind::Number:
    emitPrimitiveTypedef(PTML, Kind, Size);
    break;

  case model::PrimitiveKind::Float:
    emitFloatTypedef(PTML, Size);
    break;

  default:
    revng_abort(("Unknown `PrimitiveKind`: " + toString(Kind)).c_str());
  }
}

constexpr llvm::StringRef StaticAssertHelperMacro = "static_assert_size";
static void emitSizeStaticAssertHelper(ptml::CTokenEmitter &PTML) {
  PTML.emitDefineDirective(StaticAssertHelperMacro);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument(StaticAssertHelperMacro,
                         "TYPE",
                         ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Comma);
  PTML.emitSpace();
  PTML.emitMacroArgument(StaticAssertHelperMacro,
                         "EXPECTED_SIZE",
                         ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitBackslash();
  PTML.indent(1);
  PTML.emitNewline();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Char);
  PTML.emitSpace();

  // TODO: prevent this name from being used elsewhere.
  PTML.emitIdentifier("revng_static_assertion_",
                      "",
                      ptml::CTokenEmitter::EntityKind::Typedef,
                      ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::HashHash);
  PTML.emitSpace();
  PTML.emitMacroArgument(StaticAssertHelperMacro, "TYPE");
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::HashHash);
  PTML.emitSpace();
  PTML.emitMacroArgument(StaticAssertHelperMacro, "EXPECTED_SIZE");
  PTML.emitSpace();
  PTML.emitBackslash();
  PTML.indent(1);
  PTML.emitNewline();
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftBracket);
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Sizeof);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument(StaticAssertHelperMacro, "TYPE");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::EqualsEquals);
  PTML.emitSpace();
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument(StaticAssertHelperMacro, "EXPECTED_SIZE");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::Question);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(1);
  PTML.emitSpace();
  PTML.emitOperator(ptml::CTokenEmitter::Operator::Colon);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(-1);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightBracket);
  PTML.indent(-2);
  PTML.emitNewline();
}

static void emitPrimitiveTypeSizeAssertion(ptml::CTokenEmitter &PTML,
                                           model::PrimitiveKind::Values Kind,
                                           uint64_t Size) {
  bool NeedsGuard = Kind != model::PrimitiveKind::Float && Size == 16;

  using MaybeScope = std::optional<ptml::CTokenEmitter::Scope>;
  auto Guard = NeedsGuard ?
                 MaybeScope(std::in_place,
                            PTML,
                            ptml::CTokenEmitter::ScopeKind::MacroIfDef,
                            ptml::CTokenEmitter::Delimiter::None,
                            0) :
                 MaybeScope(std::nullopt);

  if (NeedsGuard) {
    PTML.emitSpace();
    PTML.emitMacro("__SIZEOF_INT128__");
    PTML.emitNewline();
  }

  PTML.emitMacro(StaticAssertHelperMacro);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitPrimitive(model::PrimitiveType::getCName(Kind, Size));
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Comma);
  PTML.emitSpace();
  PTML.emitSignedIntegerLiteral(Size);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitSizedPointerMacro(ptml::CTokenEmitter &PTML, uint64_t Size) {
  PTML.emitDefineDirective("pointer" + std::to_string(Size * 8) + "_t");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument("pointer" + std::to_string(Size * 8) + "_t",
                         "T",
                         ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();

  static constexpr auto UnsignedKind = model::PrimitiveKind::Unsigned;
  PTML.emitPrimitive(model::PrimitiveType::getCName(UnsignedKind, 2));
  PTML.emitNewline();
}

static void emitUndefMacros(ptml::CTokenEmitter &PTML) {
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Extern);
  PTML.emitSpace();
  PTML.emitPrimitive("uintmax_t");
  PTML.emitSpace();
  PTML.emitIdentifier("undef_value",
                      "",
                      ptml::CTokenEmitter::EntityKind::Function,
                      ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitPrimitive("void");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
  PTML.emitNewline();

  PTML.emitDefineDirective("undef");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument("undef",
                         "T",
                         ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitMacroArgument("undef", "T");
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitSpace();
  PTML.emitIdentifier("undef_value",
                      "",
                      ptml::CTokenEmitter::EntityKind::Function,
                      ptml::CTokenEmitter::IdentifierKind::Reference);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightParenthesis);
  PTML.emitNewline();
  PTML.emitNewline();
}

void mlir::clift::emitPrimitiveTypes(ptml::CTokenEmitter &PTML) {
  // Includes
  PTML.emitIncludeDirective("limits.h",
                            "",
                            ptml::CTokenEmitter::IncludeMode::Angle);
  PTML.emitIncludeDirective("stdint.h",
                            "",
                            ptml::CTokenEmitter::IncludeMode::Angle);
  PTML.emitNewline();

  // Sanity asserts
  emitSanityAssert(PTML, "CHAR_MIN", "SCHAR_MIN");
  emitSanityAssert(PTML, "CHAR_MAX", "SCHAR_MAX");
  emitSanityAssert(PTML, "CHAR_MIN", "INT8_MIN");
  emitSanityAssert(PTML, "CHAR_MAX", "INT8_MAX");

  // Primitives
  constexpr uint64_t
    PrimitiveKindCount = EnumElementsCount<model::PrimitiveKind::Values>;
  compile_time::repeat<PrimitiveKindCount - 1>([&PTML]<size_t Index>() {
    auto CurrentKind = model::PrimitiveKind::Values(Index + 1);
    if (CurrentKind == model::PrimitiveKind::Void)
      return;

    PTML.emitNewline();
    PTML.emitComment(toString(CurrentKind),
                     ptml::CTokenEmitter::CommentKind::Category);
    PTML.emitNewline();

    if (CurrentKind == model::PrimitiveKind::Generic)
      for (uint64_t Size : model::PrimitiveType::validSizes(CurrentKind))
        emitPrimitiveType(PTML, CurrentKind, Size);

    else
      for (uint64_t Size : model::PrimitiveType::staticValidSizes(CurrentKind))
        emitPrimitiveType(PTML, CurrentKind, Size);
  });
  PTML.emitNewline();

  // Assert the sizes are what they should be.
  PTML.emitComment("Assert the sizes are what they should be.",
                   ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();
  emitSizeStaticAssertHelper(PTML);
  PTML.emitNewline();
  compile_time::repeat<PrimitiveKindCount - 1>([&PTML]<size_t Index>() {
    auto CurrentKind = model::PrimitiveKind::Values(Index + 1);
    if (CurrentKind == model::PrimitiveKind::Void)
      return;

    else if (CurrentKind == model::PrimitiveKind::Generic)
      for (uint64_t Size : model::PrimitiveType::validSizes(CurrentKind))
        emitPrimitiveTypeSizeAssertion(PTML, CurrentKind, Size);

    else
      for (uint64_t Size : model::PrimitiveType::staticValidSizes(CurrentKind))
        emitPrimitiveTypeSizeAssertion(PTML, CurrentKind, Size);

    PTML.emitNewline();
  });
  PTML.emitNewline();
  PTML.emitUndefDirective(StaticAssertHelperMacro);
  PTML.emitNewline();
  PTML.emitNewline();

  PTML.emitComment("Pointers", ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();
  emitSizedPointerMacro(PTML, 2);
  emitSizedPointerMacro(PTML, 4);
  emitSizedPointerMacro(PTML, 8);
  PTML.emitNewline();

  PTML.emitComment("Undefined values",
                   ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();
  emitUndefMacros(PTML);

  PTML.emitComment("Break and continue",
                   ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();
  PTML.emitDefineDirective("break_to");
  PTML.emitSpace();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Goto);
  PTML.emitNewline();
  PTML.emitDefineDirective("continue_to");
  PTML.emitSpace();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Goto);
  PTML.emitNewline();
}
