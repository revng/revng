//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PTML/Constants.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

#include "TypeStack.h"

namespace clift = mlir::clift;
using namespace mlir::clift;

static ptml::CTokenEmitter::EntityKind
chooseEntityKind(mlir::clift::DefinedType Type) {
  if (mlir::isa<mlir::clift::FunctionType>(Type))
    return ptml::CTokenEmitter::EntityKind::Function;
  else if (mlir::isa<mlir::clift::StructType>(Type))
    return ptml::CTokenEmitter::EntityKind::Struct;
  else if (mlir::isa<mlir::clift::UnionType>(Type))
    return ptml::CTokenEmitter::EntityKind::Union;
  else if (mlir::isa<mlir::clift::EnumType>(Type))
    return ptml::CTokenEmitter::EntityKind::Enum;
  else
    return ptml::CTokenEmitter::EntityKind::Typedef;
}

class CEmitter::DeclarationEmitter {
  CEmitter &Parent;
  FunctionType OutermostFunctionType = {};
  bool NeedSpace = false;

public:
  static void
  emit(CEmitter &Parent, ValueType Type, DeclaratorInfo const *Declarator) {
    DeclarationEmitter(Parent).emitImpl(Type, Declarator);
  }

private:
  explicit DeclarationEmitter(CEmitter &Parent) : Parent(Parent) {}

  void emitSpaceIfNeeded() {
    if (NeedSpace)
      Parent.C.emitSpace();
    NeedSpace = false;
  }

  void emitConstIfNeeded(ValueType Type) {
    emitSpaceIfNeeded();

    if (Type.isConst()) {
      Parent.C.emitKeyword(CTE::Keyword::Const);
      Parent.C.emitSpace();
    }
  }

  static std::string getForeignPointerMacroName(uint64_t PointerSize) {
    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << "pointer" << (PointerSize * 8) << "_t";
    }
    return Name;
  }

  RecursiveCoroutine<void> emitImpl(ValueType Type,
                                    DeclaratorInfo const *Declarator) {
    // Expanded function parameter declarator names are only emitted for the
    // outermost function type of a function declarator. When emitting a
    // function declarator, if the specified type is a function type,
    // OutermostFunctionType is initialised to allow subsequent comparisons to
    // determine if a given function type is the outermost type and should be
    // expanded.
    if (Declarator and Declarator->Kind == CTE::EntityKind::Function) {
      if (auto Function = mlir::dyn_cast<FunctionType>(Type))
        OutermostFunctionType = Function;
    }

    auto Stack = makeTypeStack(Type, Parent.Target.PointerSize);
    for (const TypeStackItem &Item : Stack) {
      switch (Item.Kind) {
      case TypeStackItem::ItemKind::Primitive:
        emitConstIfNeeded(Item.Type);
        using PrimitiveT = mlir::clift::PrimitiveType;
        Parent.emitPrimitiveType(mlir::cast<PrimitiveT>(Item.Type));
        NeedSpace = true;
        break;

      case TypeStackItem::ItemKind::Pointer:
        // Non-terminal pointers are handled down below.
        break;

      case TypeStackItem::ItemKind::ForeignPointer: {
        auto T = mlir::cast<PointerType>(Item.Type);
        auto Macro = getForeignPointerMacroName(T.getPointerSize());

        emitConstIfNeeded(T);
        Parent.C.emitLiteralIdentifier(Macro);
        Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

        rc_recur DeclarationEmitter(Parent).emitImpl(T.getPointeeType(),
                                                     /*Declarator=*/nullptr);

        Parent.C.emitPunctuator(CTE::Punctuator::RightParenthesis);
        NeedSpace = true;
      } break;

      case TypeStackItem::ItemKind::Array:
        // Arrays are handled down below.
        break;

      case TypeStackItem::ItemKind::Defined: {
        auto F = mlir::dyn_cast<FunctionType>(Item.Type);

        // The outermost function type is expanded into a function-declarator,
        // while for any inner function type, a typedef name is emitted instead.
        if (F and F == OutermostFunctionType) {
          rc_recur DeclarationEmitter(Parent).emitImpl(F.getReturnType(),
                                                       /*Declarator=*/nullptr);

          // WIP: only emit this sometimes!!!!
          Parent.C.emitSpace();

        } else {
          emitConstIfNeeded(Item.Type);

          auto DefinedType = mlir::cast<mlir::clift::DefinedType>(Item.Type);
          Parent.C.emitIdentifier(DefinedType.getName(),
                                  DefinedType.getHandle(),
                                  chooseEntityKind(DefinedType),
                                  CTE::IdentifierKind::Reference);

          NeedSpace = true;
        }
      } break;

      default:
        revng_abort("Unknown stack item kind.");
      }
    }

    // Print type syntax appearing before the declarator name. This includes
    // cv-qualifiers, stars indicating a pointer, as well as left parentheses
    // used to disambiguate non-root array and function types. The types must be
    // handled inside out, so the stack is visited in reverse order.
    for (auto [RI, SI] : llvm::enumerate(std::views::reverse(Stack))) {
      const size_t I = Stack.size() - RI - 1;

      switch (SI.Kind) {
      case TypeStackItem::ItemKind::Pointer: {
        auto T = mlir::dyn_cast<PointerType>(SI.Type);
        emitSpaceIfNeeded();
        Parent.C.emitPunctuator(CTE::Punctuator::Star);
        emitConstIfNeeded(T);
      } break;

      case TypeStackItem::ItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != TypeStackItem::ItemKind::Array) {
          Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          NeedSpace = false;
        }
      } break;

      case TypeStackItem::ItemKind::Defined: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);
        if (F and F == OutermostFunctionType) {
          if (I != 0) {
            Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
            NeedSpace = false;
          }
        }
      } break;

      default:
        // Do nothing
        break;
      }
    }

    if (Declarator) {
      emitSpaceIfNeeded();
      Parent.C.emitIdentifier(Declarator->Identifier,
                              Declarator->Location,
                              Declarator->Kind,
                              CTE::IdentifierKind::Definition);
    }

    // Print type syntax appearing after the declarator name. This includes
    // right parentheses matching the left parentheses printed in the first
    // pass, as well as array extents and function parameter lists. The
    // declarators appearing in function parameter lists are printed by
    // recursively entering this function.
    for (auto [I, SI] : llvm::enumerate(Stack)) {
      switch (SI.Kind) {
      case TypeStackItem::ItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != TypeStackItem::ItemKind::Array)
          Parent.C.emitPunctuator(CTE::Punctuator::RightParenthesis);

        Parent.C.emitPunctuator(CTE::Punctuator::LeftBracket);

        uint64_t Extent = mlir::cast<ArrayType>(SI.Type).getElementsCount();

        // Use a wider bit-width to handle the edge-case of an extent greater
        // than the maximum value of a signed 64-bit integer. Making the value
        // unsigned would cause unnecessary type suffixes to be emitted.
        auto ExtentValue = llvm::APSInt(llvm::APInt(/*numBits=*/128, Extent),
                                        /*isUnsigned=*/false);

        Parent.C.emitIntegerLiteral(ExtentValue,
                                    CIntegerKind::Int,
                                    /*Radix=*/10);

        Parent.C.emitPunctuator(CTE::Punctuator::RightBracket);
      } break;

      case TypeStackItem::ItemKind::Defined: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);
        if (F and F == OutermostFunctionType) {
          if (I != 0)
            Parent.C.emitPunctuator(CTE::Punctuator::RightParenthesis);

          Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          if (F.getArgumentTypes().empty()) {
            Parent.emitPrimitiveType(PrimitiveKind::VoidKind, 0);
          } else {
            for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
              if (J != 0) {
                Parent.C.emitPunctuator(CTE::Punctuator::Comma);
                Parent.C.emitSpace();
              }

              DeclaratorInfo ParameterDeclarator;
              DeclaratorInfo const *InnerDeclarator = nullptr;

              if (F == OutermostFunctionType) {
                ParameterDeclarator = DeclaratorInfo{
                  .Identifier = Declarator->Parameters[J].Identifier,
                  .Location = Declarator->Parameters[J].Location,
                  .Attributes = Declarator->Parameters[J].Attributes,
                  .Kind = CTE::EntityKind::FunctionParameter,
                };

                InnerDeclarator = &ParameterDeclarator;
              }

              rc_recur DeclarationEmitter(Parent).emitImpl(PT, InnerDeclarator);
            }
          }
          Parent.C.emitPunctuator(CTE::Punctuator::RightParenthesis);
        }
      } break;

      default:
        // Do nothing
        break;
      }
    }

    if (Declarator)
      Parent.emitAttributes(Declarator->Attributes);
  }
};

//===-------------------------------- Types -------------------------------===//

static std::string getPrimitiveTypeCName(PrimitiveKind Kind, uint64_t Size) {
  auto GetPrefix = [](PrimitiveKind Kind) -> llvm::StringRef {
    switch (Kind) {
    case PrimitiveKind::UnsignedKind:
      return "uint";
    case PrimitiveKind::SignedKind:
      return "int";
    default:
      return clift::stringifyPrimitiveKind(Kind);
    }
  };

  std::string Name;
  {
    llvm::raw_string_ostream Out(Name);
    Out << GetPrefix(Kind);

    if (Kind != PrimitiveKind::VoidKind)
      Out << (Size * 8) << "_t";
  }
  return Name;
}

void CEmitter::emitPrimitiveType(clift::PrimitiveKind Kind, uint64_t Size) {
  if (Kind == PrimitiveKind::VoidKind) {
    revng_assert(Size == 0);
    C.emitKeyword(CTE::Keyword::Void);

  } else {
    C.emitPrimitive(getPrimitiveTypeCName(Kind, Size),
                    CTE::IdentifierKind::Reference);
  }
}

void CEmitter::emitType(ValueType Type) {
  DeclarationEmitter::emit(*this, Type, /*Declarator=*/nullptr);
}

bool CEmitter::isDeclarationTheSameAsDefinition(mlir::clift::DefinedType Type) {
  return not mlir::isa<mlir::clift::StructType>(Type)
         and not mlir::isa<mlir::clift::UnionType>(Type)
         and not mlir::isa<mlir::clift::EnumType>(Type);
}

static void emitTypeKeyword(ptml::CTokenEmitter &PTML,
                            mlir::clift::DefinedType Type) {
  if (mlir::isa<mlir::clift::EnumType>(Type))
    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Enum);

  else if (mlir::isa<mlir::clift::StructType>(Type))
    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Struct);

  else if (mlir::isa<mlir::clift::UnionType>(Type))
    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Union);

  else if (mlir::isa<mlir::clift::TypedefType>(Type)
           || mlir::isa<mlir::clift::FunctionType>(Type))
    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);

  else
    revng_abort("Unsupported defined type.");
}

static void emitForwardDeclaration(mlir::clift::CEmitter &Emitter,
                                   mlir::clift::DefinedType Type) {
  revng_assert(not CEmitter::isDeclarationTheSameAsDefinition(Type));

  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();
  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();
  emitTypeKeyword(PTML, Type);
  PTML.emitSpace();
  PTML.emitAttribute<"_PACKED">();
  PTML.emitSpace();
  PTML.emitIdentifier(Type.getName(),
                      Type.getHandle(),
                      chooseEntityKind(Type),
                      ptml::CTokenEmitter::IdentifierKind::Reference);
  PTML.emitSpace();
  PTML.emitIdentifier(Type.getName(),
                      Type.getHandle(),
                      chooseEntityKind(Type),
                      ptml::CTokenEmitter::IdentifierKind::Reference);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static ptml::TagEmitter markAsCommentable(ptml::CTokenEmitter &PTML,
                                          llvm::StringRef Location) {
  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::ActionContextLocation, Location);
  Tag.emitListAttribute(ptml::attributes::AllowedActions,
                        { ptml::actions::Comment });
  Tag.finalizeOpenTag();
  return Tag;
}

static void emitTypedefDefinition(mlir::clift::CEmitter &Emitter,
                                  mlir::clift::TypedefType Typedef) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();
  auto Guard = markAsCommentable(PTML, Typedef.getHandle());

  // TODO: emit model comment.

  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();

  Emitter.emitDeclaration(Typedef.getUnderlyingType(),
                          mlir::clift::CEmitter::DeclaratorInfo{
                            .Identifier = Typedef.getName(),
                            .Location = Typedef.getHandle(),

                            // WIP: should i be passing something in here?
                            .Attributes = {},

                            .Kind = ptml::CTokenEmitter::EntityKind::Typedef,
                          });
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitFunctionTypedef(mlir::clift::CEmitter &Emitter,
                                mlir::clift::FunctionType Function) {
  if (Function.getName().empty()) {
    revng_abort("DEBUG");
    revng_assert(pipeline::locationFromString(revng::ranks::HelperFunction,
                                              Function.getHandle()));
    // Skip helper typedefs.
    return;
  }

  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();
  auto Guard = markAsCommentable(PTML, Function.getHandle());

  // TODO: emit model comment.

  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  PTML.emitSpace();

  // WIP: use a modified version of `emitFunctionPrototype`;
  PTML.emitComment("TODO", ptml::CTokenEmitter::CommentKind::Block);
  PTML.emitSpace();

  PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Void);
  PTML.emitSpace();
  PTML.emitIdentifier(Function.getName(),
                      Function.getHandle(),
                      chooseEntityKind(Function),
                      ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

void CEmitter::emitTypeDeclaration(mlir::clift::DefinedType Type) {
  if (not CEmitter::isDeclarationTheSameAsDefinition(Type)) {
    emitForwardDeclaration(*this, Type);

  } else if (auto Typedef = mlir::dyn_cast<mlir::clift::TypedefType>(Type)) {
    emitTypedefDefinition(*this, Typedef);

  } else if (auto Function = mlir::dyn_cast<mlir::clift::FunctionType>(Type)) {
    emitFunctionTypedef(*this, Function);

  } else {
    Type.dump();
    revng_abort("Unknown defined type.");
  }
}

static std::string paddingFieldName(uint64_t CurrentOffset) {
  // TODO: this discards the prefix configuration option.
  //       We should fix this after the configuration is separate from the model

  model::CNameBuilder Builder(model::Binary{});
  return Builder.paddingFieldName(CurrentOffset);
}

static void emitPaddingField(ptml::CTokenEmitter &PTML,
                             uint64_t CurrentOffset,
                             uint64_t NextOffset) {
  revng_assert(CurrentOffset <= NextOffset);
  if (CurrentOffset == NextOffset)
    return; // There is no padding

  static constexpr auto Unsigned = model::PrimitiveKind::Unsigned;
  PTML.emitPrimitive(model::PrimitiveType::getCName(Unsigned, 1));
  PTML.emitSpace();

  PTML.emitIdentifier(paddingFieldName(CurrentOffset),
                      "",
                      ptml::CTokenEmitter::EntityKind::Field,
                      ptml::CTokenEmitter::IdentifierKind::Definition);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::LeftBracket);
  PTML.emitSignedIntegerLiteral(NextOffset - CurrentOffset);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::RightBracket);
  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitStructDefinition(mlir::clift::CEmitter &Emitter,
                                 mlir::clift::StructType Struct) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

  {
    auto Guard = markAsCommentable(PTML, Struct.getHandle());

    // TODO: emit model comment.

    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Struct);
    PTML.emitSpace();
    PTML.emitAttribute<"_PACKED">();
    PTML.emitSpace();

    // TODO: conditionally emit `_CAN_CONTAIN_CODE`.

    PTML.emitAnnotation<"_SIZE">(Struct.getSize());
    PTML.emitSpace();

    PTML.emitIdentifier(Struct.getName(),
                        Struct.getHandle(),
                        chooseEntityKind(Struct),
                        ptml::CTokenEmitter::IdentifierKind::Definition);
    PTML.emitSpace();
  }

  {
    auto Sc = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::StructDefinition,
                              ptml::CTokenEmitter::Delimiter::Braces);
    PTML.emitNewline();

    uint64_t PreviousOffset = 0;
    for (const auto &Field : Struct.getFields()) {
      // TODO: add a way to suppress padding fields.
      if (true)
        emitPaddingField(PTML, PreviousOffset, Field.getOffset());

      auto Guard = markAsCommentable(PTML, Field.getHandle());

      // TODO: emit model comment.

      Emitter.emitDeclaration(Field.getType(),
                              mlir::clift::CEmitter::DeclaratorInfo{
                                .Identifier = Field.getName(),
                                .Location = Field.getHandle(),

                                // WIP: should i be passing something in here?
                                .Attributes = {},

                                .Kind = ptml::CTokenEmitter::EntityKind::Field,
                              });

      // TODO: is an automatic name emitted? Or a custom one?
      if (true) {
        PTML.emitSpace();
        PTML.emitAnnotation<"_STARTS_AT">(Field.getOffset());
      }

      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();

      PreviousOffset = Field.getOffset() + Field.getType().getByteSize();
    }
  }

  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitUnionDefinition(mlir::clift::CEmitter &Emitter,
                                mlir::clift::UnionType Union) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

  {
    auto Guard = markAsCommentable(PTML, Union.getHandle());

    // TODO: emit model comment.

    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Union);
    PTML.emitSpace();
    PTML.emitAttribute<"_PACKED">();
    PTML.emitSpace();

    // WIP: this is different from the old backend, @fez, any objections?
    PTML.emitAnnotation<"_SIZE">(Union.getByteSize());
    PTML.emitSpace();

    PTML.emitIdentifier(Union.getName(),
                        Union.getHandle(),
                        chooseEntityKind(Union),
                        ptml::CTokenEmitter::IdentifierKind::Definition);
    PTML.emitSpace();
  }

  {
    auto Sc = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::UnionDefinition,
                              ptml::CTokenEmitter::Delimiter::Braces);
    PTML.emitNewline();

    for (const auto &Field : Union.getFields()) {
      auto Guard = markAsCommentable(PTML, Field.getHandle());

      // TODO: emit model comment.

      Emitter.emitDeclaration(Field.getType(),
                              mlir::clift::CEmitter::DeclaratorInfo{
                                .Identifier = Field.getName(),
                                .Location = Field.getHandle(),

                                // WIP: should i be passing something in here?
                                .Attributes = {},

                                .Kind = ptml::CTokenEmitter::EntityKind::Field,
                              });

      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();
    }
  }

  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

static void emitEnumDefinition(mlir::clift::CEmitter &Emitter,
                               mlir::clift::EnumType Enum) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

  {
    auto Guard = markAsCommentable(PTML, Enum.getHandle());

    // TODO: emit model comment.

    PTML.emitKeyword(ptml::CTokenEmitter::Keyword::Enum);
    PTML.emitSpace();
    {
      auto AnnotationGuard = PTML.emitComplexAnnotation<"_ENUM_UNDERLYING">();
      Emitter.emitType(Enum.getUnderlyingType());
    }
    PTML.emitSpace();
    PTML.emitAttribute<"_PACKED">();
    PTML.emitSpace();

    // WIP: this is different from the old backend, @fez, any objections?
    PTML.emitAnnotation<"_SIZE">(Enum.getByteSize());
    PTML.emitSpace();

    PTML.emitIdentifier(Enum.getName(),
                        Enum.getHandle(),
                        chooseEntityKind(Enum),
                        ptml::CTokenEmitter::IdentifierKind::Definition);
    PTML.emitSpace();
  }

  {
    auto Scope = PTML.enterScope(ptml::CTokenEmitter::ScopeKind::EnumDefinition,
                                 ptml::CTokenEmitter::Delimiter::Braces);
    PTML.emitNewline();

    for (const auto &Entry : Enum.getFields()) {
      auto Guard = markAsCommentable(PTML, Entry.getHandle());

      // TODO: emit model comment.

      PTML.emitIdentifier(Entry.getName(),
                          Entry.getHandle(),
                          ptml::CTokenEmitter::EntityKind::Enumerator,
                          ptml::CTokenEmitter::IdentifierKind::Definition);
      PTML.emitSpace();
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Equals);
      PTML.emitSpace();

      // TODO: do we want hex here? Or some complex logic?
      PTML.emitUnsignedIntegerLiteral(Entry.getRawValue());

      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();
    }

    // TODO: add a way to suppress this.
    if (true) {
      // We have to make the enum of the correct size of the underlying type
      auto ByteSize = Enum.getByteSize();
      revng_assert(ByteSize <= 8);
      size_t FullMask = std::numeric_limits<size_t>::max();
      size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                     FullMask :
                                     ((FullMask)
                                      xor (FullMask << (8 * ByteSize)));

      // TODO: pull the prefix from the configuration when it's available
      //       without pulling in the model dependency.
      static constexpr llvm::StringRef Prefix = "enum_max_value_";

      namespace ranks = revng::ranks;
      auto EnumLocation = *pipeline::locationFromString(ranks::TypeDefinition,
                                                        Enum.getHandle());
      auto EntryLocation = EnumLocation.extend(ranks::EnumEntry,
                                               MaxBitPatternInEnum);

      PTML.emitIdentifier(Prefix.str() + Enum.getName().str(),
                          EntryLocation.toString(),
                          ptml::CTokenEmitter::EntityKind::Enumerator,
                          ptml::CTokenEmitter::IdentifierKind::Definition);
      PTML.emitSpace();
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Equals);
      PTML.emitSpace();

      // WIP: use hex here!
      PTML.emitUnsignedIntegerLiteral(MaxBitPatternInEnum);

      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();
    }
  }

  PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  PTML.emitNewline();
}

void CEmitter::emitTypeDefinition(mlir::clift::DefinedType Type) {
  if (isDeclarationTheSameAsDefinition(Type)) {
    emitTypeDeclaration(Type);
    tokenEmitter().emitNewline();
    return;

  } else if (auto Struct = mlir::dyn_cast<mlir::clift::StructType>(Type)) {
    emitStructDefinition(*this, Struct);

  } else if (auto Union = mlir::dyn_cast<mlir::clift::UnionType>(Type)) {
    emitUnionDefinition(*this, Union);

  } else if (auto Enum = mlir::dyn_cast<mlir::clift::EnumType>(Type)) {
    emitEnumDefinition(*this, Enum);

  } else {
    Type.dump();
    revng_abort("Unknown defined type.");
  }
}

//===----------------------------- Attributes -----------------------------===//

bool CEmitter::isValidAttributeArray(mlir::ArrayAttr ArrayAttr) {
  auto IsAttributeAttr = [](mlir::Attribute Attr) {
    return mlir::isa<AttributeAttr>(Attr);
  };
  return std::ranges::all_of(ArrayAttr, IsAttributeAttr);
}

mlir::ArrayAttr CEmitter::getDeclarationOpAttributes(mlir::Operation *Op) {
  if (auto Attr = Op->getAttr("clift.attributes")) {
    auto ArrayAttr = mlir::cast<mlir::ArrayAttr>(Attr);
    revng_assert(isValidAttributeArray(ArrayAttr));
    return ArrayAttr;
  }
  return {};
}

void CEmitter::emitAttribute(AttributeAttr Attribute) {
  auto Macro = Attribute.getMacro();

  C.emitSpace();
  C.emitIdentifier(Macro.getString(),
                   Macro.getHandle(),
                   CTE::EntityKind::Attribute,
                   CTE::IdentifierKind::Reference);

  if (auto Arguments = Attribute.getArguments()) {
    C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    for (auto [I, A] : llvm::enumerate(*Arguments)) {
      if (I != 0) {
        C.emitPunctuator(CTE::Punctuator::Comma);
        C.emitSpace();
      }

      C.emitIdentifier(A.getString(),
                       A.getHandle(),
                       CTE::EntityKind::AttributeArgument,
                       CTE::IdentifierKind::Reference);
    }

    C.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }
}

void CEmitter::emitAttributes(mlir::ArrayAttr Attributes) {
  if (Attributes) {
    for (mlir::Attribute Attr : Attributes)
      emitAttribute(mlir::cast<AttributeAttr>(Attr));
  }
}

//===---------------------------- Declarations ----------------------------===//

void CEmitter::emitDeclaration(ValueType Type,
                               DeclaratorInfo const &Declarator) {
  DeclarationEmitter::emit(*this, Type, &Declarator);
}

void CEmitter::emitFunctionPrototype(FunctionOp Op) {
  llvm::SmallVector<ParameterDeclaratorInfo> ParameterDeclarators;
  for (unsigned I = 0; I < Op.getArgCount(); ++I) {
    auto Attrs = Op.getArgAttrs(I);

    auto GetStringAttr = [&Attrs](llvm::StringRef Name) {
      return mlir::cast<mlir::StringAttr>(Attrs.get(Name)).getValue();
    };

    mlir::ArrayAttr Attributes = {};
    if (auto Attr = Attrs.get("clift.attributes")) {
      Attributes = mlir::cast<mlir::ArrayAttr>(Attr);
      revng_assert(isValidAttributeArray(Attributes));
    }

    ParameterDeclarators.emplace_back(GetStringAttr("clift.name"),
                                      GetStringAttr("clift.handle"),
                                      Attributes);
  }

  // TODO: print _ABI(VALUE)
  // TODO: print attributes (`_NoReturn` and such)
  // TODO: print argument location (`_REG` and such)

  emitDeclaration(Op.getCliftFunctionType(),
                  mlir::clift::CEmitter::DeclaratorInfo{
                    .Identifier = Op.getName(),
                    .Location = Op.getHandle(),
                    .Attributes = getDeclarationOpAttributes(Op),
                    .Kind = ptml::CTokenEmitter::EntityKind::Function,
                    .Parameters = ParameterDeclarators,
                  });
}
