//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftImportModel/CAttributeListBuilder.h"
#include "revng/PTML/CDoxygenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

using namespace clift;

ptml::CTokenEmitter::EntityKind CEmitter::chooseEntityKind(DefinedType Type) {
  if (mlir::isa<FunctionType>(Type))
    return ptml::CTokenEmitter::EntityKind::Function;
  else if (mlir::isa<StructType>(Type))
    return ptml::CTokenEmitter::EntityKind::Struct;
  else if (mlir::isa<UnionType>(Type))
    return ptml::CTokenEmitter::EntityKind::Union;
  else if (mlir::isa<EnumType>(Type))
    return ptml::CTokenEmitter::EntityKind::Enum;
  else if (mlir::isa<TypedefType>(Type))
    return ptml::CTokenEmitter::EntityKind::Typedef;
  else
    revng_abort("Unsupported defined type");
}

class CEmitter::DeclarationEmitter {
  enum class StackItemKind {
    Terminal,
    Pointer,
    Array,
    Function,
  };

  struct StackItem {
    StackItemKind Kind;
    mlir::Type Type;
  };

  CEmitter &Parent;

  llvm::SmallVector<StackItem> Stack;
  FunctionType OutermostFunctionType = {};
  bool NeedSpace = false;

public:
  static void
  emit(CEmitter &Parent, mlir::Type Type, DeclaratorInfo const *Declarator) {
    DeclarationEmitter(Parent).emitImpl(Type, Declarator);
  }

private:
  explicit DeclarationEmitter(CEmitter &Parent) : Parent(Parent) {}

  void emitSpaceIfNeeded() {
    if (NeedSpace)
      Parent.Tokens.emitSpace();
    NeedSpace = false;
  }

  void emitConstIfNeeded(mlir::Type Type) {
    emitSpaceIfNeeded();

    if (isConst(Type)) {
      Parent.Tokens.emitKeyword(CTE::Keyword::Const);
      Parent.Tokens.emitSpace();
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

  std::optional<ptml::CTokenEmitter::Region>
  commentableReturnValueGuard(DeclaratorInfo const *Declarator) {
    if (OutermostFunctionType == nullptr)
      return std::nullopt;

    // Only emitting the return value guard when parameters are available might
    // look weird at the first sight, but when we are emitting a prototype
    // without parameter information (the typedef), it's weird to allow adding
    // comments to a return type that will show up elsewhere.
    if (not Declarator->Parameters)
      return std::nullopt;

    auto FTHandle = OutermostFunctionType.getHandle();
    auto FTLoc = pipeline::locationFromString(revng::ranks::TypeDefinition,
                                              FTHandle);
    if (not FTLoc)
      return std::nullopt;

    using RK = ptml::CTokenEmitter::RegionKind;
    auto RVLoc = FTLoc->transmute(revng::ranks::ReturnValue).toString();
    return std::make_optional<ptml::CTokenEmitter::Region>(Parent.Tokens,
                                                           RK::Commentable,
                                                           RVLoc);
  }

  std::optional<ptml::CTokenEmitter::Region>
  commentableParameterGuard(FunctionType const CurrentFunction,
                            DeclaratorInfo const *Declarator,
                            uint64_t Index) {
    if (OutermostFunctionType == nullptr)
      return std::nullopt;

    if (OutermostFunctionType != CurrentFunction)
      return std::nullopt;

    if (not Declarator->Parameters)
      return std::nullopt;

    auto CurrentLocation = Declarator->Parameters.value()[Index].Location;

    using RKind = ptml::CTokenEmitter::RegionKind;
    return std::make_optional<ptml::CTokenEmitter::Region>(Parent.Tokens,
                                                           RKind::Commentable,
                                                           CurrentLocation);
  }

  RecursiveCoroutine<void> emitImpl(mlir::Type Type,
                                    DeclaratorInfo const *Declarator) {
    // Expanded function parameter declarator names are only emitted for the
    // outermost function type of a function declarator. When emitting a
    // function declarator, if the specified type is a function type,
    // OutermostFunctionType is initialised to allow subsequent comparisons to
    // determine if a given function type is the outermost type and should be
    // expanded.
    if (Declarator and Declarator->Kind == CTE::EntityKind::Function) {
      if (auto Function = mlir::dyn_cast<FunctionType>(Type)) {
        if (Declarator) {
          Parent.emitCAttributes(Declarator->CAttributes,
                                 /* SpaceBefore = */ false,
                                 /* SpaceAfter = */ true);
        }

        OutermostFunctionType = Function;
      }
    }

    {
      auto Guard = commentableReturnValueGuard(Declarator);

      // Recurse through the declaration, pushing each level onto the stack
      // until a terminal type is encountered. Primitive types as well as
      // defined types are considered terminal. Function types are not
      // considered terminal if function type expansion is enabled.
      // Pointers with size not matching the pointer size of the target
      // implementation are considered terminal and are printed by recursively
      // entering this function.
      while (true) {
        StackItem Item = { StackItemKind::Terminal, Type };

        if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
          emitConstIfNeeded(T);
          Parent.emitPrimitiveType(T);
          NeedSpace = true;
        } else if (auto T = mlir::dyn_cast<PointerType>(Type)) {
          if (T.getPointerSize() == Parent.DataModel.PointerSize) {
            Item.Kind = StackItemKind::Pointer;
            Type = T.getPointeeType();
          } else {
            auto Macro = getForeignPointerMacroName(T.getPointerSize());

            emitConstIfNeeded(T);
            Parent.Tokens.emitLiteralIdentifier(Macro);
            Parent.Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

            rc_recur DeclarationEmitter(Parent)
              .emitImpl(T.getPointeeType(),
                        /*Declarator=*/nullptr);

            Parent.Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
            NeedSpace = true;
          }
        } else if (auto T = mlir::dyn_cast<ArrayType>(Type)) {
          Item.Kind = StackItemKind::Array;
          Type = T.getElementType();
        } else if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
          auto F = mlir::dyn_cast<FunctionType>(T);

          // The outermost function type is expanded into a function-declarator,
          // while for any inner function type, a typedef name is emitted
          // instead.
          if (F and F == OutermostFunctionType) {
            Item.Kind = StackItemKind::Function;
            Type = F.getReturnType();
          } else {
            emitConstIfNeeded(T);
            Parent.Tokens.emitIdentifier(T.getName(),
                                         T.getHandle(),
                                         chooseEntityKind(T),
                                         CTE::IdentifierKind::Reference);

            NeedSpace = true;
          }
        }

        Stack.push_back(Item);

        if (Item.Kind == StackItemKind::Terminal)
          break;
      }

      // Print type syntax appearing before the declarator name. This includes
      // cv-qualifiers, stars indicating a pointer, as well as left parentheses
      // used to disambiguate non-root array and function types. The types must
      // be handled inside out, so the stack is visited in reverse order.
      for (auto [RI, SI] : llvm::enumerate(std::views::reverse(Stack))) {
        const size_t I = Stack.size() - RI - 1;

        switch (SI.Kind) {
        case StackItemKind::Terminal: {
          // Do nothing
        } break;
        case StackItemKind::Pointer: {
          auto T = mlir::dyn_cast<PointerType>(SI.Type);
          emitSpaceIfNeeded();
          Parent.Tokens.emitPunctuator(CTE::Punctuator::Star);
          emitConstIfNeeded(T);
        } break;
        case StackItemKind::Array: {
          if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array) {
            Parent.Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);
            NeedSpace = false;
          }
        } break;
        case StackItemKind::Function: {
          if (I != 0) {
            Parent.Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);
            NeedSpace = false;
          }
        } break;
        }
      }
    }

    if (Declarator) {
      emitSpaceIfNeeded();
      Parent.Tokens.emitIdentifier(Declarator->Identifier,
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
      case StackItemKind::Terminal: {
        // Do nothing
      } break;
      case StackItemKind::Pointer: {
        // Do nothing
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array)
          Parent.Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);

        Parent.Tokens.emitPunctuator(CTE::Punctuator::LeftBracket);

        uint64_t Extent = mlir::cast<ArrayType>(SI.Type).getElementsCount();
        Parent.Tokens.emitIntegerLiteral(llvm::APInt(64, Extent), std::nullopt);

        Parent.Tokens.emitPunctuator(CTE::Punctuator::RightBracket);
      } break;
      case StackItemKind::Function: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);

        if (I != 0)
          Parent.Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);

        Parent.Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);
        if (F.getArgumentTypes().empty()) {
          Parent.Tokens.emitKeyword(CTE::Keyword::Void);
        } else {
          if (Declarator->Parameters.has_value())
            if (Declarator->Parameters->size() != F.getArgumentTypes().size())
              revng_abort("Declarator doesn't match the prototype");

          for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
            if (J != 0) {
              Parent.Tokens.emitPunctuator(CTE::Punctuator::Comma);
              Parent.Tokens.emitSpace();
            }

            auto Guard = commentableParameterGuard(F, Declarator, J);

            DeclaratorInfo ParameterDeclarator;
            DeclaratorInfo const *InnerDeclarator = nullptr;

            if (F == OutermostFunctionType && Declarator->Parameters) {
              ParameterDeclarator = DeclaratorInfo{
                .Identifier = Declarator->Parameters.value()[J].Identifier,
                .Location = Declarator->Parameters.value()[J].Location,
                .CAttributes = Declarator->Parameters.value()[J].CAttributes,
                .Kind = CTE::EntityKind::FunctionParameter,
              };

              InnerDeclarator = &ParameterDeclarator;
            }

            rc_recur DeclarationEmitter(Parent).emitImpl(PT, InnerDeclarator);
          }
        }
        Parent.Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
      } break;
      }
    }

    if (Declarator and not OutermostFunctionType) {
      Parent.emitCAttributes(Declarator->CAttributes,
                             /* SpaceBefore = */ true,
                             /* SpaceAfter = */ false);
    }
  }
};

//===-------------------------------- Types -------------------------------===//

static std::string getPrimitiveTypeCName(PrimitiveType Type) {
  auto GetPrefix = [](PrimitiveType Type) -> llvm::StringRef {
    if (mlir::isa<FloatType>(Type))
      return "float";

    switch (auto Kind = mlir::cast<IntegerType>(Type).getKind()) {
    case IntegerKind::Unsigned:
      return "uint";
    case IntegerKind::Signed:
      return "int";
    default:
      return stringifyIntegerKind(Kind);
    }
  };

  std::string Name;
  {
    llvm::raw_string_ostream Out(Name);
    Out << GetPrefix(Type) << (getObjectSize(Type) * 8) << "_t";
  }
  return Name;
}

void CEmitter::emitPrimitiveType(PrimitiveType Type) {
  if (mlir::isa<VoidType>(Type)) {
    Tokens.emitKeyword(CTE::Keyword::Void);
  } else {
    auto TypeName = getPrimitiveTypeCName(Type);
    auto Location = pipeline::locationString(revng::ranks::PrimitiveType,
                                             TypeName);

    Tokens.emitIdentifier(TypeName,
                          Location,
                          CTE::EntityKind::Primitive,
                          CTE::IdentifierKind::Reference);
  }
}

void CEmitter::emitType(mlir::Type Type) {
  DeclarationEmitter::emit(*this, Type, /*Declarator=*/nullptr);
}

//===----------------------------- Attributes -----------------------------===//

bool CEmitter::isValidCAttributeArray(mlir::ArrayAttr ArrayAttr) {
  auto IsCAttributeAttr = [](mlir::Attribute Attr) {
    return mlir::isa<CAttributeAttr>(Attr);
  };
  return std::ranges::all_of(ArrayAttr, IsCAttributeAttr);
}

mlir::ArrayAttr CEmitter::getDeclarationOpCAttributes(mlir::Operation *Op) {
  CAttributeListBuilder Builder(Op->getContext(),
                                Op->getAttr("clift.c_attributes"));

  if (auto Function = mlir::dyn_cast<FunctionOp>(Op)) {
    if (Function->hasAttr("noreturn"))
      Builder.setOrUpdate<"_NORETURN">();
    if (Function->hasAttr("always_inline"))
      Builder.setOrUpdate<"_ALWAYS_INLINE">();

    Builder.append(Function.getFunctionType().getCAttributes());
  }

  mlir::ArrayAttr Result = Builder.get();
  if (not isValidCAttributeArray(Result)) {
    Op->dump();
    revng_abort("Invalid `clift.c_attributes` array");
  }

  return Result;
}

void CEmitter::emitCAttribute(CAttributeAttr CAttribute) {
  auto Name = CAttribute.getName();

  Tokens.emitIdentifier(Name.getName(),
                        Name.getHandle(),
                        CTE::EntityKind::Attribute,
                        CTE::IdentifierKind::Reference);

  if (auto Arguments = CAttribute.getArguments()) {
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    for (auto [I, A] : llvm::enumerate(Arguments)) {
      if (I != 0) {
        Tokens.emitPunctuator(CTE::Punctuator::Comma);
        Tokens.emitSpace();
      }

      if (auto Id = mlir::dyn_cast<CIdentifierAttr>(A)) {
        Tokens.emitIdentifier(Id.getName(),
                              Id.getHandle(),
                              CTE::EntityKind::AttributeArgument,
                              CTE::IdentifierKind::Reference);

      } else if (auto Integer = mlir::dyn_cast<mlir::IntegerAttr>(A)) {
        Tokens.emitIntegerLiteral(Integer.getValue(), std::nullopt);

      } else if (auto Type = mlir::dyn_cast<mlir::TypeAttr>(A)) {
        auto VT = mlir::dyn_cast<mlir::Type>(Type.getValue());
        revng_assert(VT);
        emitType(VT);

      } else {
        revng_abort("Unknown c_attribute argument type!");
      }
    }

    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }
}

void // formatting
CEmitter::emitCAttributes(llvm::ArrayRef<CAttributeAttr> CAttributes,
                          bool SpaceBefore,
                          bool SpaceAfter) {
  if (CAttributes.empty())
    return;

  if (SpaceBefore)
    Tokens.emitSpace();

  bool First = true;
  for (CAttributeAttr Attribute : CAttributes) {
    if (First)
      First = false;
    else
      Tokens.emitSpace();

    emitCAttribute(Attribute);
  }

  if (SpaceAfter)
    Tokens.emitSpace();
}

void CEmitter::emitCAttributes(mlir::ArrayAttr CAttributes,
                               bool SpaceBefore,
                               bool SpaceAfter) {
  if (not CAttributes)
    return;

  auto Range = llvm::to_vector(CAttributes.getAsRange<CAttributeAttr>());
  emitCAttributes(Range, SpaceBefore, SpaceAfter);
}

//===---------------------------- Declarations ----------------------------===//

void CEmitter::emitDeclaration(mlir::Type Type,
                               DeclaratorInfo const &Declarator) {
  DeclarationEmitter::emit(*this, Type, &Declarator);
}

void CEmitter::emitFunctionPrototype(FunctionOp Op) {
  bool IsHelper = pipeline::locationFromString(revng::ranks::HelperFunction,
                                               Op.getHandle())
                    .has_value();
  if (not IsHelper)
    emitFunctionDoxygenComment(Op);

  std::optional<llvm::ArrayRef<ParameterDeclaratorInfo>> Parameters;
  llvm::SmallVector<ParameterDeclaratorInfo> ParameterDeclarators;
  if (not IsHelper) {
    for (unsigned I = 0; I < Op.getArgCount(); ++I) {
      const auto &Attrs = Op.getArgAttrs(I);

      revng_assert(Attrs.getOfType<mlir::StringAttr>("clift.name"),
                   "Function argument name (clift.name) is missing.");

      auto Attributes = Attrs.getOfType<mlir::ArrayAttr>("clift.c_attributes");
      revng_assert(not Attributes or isValidCAttributeArray(Attributes));

      ParameterDeclarators.emplace_back(Attrs.getString("clift.name"),
                                        Attrs.getStringOrEmpty("clift.handle"),
                                        Attributes);
    }

    Parameters = ParameterDeclarators;
  }

  std::optional<ptml::CTokenEmitter::Region> MaybeRegion;
  if (not IsHelper) {
    using RegionKind = ptml::CTokenEmitter::RegionKind;
    MaybeRegion.emplace(Tokens, RegionKind::Commentable, Op.getHandle());
  }

  emitDeclaration(Op.getFunctionType(),
                  CEmitter::DeclaratorInfo{
                    .Identifier = Op.getName(),
                    .Location = Op.getHandle(),
                    .CAttributes = getDeclarationOpCAttributes(Op),
                    .Kind = ptml::CTokenEmitter::EntityKind::Function,
                    .Parameters = Parameters,
                  });
}

// Accounts for a `\` before it and a space after.
static constexpr uint64_t ExtraKeywordIndentation = 2;

void CEmitter::emitFunctionDoxygenComment(FunctionOp Function) {
  auto Guard = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Function.getHandle());

  std::optional<ptml::CDoxygenEmitter> Emitter = std::nullopt;

  // Function comment
  mlir::Attribute RawAttribute = Function->getAttr("clift.comment");
  if (RawAttribute != nullptr) {
    auto CommentBody = mlir::cast<mlir::StringAttr>(RawAttribute).getValue();
    if (not CommentBody.empty()) {
      ptml::CDoxygenEmitter::emitLineComment(Emitter, Tokens);
      Emitter->emit(CommentBody);
      Emitter->emit("\n");
    }
  }

  // `\param` comments
  for (unsigned I = 0; I < Function.getArgCount(); ++I) {
    auto Attrs = Function.getArgAttrs(I);
    llvm::StringRef CommentBody = Attrs.getStringOrEmpty("clift.comment");
    if (CommentBody.empty())
      continue;

    if (Emitter.has_value())
      Emitter->emit("\n");
    else
      ptml::CDoxygenEmitter::emitLineComment(Emitter, Tokens);

    llvm::StringRef Handle = Attrs.getString("clift.handle");
    auto ArgG = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                   Handle);

    static constexpr llvm::StringRef Keyword = "param";
    Emitter->emitKeyword(Keyword);
    Emitter->emit(" ");

    llvm::StringRef EmittedName = Attrs.getString("clift.name");
    Tokens.emitIdentifier(EmittedName,
                          Handle,
                          ptml::CTokenEmitter::EntityKind::FunctionParameter,
                          ptml::CTokenEmitter::IdentifierKind::Reference);
    Emitter->emit(" ");

    uint64_t Indentation = Keyword.size() + ExtraKeywordIndentation
                           + EmittedName.size() + 1;

    Emitter->indent(Indentation);
    Emitter->emit(CommentBody);
    Emitter->indent(-Indentation);
    Emitter->emit("\n");
  }

  // `\returns` comment
  RawAttribute = Function->getAttr("clift.return_value_comment");
  if (RawAttribute != nullptr) {
    auto CommentBody = mlir::cast<mlir::StringAttr>(RawAttribute).getValue();
    if (not CommentBody.empty()) {
      if (Emitter.has_value())
        Emitter->emit("\n");
      else
        ptml::CDoxygenEmitter::emitLineComment(Emitter, Tokens);

      auto FunctionTypeHandle = Function.getFunctionType().getHandle();
      auto FTLoc = pipeline::locationFromString(revng::ranks::TypeDefinition,
                                                FunctionTypeHandle);
      revng_assert(FTLoc.has_value());
      auto RVLoc = FTLoc->transmute(revng::ranks::ReturnValue).toString();

      using RegionKind = ptml::CTokenEmitter::RegionKind;
      auto Guard = Tokens.enterRegion(RegionKind::Commentable, RVLoc);

      static constexpr llvm::StringRef Keyword = "returns";
      Emitter->emitKeyword(Keyword);
      Emitter->emit(" ");

      size_t Indentation = Keyword.size() + ExtraKeywordIndentation;

      Emitter->indent(Indentation);
      Emitter->emit(CommentBody);
      Emitter->indent(-Indentation);
    }
  }
}

void CEmitter::emitGlobalDoxygenComment(GlobalVariableOp Global) {
  if (auto CommentAttribute = Global->getAttr("clift.comment")) {
    auto String = mlir::dyn_cast<mlir::StringAttr>(CommentAttribute);
    revng_assert(String != nullptr);

    if (not String.getValue().empty())
      ptml::CDoxygenEmitter::emitLineComment(Tokens, String.getValue());
  }
}
