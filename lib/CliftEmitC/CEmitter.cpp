//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

namespace clift = mlir::clift;
using namespace mlir::clift;

class CEmitter::DeclarationEmitter {
  enum class StackItemKind {
    Terminal,
    Pointer,
    Array,
    Function,
  };

  struct StackItem {
    StackItemKind Kind;
    ValueType Type;
  };

  CEmitter &Parent;

  llvm::SmallVector<StackItem> Stack;
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

    // Recurse through the declaration, pushing each level onto the stack until
    // a terminal type is encountered. Primitive types as well as defined types
    // are considered terminal. Function types are not considered terminal if
    // function type expansion is enabled. Pointers with size not matching the
    // pointer size of the target implementation are considered terminal and
    // are printed by recursively entering this function.
    while (true) {
      StackItem Item = { StackItemKind::Terminal, Type };

      if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
        emitConstIfNeeded(T);
        Parent.emitPrimitiveType(T);
        NeedSpace = true;
      } else if (auto T = mlir::dyn_cast<PointerType>(Type)) {
        if (T.getPointerSize() == Parent.Target.PointerSize) {
          Item.Kind = StackItemKind::Pointer;
          Type = T.getPointeeType();
        } else {
          auto Macro = getForeignPointerMacroName(T.getPointerSize());

          emitConstIfNeeded(T);
          Parent.C.emitLiteralIdentifier(Macro);
          Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

          rc_recur DeclarationEmitter(Parent).emitImpl(T.getPointeeType(),
                                                       /*Declarator=*/nullptr);

          Parent.C.emitPunctuator(CTE::Punctuator::RightParenthesis);
          NeedSpace = true;
        }
      } else if (auto T = mlir::dyn_cast<ArrayType>(Type)) {
        Item.Kind = StackItemKind::Array;
        Type = T.getElementType();
      } else if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
        auto F = mlir::dyn_cast<FunctionType>(T);

        // The outermost function type is expanded into a function-declarator,
        // while for any inner function type, a typedef name is emitted instead.
        if (F and F == OutermostFunctionType) {
          Item.Kind = StackItemKind::Function;
          Type = F.getReturnType();
        } else {
          auto Kind = CTE::EntityKind::Typedef;

          if (mlir::isa<FunctionType>(T))
            Kind = CTE::EntityKind::Function;
          else if (mlir::isa<StructType>(T))
            Kind = CTE::EntityKind::Struct;
          else if (mlir::isa<UnionType>(T))
            Kind = CTE::EntityKind::Union;
          else if (mlir::isa<EnumType>(T))
            Kind = CTE::EntityKind::Enum;

          emitConstIfNeeded(T);
          Parent.C.emitIdentifier(T.getName(),
                                  T.getHandle(),
                                  Kind,
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
    // used to disambiguate non-root array and function types. The types must be
    // handled inside out, so the stack is visited in reverse order.
    for (auto [RI, SI] : llvm::enumerate(std::views::reverse(Stack))) {
      const size_t I = Stack.size() - RI - 1;

      switch (SI.Kind) {
      case StackItemKind::Terminal: {
        // Do nothing
      } break;
      case StackItemKind::Pointer: {
        auto T = mlir::dyn_cast<PointerType>(SI.Type);
        emitSpaceIfNeeded();
        Parent.C.emitPunctuator(CTE::Punctuator::Star);
        emitConstIfNeeded(T);
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array) {
          Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          NeedSpace = false;
        }
      } break;
      case StackItemKind::Function: {
        if (I != 0) {
          Parent.C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          NeedSpace = false;
        }
      } break;
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
      case StackItemKind::Terminal: {
        // Do nothing
      } break;
      case StackItemKind::Pointer: {
        // Do nothing
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array)
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
      case StackItemKind::Function: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);

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
      } break;
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
    C.emitKeyword(CTE::Keyword::Void);
  } else {
    auto TypeName = getPrimitiveTypeCName(Kind, Size);
    auto Location = pipeline::locationString(revng::ranks::PrimitiveType,
                                             TypeName);

    C.emitIdentifier(TypeName,
                     Location,
                     CTE::EntityKind::Primitive,
                     CTE::IdentifierKind::Reference);
  }
}

void CEmitter::emitType(ValueType Type) {
  DeclarationEmitter::emit(*this, Type, /*Declarator=*/nullptr);
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
