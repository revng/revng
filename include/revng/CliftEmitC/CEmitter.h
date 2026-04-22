#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/PTML/CDoxygenEmitter.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/CTarget.h"

template<typename Type>
concept EntityWithComment = requires(Type const &Value) {
  { Value.getComment() } -> std::convertible_to<llvm::StringRef>;
};

/// Base class with common utilities for emitters emitting C from Clift.
class CEmitter {
protected:
  using CTE = ptml::CTokenEmitter;

  ptml::CTokenEmitter &Tokens;
  const TargetCImplementation &Target;

public:
  explicit CEmitter(ptml::CTokenEmitter &Emitter,
                    const TargetCImplementation &Target) :
    Tokens(Emitter), Target(Target) {}

  //===------------------------------- Types ------------------------------===//

  void emitPrimitiveType(mlir::clift::PrimitiveType Type);
  void emitType(mlir::Type Type);

  //===---------------------------- Attributes ----------------------------===//

  static bool isValidCAttributeArray(mlir::ArrayAttr Array);
  mlir::ArrayAttr getDeclarationOpCAttributes(mlir::Operation *Op);

  void emitCAttribute(mlir::clift::CAttributeAttr Attribute);
  void emitCAttributes(llvm::ArrayRef<mlir::clift::CAttributeAttr> Attributes,
                       bool SpaceBefore,
                       bool SpaceAfter);
  void emitCAttributes(mlir::ArrayAttr Attributes,
                       bool SpaceBefore,
                       bool SpaceAfter);

  //===---------------------------- Prototype -----------------------------===//

  void emitFunctionPrototype(mlir::clift::FunctionOp Function);

  //===--------------------------- Declarations ---------------------------===//

  /// Describes a function parameter declarator.
  struct ParameterDeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr CAttributes;
  };

  /// Describes a declarator. This can be any function or variable declarator,
  /// including a function parameter declarator. When emitting a function
  /// declaration, the parameters declarators array must contain entries for
  /// each parameter of the outermost function type.
  struct DeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr CAttributes;
    CTE::EntityKind Kind;

    std::optional<llvm::ArrayRef<ParameterDeclaratorInfo>> Parameters;
  };

  /// Emit a function or variable declaration of the specified type.
  void emitDeclaration(mlir::Type Type, DeclaratorInfo const &Declarator);

private:
  class DeclarationEmitter;

public:
  //===----------------------------- Comments -----------------------------===//
  template<EntityWithComment Type>
  void emitDoxygenComment(const Type &Value) {
    llvm::StringRef CommentContent = Value.getComment();
    if (CommentContent.empty())
      return;

    Tokens.emitNewline();
    ptml::CDoxygenEmitter::emitLineComment(Tokens, CommentContent);
  }

  void emitFunctionDoxygenComment(mlir::clift::FunctionOp Function);

  void emitGlobalDoxygenComment(mlir::clift::GlobalVariableOp Global);

public:
  /// A convenience function for emitting a comment with extra empty lines
  /// before and after it. For example:
  ///
  /// ```cpp
  ///   //
  ///   // All of the content, no matter how long it is,
  ///   // goes *here*.
  ///   //
  /// ```
  void emitCategoryComment(llvm::StringRef Content) {
    Tokens.emitComment("\n " + Content.str() + "\n\n",
                       ptml::CTokenEmitter::CommentKind::Line);
    Tokens.emitNewline();
  }

public:
  //===--------------------------- Other Helpers --------------------------===//

  static ptml::CTokenEmitter::EntityKind
  chooseEntityKind(mlir::clift::DefinedType Type);
};

/// Determines whether the type can be forward-declared or not.
///
/// This is true for `struct`s and `union`s. False for everything else.
inline bool isSeparateDeclarationAllowed(mlir::clift::DefinedType Type) {
  return mlir::isa<mlir::clift::ClassType>(Type);
}
