#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/CTarget.h"

namespace mlir::clift {

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

  void emitPrimitiveType(PrimitiveType Type);
  void emitType(mlir::Type Type);

  //===---------------------------- Attributes ----------------------------===//

  static bool isValidCAttributeArray(mlir::ArrayAttr Array);
  mlir::ArrayAttr getDeclarationOpCAttributes(mlir::Operation *Op);

  void emitCAttribute(CAttributeAttr Attribute);
  void emitCAttributes(mlir::ArrayAttr Attributes,
                       bool SpaceBefore,
                       bool SpaceAfter);

  //===---------------------------- Prototype -----------------------------===//

  void emitFunctionPrototype(FunctionOp Function);

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
  //===--------------------------- Other Helpers --------------------------===//

  static ptml::CTokenEmitter::EntityKind
  chooseEntityKind(mlir::clift::DefinedType Type);
};

/// Determines whether the type can be forward-declared or not.
///
/// This is true for `struct`s and `union`s. False for everything else.
inline bool isSeparateDeclarationAllowed(DefinedType Type) {
  return mlir::isa<ClassType>(Type);
}

} // namespace mlir::clift
