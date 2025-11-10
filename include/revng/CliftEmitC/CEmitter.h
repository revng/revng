#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/Clift.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/CTarget.h"

namespace mlir::clift {

/// Base class with common utilities for emitters emitting C from Clift.
class CEmitter {
protected:
  using CTE = ptml::CTokenEmitter;

  // WIP: make public instead of providing a getter? A better name?
  //      Maybe `PTML`?
  ptml::CTokenEmitter &C;

public:
  const TargetCImplementation &Target;

public:
  explicit CEmitter(ptml::CTokenEmitter &Emitter,
                    const TargetCImplementation &Target) :
    C(Emitter), Target(Target) {}

  ptml::CTokenEmitter &tokenEmitter() { return C; }

  //===------------------------------- Types ------------------------------===//

  void emitPrimitiveType(PrimitiveKind Kind, uint64_t Size);

  void emitPrimitiveType(PrimitiveType Type) {
    emitPrimitiveType(Type.getKind(), Type.getSize());
  }

  void emitType(ValueType Type);

  static bool isDeclarationTheSameAsDefinition(mlir::clift::DefinedType Type);
  static bool hasSeparateForwardDeclaration(mlir::clift::DefinedType Type) {
    return not isDeclarationTheSameAsDefinition(Type);
  }

  void emitTypeDeclaration(mlir::clift::DefinedType Type);
  void emitTypeDefinition(mlir::clift::DefinedType Type);

  //===---------------------------- Attributes ----------------------------===//

  static bool isValidAttributeArray(mlir::ArrayAttr Array);
  mlir::ArrayAttr getDeclarationOpAttributes(mlir::Operation *Op);

  void emitAttribute(AttributeAttr Attribute);
  void emitAttributes(mlir::ArrayAttr Attributes);

  //===--------------------------- Declarations ---------------------------===//

  /// Describes a function parameter declarator.
  struct ParameterDeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr Attributes;
  };

  /// Describes a declarator. This can be any function or variable declarator,
  /// including a function parameter declarator. When emitting a function
  /// declaration, the parameters declarators array must contain entries for
  /// each parameter of the outermost function type.
  struct DeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr Attributes;
    CTE::EntityKind Kind;

    llvm::ArrayRef<ParameterDeclaratorInfo> Parameters;
  };

  /// Emit a function or variable declaration of the specified type.
  void emitDeclaration(ValueType Type, DeclaratorInfo const &Declarator);

public:
  void emitFunctionPrototype(FunctionOp Function);

private:
  class DeclarationEmitter;
};

} // namespace mlir::clift
