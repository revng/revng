#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "mlir/IR/MLIRContext.h"

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"

namespace mlir::clift {

class TypeDefinitionEmitter : public clift::CEmitter {
private:
  mlir::MLIRContext *Context;
  clift::TypeEmitterConfiguration Configuration;

public:
  TypeDefinitionEmitter(ptml::CTokenEmitter &PTML,
                        const TargetCImplementation &Target,
                        mlir::MLIRContext &Context,
                        clift::TypeEmitterConfiguration Configuration) :
    clift::CEmitter(PTML, Target),
    Context(&Context),
    Configuration(Configuration) {}

private:
  void emitTypeKeyword(clift::DefinedType Type);
  void emitDeclarationTypedef(mlir::MLIRContext &Context,
                              clift::DefinedType Type);
  void emitPaddingField(mlir::MLIRContext &Context,
                        uint64_t CurrentOffset,
                        uint64_t NextOffset);

public:
  void emitForwardDeclaration(mlir::MLIRContext &Context,
                              clift::DefinedType Type) {
    revng_assert(clift::isSeparateDeclarationAllowed(Type));

    emitDeclarationTypedef(Context, Type);
  }

  void emitTypedefDefinition(clift::TypedefType Typedef);
  void emitFunctionTypedef(clift::FunctionType Function);
  void emitTypeDeclaration(mlir::MLIRContext &Context, clift::DefinedType Type);

public:
  void emitClassDefinition(mlir::MLIRContext &Context,
                           clift::ClassType StructOrUnion);
  void emitEnumDefinition(mlir::MLIRContext &Context, clift::EnumType Enum);
  void emitTypeDefinition(mlir::MLIRContext &Context, clift::DefinedType Type);

private:
  using NodeSet = std::unordered_set<const clift::TypeDependencyNode *>;

public:
  /// This is a helper for emitting the entire tree of types in one fell swoop.
  ///
  /// \param Root indicates the node post-order iteration should start from.
  /// \param Emitted is passed into `llvm::post_order_ext` and should be shared
  ///         for all invocations of this method. It's a way to guarantee than
  ///         no type is emitted more than once.
  void emitTypeTree(mlir::MLIRContext &Context,
                    const clift::TypeDependencyNode &Root,
                    NodeSet &Emitted);
};

} // namespace mlir::clift
