#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "mlir/IR/MLIRContext.h"

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"

class TypeDefinitionEmitter : public CEmitter {
private:
  TypeEmitterConfiguration Configuration;

public:
  explicit TypeDefinitionEmitter(ptml::CTokenEmitter &Tokens,
                                 const CDataModel &DataModel,
                                 TypeEmitterConfiguration Configuration) :
    CEmitter(Tokens, DataModel), Configuration(Configuration) {}

private:
  void emitTypeKeyword(clift::DefinedType Type);
  void emitDeclarationTypedef(clift::DefinedType Type);
  void emitPaddingField(clift::ClassType Class,
                        uint64_t CurrentOffset,
                        uint64_t NextOffset);

public:
  void emitForwardDeclaration(clift::DefinedType Type) {
    revng_assert(isSeparateDeclarationAllowed(Type));
    emitDeclarationTypedef(Type);
  }

  void emitTypedefDefinition(clift::TypedefType Typedef);
  void emitFunctionTypedef(clift::FunctionType Function);
  void emitTypeDeclaration(clift::DefinedType Type);

private:
  void emitClassDefinition(clift::ClassType StructOrUnion);
  void emitEnumDefinition(clift::EnumType Enum);

public:
  void emitTypeDefinition(clift::DefinedType Type);

private:
  using NodeSet = std::unordered_set<const TypeDependencyNode *>;

public:
  /// This is a helper for emitting the entire tree of types in one fell swoop.
  ///
  /// \param Root indicates the node post-order iteration should start from.
  /// \param Emitted is passed into `llvm::post_order_ext` and should be shared
  ///         for all invocations of this method. It's a way to guarantee than
  ///         no type is emitted more than once.
  void emitTypeTree(const TypeDependencyNode &Root, NodeSet &Emitted);
};
