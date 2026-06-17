//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

const char *StructInitializerPrefix = "struct_initializer";

StructInitializers::StructInitializers(llvm::Module *M, bool EmitBody) :
  Pool(M, false), Context(M->getContext()), EmitBody(EmitBody) {
  Pool.setMemoryEffects(MemoryEffects::none());
  Pool.addFnAttribute(Attribute::NoUnwind);
  Pool.addFnAttribute(Attribute::WillReturn);
  Pool.setTags({ &FunctionTags::StructInitializer,
                 &FunctionTags::UniquedByPrototype });

  // Record existing initializers
  Pool.initializeFromReturnType(FunctionTags::StructInitializer);
}

CallInst *StructInitializers::createCall(revng::IRBuilder &Builder,
                                         StructType *ReturnType,
                                         ArrayRef<Value *> Values) {
  SmallVector<Type *, 8> Types;
  llvm::copy(ReturnType->elements(), std::back_inserter(Types));

  // Create struct_initializer
  Function *Initializer = Pool.get(ReturnType,
                                   ReturnType,
                                   Types,
                                   StructInitializerPrefix);

  // Lazily populate its body, unless the caller opted out (e.g., the body
  // would otherwise persist past `remove-lifting-artifacts` and confuse later
  // canonicalize passes that expect non-isolated functions to be declarations).
  if (EmitBody and Initializer->isDeclaration()) {
    auto *Entry = BasicBlock::Create(Context, "", Initializer);

    // TODO: the checks should be enabled conditionally based on the user.
    revng::NonDebugInfoCheckingIRBuilder InitializerBuilder(Entry);

    SmallVector<Value *, 8> Arguments;
    for (Argument &Arg : Initializer->args())
      Arguments.push_back(&Arg);

    InitializerBuilder.CreateAggregateRet(Arguments.data(), Arguments.size());
  }

  // Emit a call in the caller
  return cast<CallInst>(Builder.CreateCall(Initializer, Values));
}

Instruction *StructInitializers::createReturn(revng::IRBuilder &Builder,
                                              ArrayRef<Value *> Values) {
  // Obtain return StructType
  auto *FT = Builder.GetInsertBlock()->getParent()->getFunctionType();
  auto *ReturnType = cast<StructType>(FT->getReturnType());

  return Builder.CreateRet(createCall(Builder, ReturnType, Values));
}
