/// \file StructInitializers.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/FunctionTags.h"

using namespace llvm;

const char *StructInitializerPrefix = "struct_initializer";

StructInitializers::StructInitializers(llvm::Module *M) :
  Pool(M, false), Context(M->getContext()) {
  Pool.setMemoryEffects(MemoryEffects::none());
  Pool.addFnAttribute(Attribute::NoUnwind);
  Pool.addFnAttribute(Attribute::WillReturn);
  Pool.setTags({ &FunctionTags::StructInitializer,
                 &FunctionTags::UniquedByPrototype });

  // Record existing initializers
  Pool.initializeFromReturnType(FunctionTags::StructInitializer);
}

Instruction *StructInitializers::createReturn(revng::IRBuilder &Builder,
                                              ArrayRef<Value *> Values) {
  // Obtain return StructType
  auto *FT = Builder.GetInsertBlock()->getParent()->getFunctionType();
  auto *ReturnType = cast<StructType>(FT->getReturnType());

  SmallVector<Type *, 8> Types;
  llvm::copy(ReturnType->elements(), std::back_inserter(Types));

  // Create struct_initializer
  Function *Initializer = Pool.get(ReturnType,
                                   ReturnType,
                                   Types,
                                   StructInitializerPrefix);

  // Lazily populate its body
  if (Initializer->isDeclaration()) {
    auto *Entry = BasicBlock::Create(Context, "", Initializer);

    // TODO: the checks should be enabled conditionally based on the user.
    revng::NonDebugInfoCheckingIRBuilder InitializerBuilder(Entry);

    SmallVector<Value *, 8> Arguments;
    for (Argument &Arg : Initializer->args())
      Arguments.push_back(&Arg);

    InitializerBuilder.CreateAggregateRet(Arguments.data(), Arguments.size());
  }

  // Emit a call in the caller
  return Builder.CreateRet(Builder.CreateCall(Initializer, Values));
}
