#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {
class Function;
class LLVMContext;
class Module;
class Type;
} // end namespace llvm

#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace FunctionTags {
extern Tag AllocatesLocalVariable;
extern Tag MallocLike;
extern Tag IsRef;
extern Tag AddressOf;
extern Tag ModelGEP;
extern Tag SerializationMarker;
} // namespace FunctionTags

/// Returns the type of an AddressOf function with return type T, in context C.
/// This is intended to be used to get functions from the AddressOfPool created
/// with createAddresOfPool()
llvm::FunctionType *
getAddressOfFunctionType(llvm::LLVMContext &C, llvm::Type *T);

/// Initializes a pool of AddressOf functions, initializing it its internal
/// Module
void initAddressOfPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetTy, llvm::Type *BaseAddressTy);

llvm::Function *getSerializationMarker(llvm::Module &M, llvm::Type *T);
