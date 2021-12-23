#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {
class Function;
class Module;
class Type;
} // end namespace llvm

#include "revng/Support/FunctionTags.h"

namespace FunctionTags {
extern Tag AllocatesLocalVariable;
extern Tag MallocLike;
extern Tag ModelGEP;
extern Tag SerializationMarker;
} // namespace FunctionTags

llvm::Function *getModelGEP(llvm::Module &M, llvm::Type *T);
llvm::Function *getSerializationMarker(llvm::Module &M, llvm::Type *T);
