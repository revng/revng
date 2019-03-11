#ifndef REVNGC_DECOMPILATION_HELPERS_H
#define REVNGC_DECOMPILATION_HELPERS_H

#include <set>

namespace llvm {
class Function;
class GlobalVariable;
} // namespace llvm

std::set<llvm::GlobalVariable *> getDirectlyUsedGlobals(llvm::Function &F);

std::set<llvm::Function *> getDirectlyCalledFunctions(llvm::Function &F);

#endif // REVNGC_DECOMPILATION_HELPERS_H
