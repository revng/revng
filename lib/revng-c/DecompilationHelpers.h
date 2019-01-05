#include <set>

namespace llvm {
class Function;
class GlobalVariable;
class Module;
} // namespace llvm

std::set<llvm::GlobalVariable *>
getDirectlyUsedGlobals(const std::set<llvm::Function *> &Funcs);

std::set<llvm::Function *>
getDirectlyCalledFunctions(const std::set<llvm::Function *> &Funcs);

std::set<llvm::Function *>
getRecursivelyCalledFunctions(const std::set<llvm::Function *> &Funcs);

std::set<llvm::Function *> getIsolatedFunctions(llvm::Module &M);
