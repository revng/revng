#ifndef REVNGC_CDECOMPILER_H
#define REVNGC_CDECOMPILER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

namespace llvm {
class Module;
} // end namespace llvm

std::string
decompileFunction(const llvm::Module *M, const std::string &FunctionName);

#endif /* ifndef REVNGC_CDECOMPILER_H */
