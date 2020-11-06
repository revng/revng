#ifndef REVNGC_CDECOMPILER_H
#define REVNGC_CDECOMPILER_H

//
// Copyright (c) rev.ng Srls 2017-2020.
//

#include <string>

namespace llvm {
class Module;
} // end namespace llvm

std::string
decompileFunction(const llvm::Module *M, const std::string &FunctionName);

#endif /* ifndef REVNGC_CDECOMPILER_H */
