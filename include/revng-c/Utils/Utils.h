#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

std::unique_ptr<llvm::raw_fd_ostream>
openFunctionFile(const llvm::StringRef DirectoryPath,
                 const llvm::StringRef FunctionName,
                 const llvm::StringRef Suffix);
