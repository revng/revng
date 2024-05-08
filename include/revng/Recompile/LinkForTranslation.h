#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Binary.h"

void linkForTranslation(const model::Binary &Model,
                        llvm::StringRef InputBinaryPath,
                        llvm::StringRef ObjectFilePath,
                        llvm::StringRef OutputBinaryPath);

void printLinkForTranslationCommands(llvm::raw_ostream &OS,
                                     const model::Binary &Model,
                                     llvm::StringRef InputBinary,
                                     llvm::StringRef ObjectFile,
                                     llvm::StringRef OutputBinary);
