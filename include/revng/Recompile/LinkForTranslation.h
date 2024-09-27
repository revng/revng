#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

void linkForTranslation(const model::Binary &Model,
                        llvm::StringRef InputBinaryPath,
                        llvm::StringRef ObjectFilePath,
                        llvm::StringRef OutputBinaryPath);
