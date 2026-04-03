#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"

std::optional<std::string>
findPathCaseInsensitive(llvm::StringRef Root,
                        llvm::StringRef CaseInsensitivePath);
