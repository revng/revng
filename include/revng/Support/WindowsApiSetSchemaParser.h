#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/StringRef.h"

#include "revng/Support/Configuration.h"

std::optional<revng::WindowsLibraryMap>
parseWindowsApiSetSchemaFromDLL(llvm::StringRef Path);
