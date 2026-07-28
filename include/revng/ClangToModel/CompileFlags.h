#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

namespace revng {

/// The Clang flags needed to parse revng-emitted C: the flags from
/// share/revng/compile-flags.cfg, plus `-xc` and the include paths for Clang's
/// builtin headers and share/revng/include/primitive-types.h.
std::vector<std::string> getClangCompileFlags();

} // namespace revng
