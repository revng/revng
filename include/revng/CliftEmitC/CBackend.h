#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Clift/Clift.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/CTarget.h"

namespace clift {

void decompile(FunctionOp Function,
               ptml::CTokenEmitter &Emitter,
               const TargetCImplementation &Target);

} // namespace clift
