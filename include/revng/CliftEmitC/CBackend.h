#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Clift/Clift.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

void decompile(clift::FunctionOp Function,
               ptml::CTokenEmitter &Emitter,
               TypeEmitterConfiguration Configuration = {});
