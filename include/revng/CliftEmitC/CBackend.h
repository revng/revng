#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Clift/Clift.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

/// Backend-specific configuration options.
struct CBackendConfiguration {
  /// Configuration for the type emitter.
  TypeEmitterConfiguration TypeEmitter = {};

  /// Should stack frame types be inlined?
  bool InlineStackFrameType = false;
};

void decompile(clift::FunctionOp Function,
               ptml::CTokenEmitter &Emitter,
               CBackendConfiguration Configuration = {});
