#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/IRHelper.h"

/// Marks the end of a translated basic block
inline IRHelper<> ExitTBMarker("exitTB");

/// Marks a jump to a symbol of a dynamic object
inline IRHelper<> JumpToSymbolMarker("jump_to_symbol");

/// Initializes the CPU state, provided by the QEMU helpers
inline IRHelper<> InitializeEnvHelper("helper_initialize_env");

/// Leaves the translated block, provided by the QEMU helpers
inline IRHelper<> CPULoopExitHelper("cpu_loop_exit");
