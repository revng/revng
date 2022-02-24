#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef __cplusplus
#include <cstdint>

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
#include "revng/PipelineC/cpp_typedefs.h"
#else
#include "revng/PipelineC/c_typedefs.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "revng/PipelineC/prototypes.h"

#ifdef __cplusplus
} // extern C
#endif
