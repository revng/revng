#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#ifdef __cplusplus
#include <cstdint>

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
#include "revng/PipelineC/ForwardDeclarations.h"
#else
#include "revng/PipelineC/ForwardDeclarationsC.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "revng/PipelineC/Prototypes.h"

#ifdef __cplusplus
} // extern C
#endif
