#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterKind.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipeline/Registry.h"
