//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"

#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"

#include "revng-c/Pipes/Kinds.h"

#include "MakeSegmentRefPass.h"
