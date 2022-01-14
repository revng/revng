/// \file LLVMContainer.cpp
/// \brief A llvm container is a container which uses a llvm module as a
/// backend, and can be customized with downstream kinds that specify which
/// global objects in it are which target.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/LLVMContainer.h"

const char *Pipeline::LLVMContainerTypeID = "RevngLLVMContainer";

template<>
const char Pipeline::LLVMContainer::ID = '0';
