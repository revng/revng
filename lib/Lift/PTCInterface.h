#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <type_traits>

#define USE_DYNAMIC_PTC
#include "ptc.h"

template<void (*T)(PTCInstructionList *)>
using PTCDestructorWrapper = std::integral_constant<decltype(T), T>;

inline void ptcInstructionListDestructor(PTCInstructionList *This) {
  ptc_instruction_list_free(This);
  delete This;
}

using PTCDestructor = PTCDestructorWrapper<&ptcInstructionListDestructor>;

using PTCInstructionListPtr = std::unique_ptr<PTCInstructionList,
                                              PTCDestructor>;

extern PTCInterface ptc;
