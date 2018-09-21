#ifndef _PTCINTERFACE_H
#define _PTCINTERFACE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <memory>
#include <type_traits>

// Local includes
#include "revamb.h"
#define USE_DYNAMIC_PTC
#include "ptc.h"

template<void (*T)(PTCInstructionList *)>
using PTCDestructorWrapper = std::integral_constant<decltype(T), T>;

using PTCDestructor = PTCDestructorWrapper<&ptc_instruction_list_free>;

using PTCInstructionListPtr = std::unique_ptr<PTCInstructionList,
                                              PTCDestructor>;

extern PTCInterface ptc;

#endif // _PTCINTERFACE_H
