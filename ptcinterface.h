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

using PTCInstructionListDestructor =
  std::integral_constant<decltype(&ptc_instruction_list_free),
                         &ptc_instruction_list_free>;
using PTCInstructionListPtr = std::unique_ptr<PTCInstructionList,
                                              PTCInstructionListDestructor>;

extern PTCInterface ptc;

#endif // _PTCINTERFACE_H
