/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdint.h>

#include "support.h"

// The only purpose of this function is keeping alive the references to some
// symbols that are needed by revng
intptr_t ignore(void) {
  return (intptr_t) &saved_registers + (intptr_t) &setjmp
         + (intptr_t) &jmp_buffer + (intptr_t) &is_executable;
}
