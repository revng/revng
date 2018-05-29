#ifndef _SUPPORT_H
#define _SUPPORT_H

/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>

// Handle target specific information:
//
// * Register size
// * Macro to swap endianess from the host one
#if defined(TARGET_arm)

typedef uint32_t target_reg;
#define SWAP(x) (htole32(x))
#define TARGET_REG_FORMAT PRIx32

#elif defined(TARGET_x86_64)

typedef uint64_t target_reg;
#define SWAP(x) (htole64(x))
#define TARGET_REG_FORMAT PRIx64

#elif defined(TARGET_i386)

typedef uint32_t target_reg;
#define SWAP(x) (htole32(x))
#define TARGET_REG_FORMAT PRIx32

#elif defined(TARGET_mips)

typedef uint32_t target_reg;
#define SWAP(x) (htobe32(x))
#define TARGET_REG_FORMAT PRIx32

#else

#error "Architecture not supported"

#endif

// Register values before the signal was triggered
extern target_reg *saved_registers;

extern uint64_t *segment_boundaries;
extern uint64_t segments_count;

// Variables used to initialize the stack
extern target_reg phdr_address;
extern target_reg e_phentsize;
extern target_reg e_phnum;

// Setjmp/longjmp buffer
extern jmp_buf jmp_buffer;

bool is_executable(uint64_t pc);

#endif // _SUPPORT_H
