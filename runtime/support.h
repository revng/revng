#ifndef SUPPORT_H
#define SUPPORT_H

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

// TODO: these have been recovered by hand
enum {
  REGISTER_RAX = 0x8250,
  REGISTER_RCX = 0x8258,
  REGISTER_RDX = 0x8260,
  REGISTER_RBX = 0x8268,
  REGISTER_RSP = 0x8270,
  REGISTER_RBP = 0x8278,
  REGISTER_RSI = 0x8280,
  REGISTER_RDI = 0x8288,
  REGISTER_R8 = 0x8290,
  REGISTER_R9 = 0x8298,
  REGISTER_R12 = 0x82b0,
  REGISTER_R13 = 0x82b8,
  REGISTER_R14 = 0x82c0,
  REGISTER_R15 = 0x82c8,

  REGISTER_FS = 0x8370,

  REGISTER_XMM0 = 0x8558,
  REGISTER_XMM1 = 0x8598,
  REGISTER_XMM2 = 0x85d8,
  REGISTER_XMM3 = 0x8618,
  REGISTER_XMM4 = 0x8658,
  REGISTER_XMM5 = 0x8698,
  REGISTER_XMM6 = 0x86d8,
  REGISTER_XMM7 = 0x8718
};

#elif defined(TARGET_i386)

typedef uint32_t target_reg;
#define SWAP(x) (htole32(x))
#define TARGET_REG_FORMAT PRIx32

#elif defined(TARGET_mips)

typedef uint32_t target_reg;
#define SWAP(x) (htobe32(x))
#define TARGET_REG_FORMAT PRIx32

#elif defined(TARGET_mipsel)

typedef uint32_t target_reg;
#define SWAP(x) (htole32(x))
#define TARGET_REG_FORMAT PRIx32

#elif defined(TARGET_s390x)

typedef uint64_t target_reg;
#define SWAP(x) (htobe64(x))
#define TARGET_REG_FORMAT PRIx64

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
void set_register(uint32_t register_id, uint64_t value);

#endif // SUPPORT_H
