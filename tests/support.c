/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <elf.h>
#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>

// Save the program arguments for meaningful error reporting
int saved_argc;
char **saved_argv;

// Handle target specific information:
//
// * Register size
// * Register used as stack pointer
// * Macro to swap endianess from the host one
#if defined(TARGET_arm)

typedef uint32_t target_reg;
extern target_reg r13;
target_reg *stack = &r13;
#define SWAP(x) (htole32(x))

#elif defined(TARGET_x86_64)

typedef uint64_t target_reg;
extern target_reg rsp;
target_reg *stack = &rsp;
#define SWAP(x) (htole64(x))

#elif defined(TARGET_mips)

typedef uint32_t target_reg;
extern target_reg sp;
target_reg *stack = &sp;
#define SWAP(x) (htobe32(x))

#endif

void root(void);

void itoa(unsigned i, char *b){
  const char digit[] = "0123456789";
  char* p = b;

  int shifter = i;
  do {
    ++p;
    shifter = shifter / 10;
  } while(shifter);
  *p = '\0';

  do {
    *--p = digit[i % 10];
    i = i / 10;
  } while(i);
}

const unsigned align = sizeof(target_reg);
extern target_reg phdr_address;
extern target_reg e_phentsize;
extern target_reg e_phnum;

void page_set_flags(target_reg start, target_reg end, int flags) { }
void tb_invalidate_phys_range(target_reg start, target_reg end) { }
uintptr_t qemu_real_host_page_size = 1 << 12;
uintptr_t qemu_real_host_page_mask = ~((1 << 12) - 1);
uintptr_t qemu_host_page_size = 1 << 12;
uintptr_t qemu_host_page_mask = ~((1 << 12) - 1);


target_reg prepare_stack(target_reg stack, int argc, char **argv) {
  target_reg tmp;
  target_reg platform_address;
  target_reg random_address;
  target_reg arg_area;
  char **argp;
  char **arge;

#define MOVE(ptr, size) do {                        \
    (ptr) -= ((size) + align - 1) & ~(align - 1);   \
  } while (0);

#define PUSH(ptr, size, data) do {                  \
    MOVE(ptr, size);                                \
    memcpy((void *) (ptr), (data), size);           \
  } while(0)

#define PUSH_STR(ptr, data) do {                \
    tmp = strlen(data) + 1;                     \
    PUSH(ptr, (tmp), (void *) (data));          \
  } while(0)

#define PUSH_REG(ptr, data) do {                \
    tmp = SWAP(data);                           \
    PUSH(ptr, align, (void *) &tmp);            \
  } while (0);

#define PUSH_AUX(ptr, key, value) do {          \
    PUSH_REG(ptr, (target_reg) (value));        \
    PUSH_REG(ptr, key);                         \
  } while(0)

  // Reserve space for arguments and environment variables
  arg_area = stack;
  argp = argv;

  while (*++argp != NULL)
    ;
  while (*++argp != NULL)
    ;

  arge = argp;

  while (*--argp != NULL)
    MOVE(stack, strlen(*argp) + 1);
  // Push the arguments
  while (--argp != (argv - 1))
    MOVE(stack, strlen(*argp) + 1);

  PUSH_STR(stack, "revamb");
  platform_address = (target_reg) stack;

  //               1234567890123456
  PUSH_STR(stack, "4 I used a dice");
  random_address = (target_reg) stack;

  // Force 16 bytes alignment
  stack &= ~15;

  PUSH_AUX(stack, AT_NULL, 0);
  PUSH_AUX(stack, AT_PHDR, phdr_address);
  PUSH_AUX(stack, AT_PHENT, e_phentsize);
  PUSH_AUX(stack, AT_PHNUM, e_phnum);
  PUSH_AUX(stack, AT_PAGESZ, 4096);
  PUSH_AUX(stack, AT_BASE, 0);
  PUSH_AUX(stack, AT_FLAGS, 0);
  PUSH_AUX(stack, AT_ENTRY, &root);
  PUSH_AUX(stack, AT_UID, getuid());
  PUSH_AUX(stack, AT_EUID, geteuid());
  PUSH_AUX(stack, AT_GID, getgid());
  PUSH_AUX(stack, AT_EGID, getegid());
  PUSH_AUX(stack, AT_HWCAP, 0);
  PUSH_AUX(stack, AT_HWCAP2, 0);
  PUSH_AUX(stack, AT_CLKTCK, sysconf(_SC_CLK_TCK));
  PUSH_AUX(stack, AT_RANDOM, random_address);
  PUSH_AUX(stack, AT_PLATFORM, platform_address);

  PUSH_REG(stack, 0);

  // Copy arguments and environment variables, and store their address
  argp = arge;

  // First push environment variables
  while (*--argp != NULL) {
    PUSH_STR(arg_area, *argp);
    PUSH_REG(stack, arg_area);
  }

  // Push the separator
  PUSH_REG(stack, 0);

  // Push the arguments
  while (--argp != (argv - 1)) {
    PUSH_STR(arg_area, *argp);
    PUSH_REG(stack, arg_area);
  }

  PUSH_REG(stack, argc);

#undef PUSH_AUX
#undef PUSH_REG
#undef PUSH
#undef PUSH_STR
#undef MOVE

  return stack;
}

void target_set_brk(target_reg new_brk);
void syscall_init(void);

const char *path(const char *name) {
  return name;
}

void *g_malloc0_n(size_t n, size_t size) {
  return calloc(n, size);
}

void *g_malloc(size_t n_bytes) {
  if(n_bytes == 0)
    return NULL;
  else
    return malloc(n_bytes);
}

void g_free(void *memory) {
  if(memory == NULL)
    return;
  else
    return free(memory);
}

void unknownPC() {
  int arg;
  const char *error = "Unknown PC\n";
  write(2, error, strlen(error));
  for (arg = 0; arg < saved_argc; arg++) {
    write(2, saved_argv[arg], strlen(saved_argv[arg]));
    write(2, " ", 1);
  }
  write(2, "\n", 1);
  abort();
}

void newpc(uint64_t pc,
           uint64_t instruction_size,
           uint32_t is_first,
           uint8_t *vars, ...) {
  const size_t buffer_size = sizeof(uint64_t) * 2 + 3;
  char buffer[sizeof(uint64_t) * 2 + 3] = { 0 };
  char *last_pos = buffer + buffer_size - 1;

  if (!is_first)
    return;

  *last_pos = '\n';
  last_pos--;

  const char *hex_map = "0123456789abcdef";

  if (pc == 0)
    write(2, "0x0\n", 4);
  else {
    while (pc != 0) {
      *last_pos-- = hex_map[pc & 0xf];
      pc >>= 4;
    }

    *last_pos-- = 'x';
    *last_pos = '0';

    write(2, last_pos, buffer_size - (last_pos - buffer));
  }
}

int main(int argc, char *argv[]) {
  saved_argc = argc;
  saved_argv = argv;

  *stack = (target_reg) mmap((void *) NULL,
                             0x100000,
                             PROT_READ | PROT_WRITE,
                             MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                             -1,
                             0) + 0x100000 - 0x1000;

  *stack = prepare_stack(*stack, argc, argv);

  target_set_brk((target_reg) mmap((void *) NULL,
                                   0x1000,
                                   PROT_READ | PROT_WRITE,
                                   MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                                   -1,
                                   0) + 0x1000);

  syscall_init();

  root();

  return 0;
}
