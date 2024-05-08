/*
 * This file is distributed under the MIT License. See LICENSE.mit for details.
 */

#include <assert.h>
#include <elf.h>
#include <endian.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <unwind.h>

#include "revng/Runtime/PlainMetaAddress.h"

#ifdef TARGET_x86_64
#include <asm/prctl.h>
#include <sys/prctl.h>

// glibc doesn't define this function, despite providing it
int arch_prctl(int code, unsigned long *addr);

#endif

#include "revng/Runtime/PrintPlainMetaAddress.h"

#include "support.h"

// Save the program arguments for meaningful error reporting
static int saved_argc;
static char **saved_argv;

// Macros to ensure that when we downcast from a 64-bit pointer to a 32-bit
// integer for the target architecture we're not losing information
#define MAX_OF(t)                                   \
  (((0x1ULL << ((sizeof(t) * 8ULL) - 1ULL)) - 1ULL) \
   | (0xFULL << ((sizeof(t) * 8ULL) - 4ULL)))

#define SAFE_CAST(ptr)                               \
  do {                                               \
    assert((uintptr_t) (ptr) <= MAX_OF(target_reg)); \
  } while (0)

noreturn void root(target_reg stack);

static const unsigned align = sizeof(target_reg);

// Define some variables declared in support.h
jmp_buf jmp_buffer;
target_reg *saved_registers;

// Default SIGSEGV handler
static struct sigaction default_handler;

static void *prepare_stack(void *stack, int argc, char **argv) {
  target_reg tmp;
  target_reg platform_address;
  target_reg random_address;
  void *arg_area;
  char **argp;
  char **arge;

  // Define some helper macros for building the stack
#define MOVE(ptr, size)                           \
  do {                                            \
    (ptr) -= ((size) + align - 1) & ~(align - 1); \
  } while (0);

#define PUSH(ptr, size, data)             \
  do {                                    \
    MOVE(ptr, size);                      \
    memcpy((void *) (ptr), (data), size); \
  } while (0)

#define PUSH_STR(ptr, data)            \
  do {                                 \
    tmp = strlen(data) + 1;            \
    PUSH(ptr, (tmp), (void *) (data)); \
  } while (0)

#define PUSH_REG(ptr, data)          \
  do {                               \
    tmp = SWAP(data);                \
    PUSH(ptr, align, (void *) &tmp); \
  } while (0);

#define PUSH_AUX(ptr, key, value)       \
  do {                                  \
    PUSH_REG(ptr, (uintptr_t) (value)); \
    PUSH_REG(ptr, key);                 \
  } while (0)

  // Reserve space for arguments and environment variables
  arg_area = stack;
  argp = argv;

  while (*++argp != NULL) {
  }
  while (*++argp != NULL) {
  }

  arge = argp;

  // Push the environment variables
  unsigned env_count = 0;
  while (*--argp != NULL) {
    MOVE(stack, strlen(*argp) + 1);
    env_count++;
  }

  // Push the arguments
  while (--argp != (argv - 1)) {
    MOVE(stack, strlen(*argp) + 1);
  }

  PUSH_STR(stack, "revng");
  platform_address = (uintptr_t) stack;

  PUSH_STR(stack, "4 I used a dice");
  random_address = (uintptr_t) stack;

  // Compute the value of stack pointer once we'll be done

  // WARNING: keep this number in sync with the number of auxiliary entries
  const unsigned aux_count = 16;
  unsigned entries_count = aux_count * 2 + 1 + env_count + 1 + argc + 1;
  uintptr_t final_stack = ((uintptr_t) stack
                           - entries_count * sizeof(target_reg));

  // Force 256 bits alignment of the final stack value
  const unsigned alignment = 256 / 8 - 1;
  uintptr_t alignment_offset = (final_stack
                                - (final_stack & ~((uintptr_t) alignment)));
  stack -= alignment_offset;
  final_stack -= alignment_offset;
  assert((final_stack & alignment) == 0);

  // Push the auxiliary vector

  // WARNING: if you add something here, update aux_count
  uintptr_t aux_start = (uintptr_t) stack;
  PUSH_AUX(stack, AT_NULL, 0);
  PUSH_AUX(stack, AT_PHDR, &phdr_address);
  PUSH_AUX(stack, AT_PHENT, &e_phentsize);
  PUSH_AUX(stack, AT_PHNUM, &e_phnum);
  PUSH_AUX(stack, AT_PAGESZ, 4096);
  PUSH_AUX(stack, AT_BASE, 0);
  PUSH_AUX(stack, AT_FLAGS, 0);
  PUSH_AUX(stack, AT_ENTRY, &root);
  PUSH_AUX(stack, AT_UID, getuid());
  PUSH_AUX(stack, AT_EUID, geteuid());
  PUSH_AUX(stack, AT_GID, getgid());
  PUSH_AUX(stack, AT_EGID, getegid());
  PUSH_AUX(stack, AT_HWCAP, 0);
  PUSH_AUX(stack, AT_CLKTCK, sysconf(_SC_CLK_TCK));
  PUSH_AUX(stack, AT_RANDOM, random_address);
  PUSH_AUX(stack, AT_PLATFORM, platform_address);
  // WARNING: if you add something here, update aux_count

  assert(aux_start - aux_count * 2 * sizeof(target_reg) == (uintptr_t) stack);

  // Push a separator
  PUSH_REG(stack, 0);

  // Copy arguments and environment variables, and store their address
  argp = arge;

  // First push environment variables
  while (*--argp != NULL) {
    PUSH_STR(arg_area, *argp);
    SAFE_CAST(arg_area);
    PUSH_REG(stack, (uintptr_t) arg_area);
  }

  // Push the separator
  PUSH_REG(stack, 0);

  // Push the arguments
  while (--argp != (argv - 1)) {
    PUSH_STR(arg_area, *argp);
    SAFE_CAST(arg_area);
    PUSH_REG(stack, (uintptr_t) arg_area);
  }

  PUSH_REG(stack, argc);

  assert((((uintptr_t) stack) & alignment) == 0);
  assert((uintptr_t) stack == final_stack);

#undef PUSH_AUX
#undef PUSH_REG
#undef PUSH
#undef PUSH_STR
#undef MOVE

  return stack;
}

// Helper functions we need
void target_set_brk(target_reg new_brk);
void syscall_init(void);

// Variables and functions required by helpers
uintptr_t qemu_real_host_page_size = 1 << 12;
uintptr_t qemu_real_host_page_mask = ~((1 << 12) - 1);
uintptr_t qemu_host_page_size = 1 << 12;
uintptr_t qemu_host_page_mask = ~((1 << 12) - 1);

void page_set_flags(target_reg start, target_reg end, int flags) {
}

void tb_invalidate_phys_range(target_reg start, target_reg end) {
}

const char *path(const char *name) {
  return name;
}

void *g_realloc(void *mem, size_t size) {
  return realloc(mem, size);
}

void *g_malloc0_n(size_t n, size_t size) {
  return calloc(n, size);
}

void *g_malloc(size_t n_bytes) {
  if (n_bytes == 0)
    return NULL;
  else
    return malloc(n_bytes);
}

void g_free(void *memory) {
  if (memory == NULL)
    return;
  else
    return free(memory);
}

void g_assertion_message_expr(const char *domain,
                              const char *file,
                              int line,
                              const char *func,
                              const char *expr) {
}

void unknown_pc() {
  int arg;
  fprintf(stderr, "Unknown PC:");
  fprint_metaaddress(stderr, &current_pc);
  fprintf(stderr, "\n");

  for (arg = 0; arg < saved_argc; arg++) {
    write(2, saved_argv[arg], strlen(saved_argv[arg]));
    write(2, " ", 1);
  }
  write(2, "\n", 1);

  abort();
}

void jump_to_symbol(char *Symbol) {
  _abort(Symbol);
}

#ifdef TRACE

// Execution tracing support
static int trace_fd = -1;
static size_t trace_buffer_size = 1024 * 1024;
static size_t trace_buffer_index = 0;
static uint64_t *trace_buffer;

static void flush_trace_buffer(void);

void flush_trace_buffer(void);
void flush_trace_buffer_signal_handler(int signal);

void init_tracing(void) {
  // If REVNG_TRACE_PATH contains a path, enable tracing
  char *trace_path = getenv("REVNG_TRACE_PATH");
  if (trace_path != NULL && strlen(trace_path) > 0) {
    trace_fd = open(trace_path,
                    O_WRONLY | O_CREAT | O_TRUNC,
                    S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(trace_fd != -1);

    // Set REVNG_TRACE_BUFFER_SIZE to customimze buffer size, default is 1024
    // * 1024 instructions
    char *trace_buffer_size_string = getenv("REVNG_TRACE_BUFFER_SIZE");
    if (trace_buffer_size_string != NULL
        && strlen(trace_buffer_size_string) > 0) {
      char **first_invalid = NULL;
      trace_buffer_size = strtoll(trace_buffer_size_string, first_invalid, 0);
      assert(**first_invalid == '\0');
    }

    // Allocate buffer to hold program counters
    trace_buffer = malloc(trace_buffer_size * sizeof(uint64_t));
    assert(trace_buffer != NULL);

    // In case of a crash, flush the buffer
    static const int signals[] = { SIGINT, SIGABRT, SIGTERM, SIGSEGV };
    for (unsigned c = 0; c < sizeof(signals) / sizeof(int); c++) {
      struct sigaction new_handler;
      struct sigaction old_handler;
      new_handler.sa_handler = flush_trace_buffer_signal_handler;
      int result = sigaction(signals[c], &new_handler, &old_handler);
      assert(result == 0);
      assert(old_handler.sa_handler == SIG_IGN
             || old_handler.sa_handler == SIG_DFL);
    }

    // Upon exit, flush the buffer too
    int result = atexit(flush_trace_buffer);
    assert(result == 0);
  }
}

static void flush_trace_buffer(void) {
  if (trace_fd == -1 || trace_buffer_index == 0)
    return;

  // Write the all buffer out and reset the counter
  write(trace_fd, trace_buffer, sizeof(uint64_t) * trace_buffer_index);
  trace_buffer_index = 0;
}

void flush_trace_buffer_signal_handler(int signal) {
  flush_trace_buffer();
}

// This function is called by the syscall helpers in case of exit/exit_group
void on_exit_syscall(void) {
  flush_trace_buffer();
}

void newpc(uint64_t pc,
           uint64_t instruction_size,
           uint32_t is_first,
           uint8_t *vars,
           ...) {
  // Check if tracing is enabled
  if (trace_fd == -1)
    return;

  // Record the program counter
  trace_buffer[trace_buffer_index++] = pc;

  // If the buffer is full, flush it out
  if (trace_buffer_index >= trace_buffer_size)
    flush_trace_buffer();
}

#else

void init_tracing(void) {
}

void on_exit_syscall(void) {
}

void newpc(uint64_t pc,
           uint64_t instruction_size,
           uint32_t is_first,
           uint8_t *vars,
           ...) {
}

#endif

// Check if the target address is inside an executable segment,
// if so serialize and jump
bool is_executable(uint64_t pc) {
  assert(segments_count != 0);

  // Check if the pc is inside one of the executable segments
  for (int i = 0; i < segments_count; i++)
    if (pc >= segment_boundaries[2 * i] && pc < segment_boundaries[2 * i + 1])
      return true;

  return false;
}

void handle_sigsegv(int signo, siginfo_t *info, void *opaque_context) {
  // If we are catching a SIGSEGV not thrown by the kill command
  if (signo == SIGSEGV && info->si_code != SI_USER
      && is_executable((uint64_t) info->si_addr)) {
    ucontext_t *context = opaque_context;
    saved_registers = (target_reg *) &context->uc_mcontext.gregs;
    longjmp(jmp_buffer, 0);
  }

  // If the address is not executable, this is not a jump into our code
  default_handler.sa_sigaction(SIGSEGV, info, opaque_context);
}

// Implant our custom SIGSEGV handler
void install_sigsegv_handler(void) {
  struct sigaction segv_handler;
  segv_handler.sa_sigaction = &handle_sigsegv;
  sigemptyset(&segv_handler.sa_mask);
  segv_handler.sa_flags = SA_SIGINFO | SA_NODEFER;

  int result = 0;
  result = sigaction(SIGSEGV, &segv_handler, &default_handler);
  assert(result == 0);
}

int main(int argc, char *argv[]) {
  // Save the program arguments for error reporting purposes
  saved_argc = argc;
  saved_argv = argv;

  // Initialize the tracing system
  init_tracing();

  // Allocate and initialize the stack
  void *stack = mmap((void *) NULL,
                     16 * 0x100000,
                     PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                     -1,
                     0)
                + 16 * 0x100000 - 0x1000;
  assert(stack != NULL);
  stack = prepare_stack(stack, argc, argv);

  // Allocate the brk page
  void *brk = mmap((void *) NULL,
                   0x1000,
                   PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                   -1,
                   0);
  assert(brk != NULL);
  brk += 0x1000;

  SAFE_CAST(brk);
  target_set_brk((uintptr_t) brk);

  // Initialize the syscall system
  syscall_init();

  // Implant custom SIGSEGV handler
  install_sigsegv_handler();

#ifdef TARGET_x86_64
  unsigned long fs_value;
  int result = arch_prctl(ARCH_GET_FS, &fs_value);
  assert(result == 0);
  set_register(REGISTER_FS, fs_value);
#endif

  // Run the translated program
  SAFE_CAST(stack);
  root((uintptr_t) stack);
}

static noreturn void fail(const char *reason,
                          PlainMetaAddress *source,
                          PlainMetaAddress *destination) {
  // Dump information about the exception
  fprintf(stderr, "Exception: %s", reason);
  fprintf(stderr, " (");
  fprint_metaaddress(stderr, source);
  fprintf(stderr, " -> ");
  fprint_metaaddress(stderr, destination);
  fprintf(stderr, ")\n");

  // Declare the exception object
  static struct _Unwind_Exception exc;

  // Raise the exception using the function provided by the unwind library
  _Unwind_RaiseException(&exc);

  abort();
}

noreturn void _abort(const char *reason) {
  fail(reason, &last_pc, &current_pc);
}

noreturn void _unreachable(const char *reason) {
  fail(reason, &last_pc, &current_pc);
}
