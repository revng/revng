#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>

#if defined(TARGET_arm)

typedef uint32_t target_reg;

extern target_reg r0;
extern target_reg r1;
extern target_reg r13;

target_reg *stack = &r13;
target_reg *return_value = &r0;
target_reg *first_argument = &r0;
target_reg *second_argument = &r1;

#elif defined(TARGET_x86_64)
typedef uint64_t target_reg;

extern target_reg rax;
extern target_reg rdi;
extern target_reg rsi;
extern target_reg rsp;

target_reg *stack = &rsp;
target_reg *return_value = &rax;
target_reg *first_argument = &rdi;
target_reg *second_argument = &rsi;
#endif


void root(void);

int main(int argc, char *argv[]) {
  int size = strlen(argv[1]);
  *first_argument = (target_reg) mmap((void *) 0x9000,
                                    0x3000,
                                    PROT_READ | PROT_WRITE, MAP_FIXED
                                    | MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                                    -1,
                                    0);
  memcpy((void *) *first_argument, argv[1], size);
  *second_argument = size;

  *stack = (target_reg) mmap((void *) 0x3000,
                           0x3000,
                           PROT_READ | PROT_WRITE, MAP_FIXED
                           | MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE,
                           -1,
                           0) + 0x1000;

  root();

  printf("%u\n", (unsigned) *return_value);

  return 0;
}
