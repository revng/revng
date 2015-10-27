#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <string.h>

extern uint32_t r0;
extern uint32_t r1;
extern uint32_t r13;
void root(void);

int main(int argc, char *argv[]) {
  int size = strlen(argv[1]);
  r0 = (int) mmap((void *) 0x9000, 0x3000, PROT_READ | PROT_WRITE, MAP_FIXED
                  | MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE, -1, 0);
  memcpy((void *) r0, argv[1], size);
  r1 = size;
  r13 = (int) mmap((void *) 0x3000, 0x3000, PROT_READ | PROT_WRITE, MAP_FIXED
                   | MAP_ANONYMOUS | MAP_32BIT | MAP_PRIVATE, -1, 0) + 0x1000;
  root();
  printf("%u\n", (unsigned) r0);

  return 0;
}
