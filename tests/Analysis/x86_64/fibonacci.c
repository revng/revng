/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

unsigned fibonacci(unsigned n) {
  if (n <= 1)
    return n;
  else
    return (fibonacci(n - 1) + fibonacci(n - 2));
}

unsigned _start(unsigned a) {
  return fibonacci(a);
}
