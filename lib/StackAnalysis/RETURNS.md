# Approach

1. Identify all the return candidates [DONE]
   * Indirect jumps to the return address
   * Indirect tail calls, i.e., indirect jumps to unknown addresses with SP >= 0
2. Go through all the candidates and elect a return stack pointer
   * Mark all non-complaint ones as broken returns/indirect tail calls. They are *not* returns. [DONE]
   * Combine the result associated to each propore return/indirect tail call, that's the grand result. Mark the ABIIRBasicBlock as return. [DONE]

   a. If no proper stack pointer can be elected, but all the return points agree on a single SP <0, mark the function as outlined.
   b. If no proper stack pointer can be elected, mark the function as noreturn
   c. If a proper stack pointer has been elected, mark the function as regular.

# TODOs

* Indirect tail calls must become longjmps, or longjmps should be treated as indirect tail calls

