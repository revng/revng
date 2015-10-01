#include <iostream>

/// Writes to a stream the string representation of the PTC instruction with the
/// specified index withing the instruction list.
///
/// @param Result the output stream.
/// @param Instructions the instruction list.
/// @param Index the index of the target instruction in the instruction list.
///
/// @return EXIT_SUCCESS in case of success, EXIT_FAILURE otherwise.
int dumpInstruction(std::ostream& Result, PTCInstructionList *Instructions,
                    unsigned Index);

/// Write to a stream all the instructions in an instruction list.
///
/// @param Result the output stream.
/// @param Instructions the instruction list.
///
/// @return EXIT_SUCCESS in case of success, EXIT_FAILURE otherwise.
int dumpTranslation(std::ostream& Result, PTCInstructionList *Instructions);

/// Write to a stream the dissasembled version of the instruction at the
/// specified program counter.
///
/// @param Result the output stream
/// @param PC the program counter in the current context.
void disassembleOriginal(std::ostream& Result, uint64_t PC);
