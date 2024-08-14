#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace model::Architecture {
enum Values;
}

namespace abi::runtime_test {

struct Argument {
  llvm::StringRef Type;
  uint64_t Address;
  std::vector<std::byte> AddressBytes;
  std::vector<std::byte> ExpectedBytes;
  std::vector<std::byte> FoundBytes;
};

struct RawWord {
  llvm::StringRef Name;
  uint64_t Value;
  std::vector<std::byte> Bytes;
};

struct Register {
  uint64_t Value;
  std::vector<std::byte> Bytes;

  Register(uint64_t Value = 0, std::vector<std::byte> Bytes = {}) :
    Value(Value), Bytes(Bytes) {}
};

struct State {
  std::unordered_map<model::Register::Values, Register> Registers;
  std::vector<std::byte> Stack;
};

struct ArgumentTest {
  State StateBeforeTheCall;
  State StateAfterTheCall;
  State StateAfterTheReturn;

  std::vector<Argument> Arguments;
};
struct ReturnValueTest {
  State StateBeforeTheCall;
  State StateAfterTheCall;
  State StateAfterTheReturn;

  Argument ReturnValue;
};

struct TestedFunctions {
  llvm::StringRef Architecture;
  bool IsLittleEndian = false;
  std::map<llvm::StringRef, std::vector<ArgumentTest>> ArgumentTests;
  std::map<llvm::StringRef, std::vector<ReturnValueTest>> ReturnValueTests;
};

TestedFunctions parse(llvm::StringRef RuntimeArtifact,
                      model::Architecture::Values Architecture);

} // namespace abi::runtime_test
