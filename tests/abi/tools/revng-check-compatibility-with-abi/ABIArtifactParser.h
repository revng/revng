#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Register.h"

namespace abi::artifact {

struct Register {
  llvm::StringRef Name;
  uint64_t Value;
  std::vector<std::byte> Bytes;
};
using Registers = llvm::SmallVector<Register, 32>;
using Stack = llvm::SmallVector<std::byte, 32 * 8>;

struct Argument {
  llvm::StringRef Type;
  std::vector<std::byte> Bytes;
  uint64_t MaybePointer;
};
using Arguments = llvm::SmallVector<Argument, 32>;

struct ModelRegister {
  model::Register::Values Name;
  uint64_t Value = 0;
  std::vector<std::byte> Bytes = {};
};

} // namespace abi::artifact

template<>
struct KeyedObjectTraits<abi::artifact::ModelRegister> {
  static model::Register::Values
  key(const abi::artifact::ModelRegister &Object) {
    return Object.Name;
  }
  static abi::artifact::ModelRegister
  fromKey(const model::Register::Values &Key) {
    return { Key };
  }
};

namespace abi::artifact {

using ModelRegisters = SortedVector<ModelRegister>;

struct Iteration {
  ModelRegisters Registers;
  Stack Stack;
  Arguments Arguments;
  Argument ReturnValue;
};

struct FunctionArtifact {
  llvm::StringRef Name;
  llvm::SmallVector<Iteration, 5> Iterations = {};
};

} // namespace abi::artifact

template<>
struct KeyedObjectTraits<abi::artifact::FunctionArtifact> {
  static llvm::StringRef key(const abi::artifact::FunctionArtifact &Object) {
    return Object.Name;
  }
  static abi::artifact::FunctionArtifact fromKey(const llvm::StringRef &Key) {
    return { Key };
  }
};

namespace abi::artifact {

struct Parsed {
  llvm::StringRef Architecture;
  bool IsLittleEndian = false;
  SortedVector<FunctionArtifact> Functions;
};

Parsed parse(llvm::StringRef RuntimeArtifact,
             model::Architecture::Values Architecture);

} // namespace abi::artifact
