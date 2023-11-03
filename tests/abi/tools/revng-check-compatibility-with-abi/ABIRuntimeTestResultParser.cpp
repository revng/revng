/// \file Verify.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"

#include "ABIRuntimeTestResultParser.h"

namespace abi::runtime_test {

struct RawState {
  std::vector<RawWord> Registers;
  std::vector<std::byte> Stack;

  bool empty() const { return Registers.empty() && Stack.empty(); }

  State extract(model::Architecture::Values Architecture) {
    State Result;
    for (auto Register : model::Architecture::registers(Architecture)) {
      constexpr auto PoN = model::PrimitiveTypeKind::PointerOrNumber;
      if (model::Register::primitiveKind(Register) != PoN) {
        // Only support generic registers for now.
        // TODO: add vector register support.
        continue;
      }

      auto Iterator = llvm::find_if(Registers, [Register](const RawWord &Raw) {
        return Raw.Name == getRegisterName(Register);
      });
      if (Iterator == Registers.end()) {
        std::string Error = "Unable to find '" + getRegisterName(Register).str()
                            + "' in the artifact.\n";
        revng_abort(Error.c_str());
      }
      Result.Registers.try_emplace(Register, Iterator->Value, Iterator->Bytes);
    }

    Result.Stack = std::move(Stack);
    return Result;
  }
};

struct Iteration {
  // Basic information.
  llvm::StringRef Function;
  uint64_t Index;

  // General state.
  RawState StateBeforeTheCall;
  RawState StateAfterTheCall;
  RawState StateAfterTheReturn;

  // Test specific.
  std::vector<RawWord> ReturnValueAddress; // Return value tests only.
  std::vector<Argument> ReturnValue; // Return value tests only.
  std::vector<Argument> ExpectedReturnValue; // Return value tests only.
  std::vector<Argument> Arguments; // Argument tests only.

  bool isArgumentTest() const {
    return !Arguments.empty() && ReturnValueAddress.empty()
           && ReturnValue.empty() && ExpectedReturnValue.empty();
  }

  bool isReturnValueTest() const {
    return Arguments.empty() && !ReturnValueAddress.empty()
           && !ReturnValue.empty() && !ExpectedReturnValue.empty();
  }
};

struct Deserialized {
  // TODO: now that we have a lot more control over the revng-qa side (thanks
  //       to the interrupt handler, the possibilities are limitless), we
  //       definitely want to shift this struct a lot closer towards
  //       `TestedFunctions`, so that we don't need as much preprocessing here.
  llvm::StringRef TargetArchitecture;
  bool IsLittleEndian;
  std::vector<Iteration> Iterations;
};

} // namespace abi::runtime_test

//
// `llvm::yaml` traits for the internal state.
// TODO: Consider using `TupleTreeGenerator` instead.
//

template<>
struct llvm::yaml::MappingTraits<abi::runtime_test::RawWord> {
  static void mapping(IO &IO, abi::runtime_test::RawWord &N) {
    IO.mapRequired("Name", N.Name);
    IO.mapRequired("Value", N.Value);
    IO.mapRequired("Bytes", N.Bytes);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(abi::runtime_test::RawWord)

template<>
struct llvm::yaml::MappingTraits<abi::runtime_test::RawState> {
  static void mapping(IO &IO, abi::runtime_test::RawState &N) {
    IO.mapRequired("Registers", N.Registers);
    IO.mapRequired("Stack", N.Stack);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(abi::runtime_test::RawState);

template<>
struct llvm::yaml::MappingTraits<abi::runtime_test::Argument> {
  static void mapping(IO &IO, abi::runtime_test::Argument &N) {
    IO.mapRequired("Type", N.Type);
    IO.mapRequired("Bytes", N.Bytes);
    IO.mapOptional("Pointer", N.MaybePointer);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(abi::runtime_test::Argument);

template<>
struct llvm::yaml::MappingTraits<abi::runtime_test::Iteration> {
  static void mapping(IO &IO, abi::runtime_test::Iteration &N) {
    IO.mapRequired("Function", N.Function);
    IO.mapRequired("Iteration", N.Index);

    IO.mapOptional("StateBeforeTheCall", N.StateBeforeTheCall);
    IO.mapOptional("StateAfterTheCall", N.StateAfterTheCall);
    IO.mapOptional("StateAfterTheReturn", N.StateAfterTheReturn);

    IO.mapOptional("ReturnValueAddress", N.ReturnValueAddress);
    IO.mapOptional("ReturnValue", N.ReturnValue);
    IO.mapOptional("ExpectedReturnValue", N.ExpectedReturnValue);
    IO.mapOptional("Arguments", N.Arguments);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(abi::runtime_test::Iteration);

template<>
struct llvm::yaml::MappingTraits<abi::runtime_test::Deserialized> {
  static void mapping(IO &IO, abi::runtime_test::Deserialized &N) {
    IO.mapRequired("TargetArchitecture", N.TargetArchitecture);
    IO.mapRequired("IsLittleEndian", N.IsLittleEndian);
    IO.mapRequired("Iterations", N.Iterations);
  }
};

static void verify(const abi::runtime_test::Deserialized &Data) {
  if (Data.Iterations.empty())
    revng_abort("Nothing to test. Did the runner execute successfully?");

  struct Counts {
    size_t ArgumentIterations = 0;
    size_t ReturnValueIterations = 0;
    size_t full() const { return ArgumentIterations + ReturnValueIterations; }
  };

  std::map<llvm::StringRef, Counts> Counter;
  for (const abi::runtime_test::Iteration &Run : Data.Iterations) {
    if (!Run.Arguments.empty())
      ++Counter[Run.Function].ArgumentIterations;
    if (!Run.ReturnValue.empty())
      ++Counter[Run.Function].ReturnValueIterations;
  }

  size_t ExpectedIterationCount = Counter.begin()->second.full();
  for (auto [_, Counts] : Counter) {
    bool IsAnArgumentTest = Counts.ArgumentIterations != 0;
    bool IsAnReturnValueTest = Counts.ReturnValueIterations != 0;
    if (IsAnArgumentTest == IsAnReturnValueTest) {
      if (IsAnArgumentTest)
        revng_abort("Argument and return value tests must be separate.");
      else
        revng_abort("Tests must do something.");
    }

    if (Counts.full() != ExpectedIterationCount)
      revng_abort("Iteration count is not consistent.");
  }
}

abi::runtime_test::TestedFunctions
abi::runtime_test::parse(llvm::StringRef RuntimeArtifact,
                         model::Architecture::Values Architecture) {
  llvm::yaml::Input Reader(RuntimeArtifact);

  abi::runtime_test::Deserialized Deserialized;
  Reader >> Deserialized;
  revng_assert(!Reader.error());
  verify(Deserialized);

  llvm::StringRef ArchitectureName = model::Architecture::getName(Architecture);
  if (ArchitectureName != Deserialized.TargetArchitecture) {
    std::string Error = "Target architecture ('"
                        + Deserialized.TargetArchitecture.str()
                        + "') does not match the expected one: '"
                        + ArchitectureName.str() + "'\n";
    revng_abort(Error.c_str());
  }

  abi::runtime_test::TestedFunctions Fs;
  Fs.Architecture = Deserialized.TargetArchitecture;
  Fs.IsLittleEndian = Deserialized.IsLittleEndian;

  for (auto &I : Deserialized.Iterations) {
    llvm::StringRef Name = I.Function;

    revng_assert(I.isArgumentTest() != I.isReturnValueTest());
    if (I.isArgumentTest()) {
      auto &Test = Fs.ArgumentTests[Name].emplace_back();
      Test.StateBeforeTheCall = I.StateBeforeTheCall.extract(Architecture);
      Test.StateAfterTheCall = I.StateAfterTheCall.extract(Architecture);
      Test.StateAfterTheReturn = I.StateAfterTheReturn.extract(Architecture);

      Test.Arguments = I.Arguments;
    } else {
      auto &Test = Fs.ReturnValueTests[Name].emplace_back();
      Test.StateBeforeTheCall = I.StateBeforeTheCall.extract(Architecture);
      Test.StateAfterTheCall = I.StateAfterTheCall.extract(Architecture);
      Test.StateAfterTheReturn = I.StateAfterTheReturn.extract(Architecture);

      revng_assert(I.ReturnValueAddress.size() == 1);
      Test.ReturnValueAddress = I.ReturnValueAddress[0];

      revng_assert(I.ReturnValue.size() == 1);
      Test.ReturnValue = I.ReturnValue[0];

      revng_assert(I.ExpectedReturnValue.size() == 1);
      Test.ExpectedReturnValue = I.ExpectedReturnValue[0];
    }
  }

  return Fs;
}
