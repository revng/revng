/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"

#include "ABIArtifactParser.h"

namespace State {

using namespace abi::artifact;

struct SingleRun {
  llvm::StringRef Function;
  Registers Registers;
  Stack Stack;
  Arguments ReturnValue;
  Arguments Arguments;
};

struct Deserialized {
  llvm::StringRef TargetArchitecture;
  bool IsLittleEndian;
  llvm::SmallVector<SingleRun, 16> Iterations;
};

} // namespace State

//
// `llvm::yaml` traits for the internal state.
// TODO: Consider using `TupleTreeGenerator` instead.
//

template<>
struct llvm::yaml::ScalarTraits<std::byte> {
  static_assert(sizeof(std::byte) == sizeof(uint8_t));

  static void output(const std::byte &Value, void *, llvm::raw_ostream &Out) {
    Out << uint8_t(Value);
  }

  static StringRef input(StringRef Scalar, void *Ptr, std::byte &Value) {
    uint8_t Temporary;
    auto Err = llvm::yaml::ScalarTraits<uint8_t>::input(Scalar, Ptr, Temporary);
    if (!Err.empty())
      return Err;

    Value = static_cast<std::byte>(Temporary);
    return StringRef{};
  }

  static QuotingType mustQuote(StringRef Scalar) {
    return llvm::yaml::ScalarTraits<uint8_t>::mustQuote(Scalar);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::Register> {
  static void mapping(IO &IO, State::Register &N) {
    IO.mapRequired("Name", N.Name);
    IO.mapRequired("Value", N.Value);
    IO.mapRequired("Bytes", N.Bytes);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::Argument> {
  static void mapping(IO &IO, State::Argument &N) {
    IO.mapRequired("Type", N.Type);
    IO.mapRequired("Bytes", N.Bytes);
    IO.mapOptional("Pointer", N.MaybePointer);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::SingleRun> {
  static void mapping(IO &IO, State::SingleRun &N) {
    IO.mapRequired("Function", N.Function);
    IO.mapRequired("Registers", N.Registers);
    IO.mapRequired("Stack", N.Stack);
    IO.mapRequired("Arguments", N.Arguments);
    IO.mapRequired("ReturnValue", N.ReturnValue);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::Deserialized> {
  static void mapping(IO &IO, State::Deserialized &N) {
    IO.mapRequired("TargetArchitecture", N.TargetArchitecture);
    IO.mapRequired("IsLittleEndian", N.IsLittleEndian);
    IO.mapRequired("Iterations", N.Iterations);
  }
};

template<>
struct llvm::yaml::SequenceElementTraits<std::byte> {
  static constexpr bool flow = true;
};
template<>
struct llvm::yaml::SequenceElementTraits<State::Register> {
  static constexpr bool flow = false;
};
template<>
struct llvm::yaml::SequenceElementTraits<State::Argument> {
  static constexpr bool flow = false;
};
template<>
struct llvm::yaml::SequenceElementTraits<State::SingleRun> {
  static constexpr bool flow = false;
};

struct VerificationHelper {
  llvm::StringRef FunctionName;
  size_t IterationCount = 0;
};

template<>
struct KeyedObjectTraits<VerificationHelper> {
  static llvm::StringRef key(const VerificationHelper &Object) {
    return Object.FunctionName;
  }
  static VerificationHelper fromKey(const llvm::StringRef &Key) {
    return { Key };
  }
};

static bool verify(const State::Deserialized &Data, bool ShouldAssert) {
  if (Data.Iterations.empty()) {
    revng_assert(!ShouldAssert);
    return false;
  }

  SortedVector<VerificationHelper> VH;
  for (const State::SingleRun &Run : Data.Iterations)
    if (auto Iterator = VH.find(Run.Function); Iterator == VH.end())
      VH.insert(VerificationHelper{ Run.Function, 1 });
    else
      ++Iterator->IterationCount;

  if (VH.empty()) {
    revng_assert(!ShouldAssert);
    return false;
  }

  size_t ExpectedIterationCount = VH.begin()->IterationCount;
  for (const VerificationHelper &Helper : VH) {
    if (ExpectedIterationCount != Helper.IterationCount) {
      revng_assert(!ShouldAssert);
      return false;
    }
  }

  return true;
}

static State::ModelRegisters
toModelRegisters(const State::Registers &Input,
                 model::Architecture::Values Arch) {
  State::ModelRegisters Result;

  for (const State::Register &Reg : Input) {
    auto &Output = Result[model::Register::fromRegisterName(Reg.Name, Arch)];
    Output.Value = Reg.Value;
    Output.Bytes = Reg.Bytes;
  }

  return Result;
}

static State::Iteration toIteration(const State::SingleRun &SingleRun,
                                    model::Architecture::Values Architecture) {
  revng_assert(SingleRun.ReturnValue.size() <= 1);
  return State::Iteration{ toModelRegisters(SingleRun.Registers, Architecture),
                           SingleRun.Stack,
                           SingleRun.Arguments,
                           SingleRun.ReturnValue.empty() ?
                             State::Argument{ "void", {}, {} } :
                             SingleRun.ReturnValue[0] };
}

State::Parsed abi::artifact::parse(llvm::StringRef RuntimeArtifact,
                                   model::Architecture::Values Architecture) {
  llvm::yaml::Input Reader(RuntimeArtifact);

  State::Deserialized Deserialized;
  Reader >> Deserialized;
  revng_assert(!Reader.error());

  llvm::StringRef ArchitectureName = model::Architecture::getName(Architecture);
  if (ArchitectureName != Deserialized.TargetArchitecture) {
    std::string Error = "Target architecture ('"
                        + Deserialized.TargetArchitecture.str()
                        + "') does not match the expected one: '"
                        + ArchitectureName.str() + "'\n";
    revng_abort(Error.c_str());
  }

  verify(Deserialized, true);

  State::Parsed Result;
  Result.Architecture = Deserialized.TargetArchitecture;
  Result.IsLittleEndian = Deserialized.IsLittleEndian;
  for (const State::SingleRun &SingleRun : Deserialized.Iterations) {
    State::FunctionArtifact &Reference = Result.Functions[SingleRun.Function];
    Reference.Iterations.emplace_back(toIteration(SingleRun, Architecture));
  }

  static_assert(TupleTreeCompatible<model::Binary>);
  return Result;
}
