/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/FunctionType.h"
#include "revng/Model/Binary.h"

#include "ABIArtifactParser.h"

namespace State {

using namespace abi::artifact;

struct SingleRun {
  llvm::StringRef Function;
  bool IsLittleEndian;
  Registers Registers;
  Stack Stack;
  Arguments ReturnValue;
  Arguments Arguments;
};
using Deserialized = llvm::SmallVector<SingleRun, 16>;

} // namespace State

//
// `llvm::yaml` traits for the internal state.
// TODO: Consider using `TupleTreeGenerator` instead.
//

template<>
struct llvm::yaml::MappingTraits<State::Register> {
  static void mapping(IO &IO, State::Register &N) {
    IO.mapRequired("Name", N.Name);
    IO.mapRequired("Value", N.Value);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::StackValue> {
  static void mapping(IO &IO, State::StackValue &N) {
    IO.mapRequired("Offset", N.Offset);
    IO.mapRequired("Value", N.Value);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::Argument> {
  static void mapping(IO &IO, State::Argument &N) {
    IO.mapRequired("Type", N.Type);
    IO.mapRequired("Value", N.Value);
    IO.mapRequired("Address", N.Address);
  }
};

template<>
struct llvm::yaml::MappingTraits<State::SingleRun> {
  static void mapping(IO &IO, State::SingleRun &N) {
    IO.mapRequired("Function", N.Function);
    IO.mapRequired("IsLittleEndian", N.IsLittleEndian);
    IO.mapRequired("Registers", N.Registers);
    IO.mapRequired("Stack", N.Stack);
    IO.mapRequired("Arguments", N.Arguments);
    IO.mapRequired("ReturnValue", N.ReturnValue);
  }
};

template<>
struct llvm::yaml::SequenceElementTraits<State::Register> {
  static constexpr bool flow = false;
};
template<>
struct llvm::yaml::SequenceElementTraits<State::StackValue> {
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
  bool IsLittleEndian = false;
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
  if (Data.empty()) {
    revng_assert(!ShouldAssert);
    return false;
  }

  SortedVector<VerificationHelper> VH;
  for (const State::SingleRun &Run : Data) {
    if (auto Iterator = VH.find(Run.Function); Iterator == VH.end()) {
      VH.insert(VerificationHelper{ Run.Function, Run.IsLittleEndian, 1 });
    } else {
      ++Iterator->IterationCount;
      if (Run.IsLittleEndian != Iterator->IsLittleEndian) {
        revng_assert(!ShouldAssert);
        return false;
      }
    }
  }

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
                             State::Argument{ "void", "", 0 } :
                             SingleRun.ReturnValue[0] };
}

State::Parsed abi::artifact::parse(llvm::StringRef RuntimeArtifact,
                                   model::Architecture::Values Architecture) {
  llvm::yaml::Input Reader(RuntimeArtifact);

  State::Deserialized Deserialized;
  Reader >> Deserialized;
  revng_assert(!Reader.error());
  verify(Deserialized, true);

  State::Parsed Result;
  for (const State::SingleRun &SingleRun : Deserialized) {
    abi::artifact::FunctionArtifact &Reference = Result[SingleRun.Function];
    Reference.IsLittleEndian = SingleRun.IsLittleEndian;
    Reference.Iterations.emplace_back(toIteration(SingleRun, Architecture));
  }

  static_assert(TupleTreeCompatible<model::Binary>);
  return Result;
}
