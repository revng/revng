/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "Verify.h"

//
// Internal state
//

namespace State {

struct Register {
  llvm::StringRef Name;
  uint64_t Value;
};
using Registers = llvm::SmallVector<Register, 32>;

struct StackValue {
  int64_t Offset;
  uint64_t Value;
};
using Stack = llvm::SmallVector<StackValue, 32>;

struct Argument {
  llvm::StringRef Type;
  llvm::StringRef Value;
};
using Arguments = llvm::SmallVector<Argument, 32>;

struct SingleRun {
  llvm::StringRef Function;
  Registers Registers;
  Stack Stack;
  Arguments ReturnValues;
  Arguments Arguments;
};
using Deserialized = llvm::SmallVector<SingleRun, 16>;

} // namespace State

//
// `llvm::yaml` traits for the internal state.
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
  }
};

template<>
struct llvm::yaml::MappingTraits<State::SingleRun> {
  static void mapping(IO &IO, State::SingleRun &N) {
    IO.mapRequired("Function", N.Function);
    IO.mapRequired("Registers", N.Registers);
    IO.mapRequired("Stack", N.Stack);
    IO.mapRequired("Arguments", N.Arguments);
    IO.mapRequired("ReturnValues", N.ReturnValues);
  }
};

template<typename Type>
struct GenericSmallVectorYamlTrait {
  static size_t size(llvm::yaml::IO &, Type &Value) { return Value.size(); }
  static auto &element(llvm::yaml::IO &, Type &Value, size_t Index) {
    if (Index >= Value.size())
      Value.resize(Index + 1);
    return Value[Index];
  }
};

template<>
struct llvm::yaml::SequenceTraits<State::Registers>
  : GenericSmallVectorYamlTrait<State::Registers> {};
template<>
struct llvm::yaml::SequenceTraits<State::Stack>
  : GenericSmallVectorYamlTrait<State::Stack> {};
template<>
struct llvm::yaml::SequenceTraits<State::Arguments>
  : GenericSmallVectorYamlTrait<State::Arguments> {};
template<>
struct llvm::yaml::SequenceTraits<State::Deserialized>
  : GenericSmallVectorYamlTrait<State::Deserialized> {};

//
// Parsed State
//

namespace State {

struct ModelRegister {
  model::Register::Values Name;
  uint64_t Value = 0;
};

} // namespace State

template<>
struct KeyedObjectTraits<State::ModelRegister> {
  static model::Register::Values key(const State::ModelRegister &Object) {
    return Object.Name;
  }
  static State::ModelRegister fromKey(const model::Register::Values &Key) {
    return { Key };
  }
};

namespace State {

using ModelRegisters = SortedVector<ModelRegister>;

struct Iteration {
  ModelRegisters Registers;
  Stack Stack;
  Arguments ReturnValues;
  Arguments Arguments;
};

struct FunctionArtifact {
  llvm::StringRef Name;
  llvm::SmallVector<Iteration, 5> Iterations = {};
};

} // namespace State

template<>
struct KeyedObjectTraits<State::FunctionArtifact> {
  static llvm::StringRef key(const State::FunctionArtifact &Object) {
    return Object.Name;
  }
  static State::FunctionArtifact fromKey(const llvm::StringRef &Key) {
    return { Key };
  }
};

namespace State {

using Parsed = SortedVector<FunctionArtifact>;

} // namespace State

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
  return State::Iteration{ toModelRegisters(SingleRun.Registers, Architecture),
                           SingleRun.Stack,
                           SingleRun.ReturnValues,
                           SingleRun.Arguments };
}

static State::Parsed parse(llvm::StringRef RuntimeArtifact,
                           model::Architecture::Values Architecture) {
  llvm::yaml::Input Reader(RuntimeArtifact);

  State::Deserialized Deserialized;
  Reader >> Deserialized;

  State::Parsed Result;
  for (const State::SingleRun &SingleRun : Deserialized) {
    const State::Iteration &Iteration = toIteration(SingleRun, Architecture);
    Result[SingleRun.Function].Iterations.emplace_back(Iteration);
  }

  return Result;
}

//
// Verification
//

template<typename ValueType>
static bool
tryVerifyRegisterImpl(llvm::StringRef ArgumentValue, uint64_t RegisterValue) {
  ValueType Converted;
  bool Failed = ArgumentValue.getAsInteger(16, Converted);
  revng_assert(Failed == false);

  return static_cast<ValueType>(RegisterValue) == Converted;
}

static bool tryVerifyRegister(llvm::StringRef ArgumentValue,
                              uint64_t RegisterValue,
                              size_t RegisterSize) {
  bool Res = false;
  while (ArgumentValue.size() > RegisterSize * 2) {
    auto F8B = ArgumentValue.substr(0, RegisterSize * 2);
    Res = Res || tryVerifyRegister(F8B, RegisterValue, RegisterSize);
    ArgumentValue = ArgumentValue.substr(RegisterSize * 2);
  }

  switch (ArgumentValue.size()) {
  case 2:
    return Res || tryVerifyRegisterImpl<uint8_t>(ArgumentValue, RegisterValue);
  case 4:
    return Res || tryVerifyRegisterImpl<uint16_t>(ArgumentValue, RegisterValue);
  case 8:
    return Res || tryVerifyRegisterImpl<uint32_t>(ArgumentValue, RegisterValue);
  case 16:
    return Res || tryVerifyRegisterImpl<uint64_t>(ArgumentValue, RegisterValue);
  default:
    revng_abort("Unsupported argument size.");
  };
}

using ArgumentIndices = std::set<size_t>;
static ArgumentIndices verifyRegister(model::Register::Values Location,
                                      const State::FunctionArtifact &Artifact,
                                      size_t RegisterSize) {
  ArgumentIndices Result;

  bool IsFirst = true;
  for (const State::Iteration &Iteration : Artifact.Iterations) {
    const State::ModelRegister &Register = Iteration.Registers.at(Location);

    ArgumentIndices CurrentIterationResult;
    for (size_t Index = 0; Index < Iteration.Arguments.size(); ++Index) {
      auto ArgumentValue = Iteration.Arguments[Index].Value;
      if (ArgumentValue.size() > 2 && ArgumentValue.substr(0, 2) == "0x")
        ArgumentValue = ArgumentValue.substr(2);
      if (tryVerifyRegister(ArgumentValue, Register.Value, RegisterSize))
        CurrentIterationResult.insert(Index);
    }

    if (IsFirst) {
      std::swap(Result, CurrentIterationResult);
      IsFirst = false;
    } else {
      ArgumentIndices Temporary;
      std::set_intersection(Result.begin(),
                            Result.end(),
                            CurrentIterationResult.begin(),
                            CurrentIterationResult.end(),
                            std::inserter(Temporary, Temporary.begin()));
      std::swap(Result, Temporary);
    }
  }

  return Result;
}

static bool areArgumentsCompatible(const State::Argument &Expected,
                                   const model::Argument &Found,
                                   size_t RegisterSize) {
  llvm::StringRef ArgumentValue = Expected.Value;
  if (ArgumentValue.size() > 2 && ArgumentValue.substr(0, 2) == "0x")
    ArgumentValue = ArgumentValue.substr(2);

  std::optional<uint64_t> MaybeSize = Found.Type.size();
  revng_assert(MaybeSize != std::nullopt);

  if (ArgumentValue.size() == *MaybeSize * 2)
    return true; // Exact size.
  else if (*MaybeSize == RegisterSize
           && ArgumentValue.size() < RegisterSize * 2)
    return true; // Expanded to the register size.
  else
    return false;
}

ExitCode verifyABI(const TupleTree<model::Binary> &Binary,
                   llvm::StringRef RuntimeArtifact,
                   model::ABI::Values ABI,
                   llvm::raw_fd_ostream &OutputStream) {
  model::Architecture::Values Architecture = model::ABI::getArchitecture(ABI);
  size_t RegisterSize = model::Architecture::getPointerSize(Architecture);
  State::Parsed ParsedArtifact = parse(RuntimeArtifact, Architecture);

  for (auto &Type : Binary->Types) {
    ExitCode Result = ExitCode::Success;
    auto Visitor = [&]<typename UpcastedType>(const UpcastedType &Upcasted) {
      using namespace model;
      if constexpr (std::is_same_v<UpcastedType, CABIFunctionType>) {
        auto CurrentArtifact = ParsedArtifact.find(Upcasted.CustomName);
        if (CurrentArtifact != ParsedArtifact.end()) {
          for (auto &Iteration : CurrentArtifact->Iterations) {
            size_t ArgumentCount = Iteration.Arguments.size();
            if (ArgumentCount != Upcasted.Arguments.size()) {
              size_t ItID = std::distance(CurrentArtifact->Iterations.begin(),
                                          &Iteration);
              std::string_view NameView = llvm::StringRef(Upcasted.CustomName);
              dbg << ArgumentCount << " arguments expected, but "
                  << Upcasted.Arguments.size() << " were found instead "
                  << "on the iteration #" << ItID << " of a function named \""
                  << NameView << "\":\n";
              Upcasted.dump();
              Result = ExitCode::FailedArgumentCountCheck;
              return;
            }
            for (size_t Index = 0; Index < ArgumentCount; ++Index) {
              if (!areArgumentsCompatible(Iteration.Arguments[Index],
                                          Upcasted.Arguments.at(Index),
                                          RegisterSize)) {
                size_t ItID = std::distance(CurrentArtifact->Iterations.begin(),
                                            &Iteration);
                std::string_view NmView = llvm::StringRef(Upcasted.CustomName);
                dbg << "Argument " << Index << " is not compatible with "
                    << "the expected type on the iteration #" << ItID
                    << " of a function named \"" << NmView << "\":\n";
                Upcasted.dump();
                Result = ExitCode::FailedArgumentCompatibilityCheck;
                return;
              }
            }
          }
        }
      } else if constexpr (std::is_same_v<UpcastedType, RawFunctionType>) {
        auto CurrentArtifact = ParsedArtifact.find(Upcasted.CustomName);
        if (CurrentArtifact != ParsedArtifact.end()) {
          for (auto &Argument : Upcasted.Arguments) {
            auto Indices = verifyRegister(Argument.Location,
                                          *CurrentArtifact,
                                          RegisterSize);
            if (Indices.empty()) {
              std::string_view NameView = llvm::StringRef(Upcasted.CustomName);
              dbg << "Cannot locate an argument in the artifact:\n";
              Argument.dump();
              dbg << "The function named \"" << NameView << "\":\n";
              Upcasted.dump();
              Result = ExitCode::FailedLocatingAnArgument;
              return;
            }

            if (Indices.size() > 1) {
              std::string_view NameView = llvm::StringRef(Upcasted.CustomName);
              dbg << "Multiple locations for an argument in the artifact:\n";
              Argument.dump();
              dbg << "The locations are: ";
              for (auto Index : Indices)
                dbg << Index << ' ';
              dbg << "\nThe function named \"" << NameView << "\":\n";
              Upcasted.dump();
              Result = ExitCode::FailedSelectingSingleArgumentLocation;
              return;
            }
          }
        }
      }
    };
    Type.upcast(Visitor);
    if (Result != ExitCode::Success)
      return Result;
  }

  return ExitCode::Success;
}
