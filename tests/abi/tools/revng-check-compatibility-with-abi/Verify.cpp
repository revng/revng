/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/RegisterOrder.h"

#include "ABIArtifactParser.h"
#include "Verify.h"

struct ABIVerificationToolErrorCategory : public std::error_category {
  virtual const char *name() const noexcept {
    return "ABIVerificationToolErrorCategory";
  }
  virtual std::error_condition
  default_error_condition(int Code) const noexcept {
    return std::error_condition(Code, *this);
  }

  virtual bool
  equivalent(int Code, const std::error_condition &Condition) const noexcept {
    return default_error_condition(Code) == Condition;
  }

  virtual bool
  equivalent(const std::error_code &Code, int Condition) const noexcept {
    return *this == Code.category() && Code.value() == Condition;
  }
  virtual std::string message(int) const { return ""; }
};

const std::error_category &thisToolError() {
  static ABIVerificationToolErrorCategory Instance;
  return Instance;
}

struct VerificationHelper {
public:
  llvm::Error verify(const abi::FunctionType::Layout &FunctionLayout,
                     const abi::artifact::FunctionArtifact &Artifact) const;

public:
  const model::Architecture::Values Architecture;
  const model::ABI::Values ABI;
  const bool IsLittleEndian;

public:
  llvm::StringRef FunctionName = "";
};

struct IterationAccessHelper {
public:
  IterationAccessHelper(const abi::artifact::Iteration &Iteration,
                        const model::Architecture::Values Architecture) :
    Iteration(Iteration),
    RegisterSize(model::Architecture::getPointerSize(Architecture)) {}

  std::optional<llvm::ArrayRef<std::byte>>
  registerValue(model::Register::Values Register) {
    auto Iterator = Iteration.Registers.find(Register);
    if (Iterator != Iteration.Registers.end())
      return Iterator->Bytes;
    else
      return std::nullopt;
  }

public:
  const abi::artifact::Iteration &Iteration;
  const size_t RegisterSize;
};

using VH = VerificationHelper;
llvm::Error VH::verify(const abi::FunctionType::Layout &FunctionLayout,
                       const abi::artifact::FunctionArtifact &Artifact) const {
  for (size_t Index = 0; Index < Artifact.Iterations.size(); ++Index) {
    IterationAccessHelper Helper(Artifact.Iterations[Index], Architecture);

    // Sort the argument registers
    auto ARegisters = FunctionLayout.argumentRegisters();
    auto OrderedRegisterList = abi::orderArguments(std::move(ARegisters), ABI);

    // Compute the relevant stack slice.
    abi::FunctionType::Layout::Argument::StackSpan StackSpan{ 0, 0 };
    for (const auto &Argument : FunctionLayout.Arguments) {
      if (Argument.Stack.has_value()) {
        revng_assert(Argument.Stack->Size != 0);
        if (StackSpan.Size == 0) {
          StackSpan = *Argument.Stack;
        } else {
          if (Argument.Stack->Offset != StackSpan.Offset + StackSpan.Size)
            return ERROR(ExitCode::OnlyContinuousStackArgumentsAreSupported,
                         "Only continuous stack arguments are supported.\n");

          StackSpan.Size += Argument.Stack->Size;
        }
      }
    }
    llvm::ArrayRef<std::byte> StackBytes = Helper.Iteration.Stack;
    StackBytes = StackBytes.slice(StackSpan.Offset, StackSpan.Size);
    if (StackBytes.size() != StackSpan.Size)
      return ERROR(ExitCode::InsufficientStackSize,
                   "The piece of stack provided by the artifact is "
                   "insufficient to hold all the arguments.\n");

    // Check the arguments.
    size_t CurrentRegisterIndex = 0;
    for (const auto &Argument : Helper.Iteration.Arguments) {
      llvm::ArrayRef<std::byte> ArgumentBytes = Argument.Bytes;
      if (ArgumentBytes.empty())
        return ERROR(ExitCode::NoArgumentBytesProvided,
                     "The `Bytes` field of the artifact is empty for one of "
                     "the arguments of '"
                       + FunctionName + "' function.\n");

      do {
        if (CurrentRegisterIndex < OrderedRegisterList.size()) {
          auto CurrentRegister = OrderedRegisterList[CurrentRegisterIndex];
          auto RegValue = Helper.registerValue(CurrentRegister);
          if (!RegValue.has_value())
            return ERROR(ExitCode::UnknownArgumentRegister,
                         "Verification of '"
                           + model::Register::getName(CurrentRegister)
                           + "' register failed. It's not specified by the "
                             "artifact.\n");

          if (ArgumentBytes.take_front(RegValue->size()).equals(*RegValue)) {
            ++CurrentRegisterIndex;
            ArgumentBytes = ArgumentBytes.drop_front(RegValue->size());

            // Current register value checks out, go to the next one.
            continue;
          } else if (auto RegPiece = RegValue->take_front(ArgumentBytes.size());
                     RegPiece.equals(ArgumentBytes)) {
            ++CurrentRegisterIndex;
            ArgumentBytes = {};

            break; // The last part of the argument was found.
          }
        }

        if (!StackBytes.take_front(ArgumentBytes.size()).equals(ArgumentBytes))
          return ERROR(ExitCode::ArgumentCanNotBeLocated,
                       "An argument cannot be found, it uses neither the "
                       "expected stack part nor the expected registers.\n");

        // Stack matches.
        auto Min = model::Architecture::getPointerSize(Architecture);
        StackBytes = StackBytes.drop_front(std::max(ArgumentBytes.size(), Min));
        ArgumentBytes = {};
      } while (!ArgumentBytes.empty());
    }

    if (CurrentRegisterIndex != OrderedRegisterList.size())
      return ERROR(ExitCode::LeftoverArgumentRegistersDetected,
                   "Function signature indicates the need for more register "
                   "to pass an argument than the actual count.\n");

    if (!StackBytes.empty())
      return ERROR(ExitCode::CombinedStackArgumentsSizeIsWrong,
                   "Combined stack argument size is different from what was "
                   "expected.\n");

    // Check the return value.
    auto RVR = FunctionLayout.returnValueRegisters();
    auto ReturnValueRegisterList = abi::orderReturnValues(std::move(RVR), ABI);
    if (!Helper.Iteration.ReturnValue.Bytes.empty()) {
      if (ReturnValueRegisterList.size() == 0)
        return ERROR(ExitCode::FoundUnexpectedReturnValue,
                     "Found a return value that should not be there.\n");

      size_t UsedRegisterCounter = 0;
      llvm::ArrayRef ReturnValueBytes = Helper.Iteration.ReturnValue.Bytes;
      for (auto &CurrentRegister : ReturnValueRegisterList) {
        auto RegValue = Helper.registerValue(CurrentRegister);
        if (!RegValue.has_value())
          return ERROR(ExitCode::UnknownReturnValueRegister,
                       "Verification of '"
                         + model::Register::getName(CurrentRegister)
                         + "' register failed. It's not specified by the "
                           "artifact.\n");

        if (ReturnValueBytes.take_front(RegValue->size()).equals(*RegValue)) {
          ReturnValueBytes = ReturnValueBytes.drop_front(RegValue->size());
          ++UsedRegisterCounter;
        } else if (auto Piece = RegValue->take_front(ReturnValueBytes.size());
                   Piece.equals(ReturnValueBytes)) {
          ReturnValueBytes = {};
          ++UsedRegisterCounter;
        } else {
          return ERROR(ExitCode::ReturnValueCanNotBeLocated,
                       "Fail to locate where the return value is passed.\n");
        }
      }

      if (UsedRegisterCounter != ReturnValueRegisterList.size())
        return ERROR(ExitCode::LeftoverReturnValueRegistersDetected,
                     "Function signature indicates the need for more register "
                     "to return a value than the actual count.\n");

    } else if (ReturnValueRegisterList.size() != 0) {
      return ERROR(ExitCode::ExpectedReturnValueNotFound,
                   "A return value is expected but function signature doesn't "
                   "mention it.\n");
    }
  }

  return llvm::Error::success();
}

llvm::Error verifyABI(const TupleTree<model::Binary> &Binary,
                      llvm::StringRef RuntimeArtifact,
                      model::ABI::Values ABI) {
  model::Architecture::Values Architecture = model::ABI::getArchitecture(ABI);
  auto ParsedArtifact = abi::artifact::parse(RuntimeArtifact, Architecture);

  llvm::StringRef ArchitectureName = model::Architecture::getName(Architecture);
  revng_assert(ArchitectureName == ParsedArtifact.Architecture);
  VerificationHelper Helper{ Architecture, ABI, ParsedArtifact.IsLittleEndian };

  for (auto &Type : Binary->Types()) {
    revng_assert(Type.get() != nullptr);

    auto CurrentFunction = ParsedArtifact.Functions.find(Type->name());
    if (CurrentFunction == ParsedArtifact.Functions.end()) {
      // Ignore types not present in the artifact.
      continue;
    }

    Helper.FunctionName = Type->name();

    using Layout = abi::FunctionType::Layout;
    if (auto *CABI = llvm::dyn_cast<model::CABIFunctionType>(Type.get())) {
      if (auto Error = Helper.verify(Layout(*CABI), *CurrentFunction))
        return Error;
    } else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(Type.get())) {
      if (auto Error = Helper.verify(Layout(*Raw), *CurrentFunction))
        return Error;
    } else {
      // Ignore non-function types.
    }
  }

  return llvm::Error::success();
}
