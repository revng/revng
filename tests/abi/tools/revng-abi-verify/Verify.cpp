/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/FunctionType.h"
#include "revng/ABI/RegisterOrder.h"

#include "ABIArtifactParser.h"
#include "Verify.h"

struct ABIVerificationToolErrorCategory : public std::error_category {
  virtual const char *name() const noexcept {
    return "ABIVerificationToolErrorCategory";
  }
  virtual std::error_condition
  default_error_condition(int code) const noexcept {
    return std::error_condition(code, *this);
  }
  virtual bool
  equivalent(int code, const std::error_condition &condition) const noexcept {
    return default_error_condition(code) == condition;
  }
  virtual bool
  equivalent(const std::error_code &code, int condition) const noexcept {
    return *this == code.category() && code.value() == condition;
  }
  virtual std::string message(int) const { return ""; }
};

const std::error_category &thisToolError() {
  static ABIVerificationToolErrorCategory Instance;
  return Instance;
}

template<typename ValueType>
static bool
compareImpl(llvm::StringRef ArgumentValue, ValueType RegisterValue) {
  revng_assert(ArgumentValue.size() == sizeof(ValueType) * 2);

  ValueType Converted;
  bool Failed = ArgumentValue.getAsInteger(16, Converted);
  revng_assert(Failed == false);

  return RegisterValue == Converted;
}

static bool
tryVerifyRegister(llvm::StringRef ArgumentValue, uint64_t RegisterValue) {
  switch (ArgumentValue.size()) {
  case 2:
    return compareImpl<uint8_t>(ArgumentValue, RegisterValue)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 8)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 16)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 24)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 32)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 40)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 48)
           || compareImpl<uint8_t>(ArgumentValue, RegisterValue >> 56);
  case 4:
    return compareImpl<uint16_t>(ArgumentValue, RegisterValue)
           || compareImpl<uint16_t>(ArgumentValue, RegisterValue >> 16)
           || compareImpl<uint16_t>(ArgumentValue, RegisterValue >> 32)
           || compareImpl<uint16_t>(ArgumentValue, RegisterValue >> 48);
  case 8:
    return compareImpl<uint32_t>(ArgumentValue, RegisterValue)
           || compareImpl<uint32_t>(ArgumentValue, RegisterValue >> 32);
  case 16:
    return compareImpl<uint64_t>(ArgumentValue, RegisterValue);
  default:
    revng_abort("Unsupported argument size.");
  };
}

static llvm::SmallVector<llvm::StringRef, 8>
splitValue(llvm::StringRef Value, size_t ChunkSize) {
  if (Value.size() > 1 && Value.substr(0, 2) == "0x")
    Value = Value.substr(2);

  size_t ExpectedChunkCount = Value.size() / ChunkSize
                              + (Value.size() % ChunkSize != 0);

  llvm::SmallVector<llvm::StringRef, 8> Result;

  while (Value.size() > ChunkSize) {
    Result.emplace_back(Value.take_back(ChunkSize));
    Value = Value.drop_back(ChunkSize);
  }

  revng_assert(!Value.empty());
  Result.emplace_back(std::move(Value));

  revng_assert(ExpectedChunkCount == Result.size());
  return Result;
}

struct IterationAccessHelper {
  const abi::artifact::Iteration Iteration;
  size_t RegisterSize;

  std::optional<uint64_t> registerValue(model::Register::Values Register) {
    auto Iterator = Iteration.Registers.find(Register);
    if (Iterator != Iteration.Registers.end())
      return Iterator->Value;
    else
      return std::nullopt;
  }

  std::optional<uint64_t> stackValue(size_t OffsetIndex) {
    if (OffsetIndex >= Iteration.Stack.size())
      return std::nullopt;

    auto &Result = Iteration.Stack[OffsetIndex];
    revng_assert(Result.Offset >= 0);
    if (static_cast<uint64_t>(Result.Offset) != OffsetIndex * RegisterSize)
      return std::nullopt;
    else
      return Result.Value;
  }
};

static llvm::Error verifyImpl(const abi::FunctionType::Layout &FunctionLayout,
                              const abi::artifact::FunctionArtifact &Artifact,
                              model::ABI::Values ABI) {
  model::Architecture::Values Architecture = model::ABI::getArchitecture(ABI);
  size_t RegisterSize = model::Architecture::getPointerSize(Architecture);
  for (size_t Index = 0; Index < Artifact.Iterations.size(); ++Index) {
    IterationAccessHelper Helper{ Artifact.Iterations[Index], RegisterSize };

    // Get the data ready
    auto ArgRegisters = FunctionLayout.argumentRegisters();
    auto OrderedRegisterList = abi::orderArguments(std::move(ArgRegisters),
                                                   ABI);
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

    // Check the arguments.
    size_t CurrentRegisterIndex = 0;
    size_t UsedStackOffset = StackSpan.Offset;
    for (const auto &Argument : Helper.Iteration.Arguments) {
      revng_assert(Argument.Value.size() % 2 == 0);
      auto RegisterSizedChunks = splitValue(Argument.Value, RegisterSize * 2);
      if (!Artifact.IsLittleEndian)
        std::reverse(RegisterSizedChunks.begin(), RegisterSizedChunks.end());

      bool UsesStack = false;
      for (size_t Index = 0; Index < RegisterSizedChunks.size(); ++Index) {
        llvm::StringRef Chunk = RegisterSizedChunks[Index];
        if (!UsesStack && CurrentRegisterIndex < OrderedRegisterList.size()) {
          auto CurrentRegister = OrderedRegisterList[CurrentRegisterIndex];
          auto MaybeValue = Helper.registerValue(CurrentRegister);
          if (!MaybeValue.has_value())
            return ERROR(ExitCode::UnknownRegister,
                         "Verification of '"
                           + model::Register::getName(CurrentRegister)
                           + "' register failed. It doesn't "
                             "contain the expected argument.\n");

          if (tryVerifyRegister(Chunk, *MaybeValue)) {
            ++CurrentRegisterIndex;

            // Current register value checks out, go to the next one.
            continue;
          }
        }

        size_t OffsetCorrection = !Artifact.IsLittleEndian ?
                                    RegisterSizedChunks.size() - Index * 2 - 1 :
                                    0;
        auto MaybeValue = Helper.stackValue(UsedStackOffset + OffsetCorrection);
        if (!MaybeValue.has_value())
          return ERROR(ExitCode::UnknownStackOffset,
                       "Verification of '{0}' stack offset failed. It isn't "
                       "mentioned in the artifact. Are there too many "
                       "arguments?.\n",
                       UsedStackOffset);

        if (!tryVerifyRegister(Chunk, *MaybeValue))
          return ERROR(ExitCode::ArgumentCouldNotBeLocated,
                       "An argument cannot be found, it doesn't use either "
                       "stack nor registers.\n");

        UsesStack = true;
        ++UsedStackOffset;
      }
    }

    if (UsedStackOffset * RegisterSize != StackSpan.Offset + StackSpan.Size)
      return ERROR(ExitCode::CombinedStackArgumentsSizeIsWrong,
                   "Combined stack argument size is different from what was "
                   "expected.\n");

    // Check the return value.
    auto Registers = FunctionLayout.returnValueRegisters();
    auto ReturnValueRegisterList = abi::orderReturnValues(std::move(Registers),
                                                          ABI);
    if (Helper.Iteration.ReturnValue.Value.size() != 0) {
      if (ReturnValueRegisterList.size() == 0)
        return ERROR(ExitCode::FoundUnexpectedReturnValue,
                     "Found a return value that should not be there.\n");

      revng_assert(Helper.Iteration.ReturnValue.Value.size() % 2 == 0);
      auto RegisterSizedChunks = splitValue(Helper.Iteration.ReturnValue.Value,
                                            RegisterSize * 2);

      // TODO: investigate how endianness affects return values more thoroughly.
      if (!Artifact.IsLittleEndian)
        std::reverse(RegisterSizedChunks.begin(), RegisterSizedChunks.end());

      size_t ReturnValueRegisterIndex = 0;
      for (llvm::StringRef Chunk : RegisterSizedChunks) {
        auto Current = ReturnValueRegisterList[ReturnValueRegisterIndex];
        auto MaybeValue = Helper.registerValue(Current);
        if (!MaybeValue.has_value())
          return ERROR(ExitCode::UnknownReturnValueRegister,
                       "Verification of '" + model::Register::getName(Current)
                         + "' register failed. It doesn't "
                           "contain the expected return value.\n");

        if (!tryVerifyRegister(Chunk, *MaybeValue))
          return ERROR(ExitCode::ReturnValueCouldNotBeLocated,
                       "Fail to locate where the return value is passed.\n");

        ++ReturnValueRegisterIndex;
      }
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

  for (auto &Type : Binary->Types) {
    revng_assert(Type.get() != nullptr);

    auto CurrentArtifact = ParsedArtifact.find(Type->name());
    if (CurrentArtifact == ParsedArtifact.end()) {
      // Ignore types with no artifact provided.
      continue;
    }

    if (auto *CABI = llvm::dyn_cast<model::CABIFunctionType>(Type.get())) {
      if (auto Error = verifyImpl(abi::FunctionType::Layout(*CABI),
                                  *CurrentArtifact,
                                  ABI))
        return Error;
    } else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(Type.get())) {
      if (auto Error = verifyImpl(abi::FunctionType::Layout(*Raw),
                                  *CurrentArtifact,
                                  ABI))
        return Error;
    } else {
      // Ignore non-function types.
    }
  }

  return llvm::Error::success();
}
