/// \file Verify.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"

#include "ABIRuntimeTestResultParser.h"
#include "Verify.h"

struct VerificationHelper {
  const model::Architecture::Values Architecture;
  const abi::Definition &ABI;
  const bool IsLittleEndian;

public:
  llvm::StringRef FunctionName = "";
  abi::FunctionType::Layout FunctionLayout = {};

public:
  void arguments(const abi::runtime_test::ArgumentTest &Test) const;
  void returnValue(const abi::runtime_test::ReturnValueTest &Test) const;

private:
  void verifyValuePreservation(llvm::ArrayRef<std::byte> ExpectedBytes,
                               llvm::ArrayRef<std::byte> FoundBytes) const;
  std::vector<std::byte>
  dropInterArgumentPadding(llvm::ArrayRef<std::byte> Bytes) const;

  struct LeftToVerify {
    llvm::ArrayRef<model::Register::Values> Registers;
    llvm::ArrayRef<std::byte> Stack;
  };
  LeftToVerify adjustForSPTAR(LeftToVerify Remaining) const;
  LeftToVerify verifyAnArgument(const abi::runtime_test::State &State,
                                const abi::runtime_test::Argument &Argument,
                                LeftToVerify Remaining,
                                uint64_t Index) const;

  bool tryToVerifyStack(llvm::ArrayRef<std::byte> &Bytes,
                        llvm::ArrayRef<std::byte> ExpectedBytes) const;
  uint64_t valueFromBytes(llvm::ArrayRef<std::byte> Input) const;

  [[noreturn]] void fail(std::string &&Message) const;
};

using VH = VerificationHelper;

void VH::fail(std::string &&Message) const {
  dbg << "Verification of '" << FunctionName.str() << "' failed.\n"
      << "The layout is:\n";
  FunctionLayout.dump();
  revng_abort(Message.c_str());
}

void VH::verifyValuePreservation(llvm::ArrayRef<std::byte> ExpectedBytes,
                                 llvm::ArrayRef<std::byte> FoundBytes) const {
  if (FoundBytes.size() != ExpectedBytes.size()) {
    fail("Return value size  is not consistent: something is *very* wrong with "
         "the test toolchain.");
  }

  uint64_t MatchingByteCount = 0;
  for (auto [FByte, EByte] : llvm::zip(FoundBytes, ExpectedBytes))
    if (FByte == EByte)
      ++MatchingByteCount;

  // Sadly, we cannot do a full comparison since return value structs often
  // contain padding, value of which is not guaranteed to be preserved,
  // causing the test to fail. We either need a way to tell if a specific byte
  // belongs to the padding or not (which would restrict stuff we can test,
  // for example, nested structs) or assume that non-perfect matches are
  // acceptable.
  // An example of a struct that exhibits this behaviour (i386):
  // struct ReturnValue {
  //   uint8_t b; // _Alignof(uint8_t) == 1
  //   // 3 invisible bytes here
  //   // (_Alignof(ReturnValue) - _Alignof(uint8_t) = 4 - 1 = 3)
  //   uint32_t a; // _Alignof(uint32_t) == 4
  // };
  //
  // So, when looking at the return value we detect, it looks something like
  // `[ 0x11, 0xXX, 0xXX, 0xXX, 0x22, 0x33, 0x44, 0x55 ]`
  // where `0xXX` is unuinitialized value that is not stable and changes from
  // execution to execution.
  //
  // But, luckily, by the definition of how such padding is introduced, we are
  // guaranteed that in the worst case it's going to occupy `half_the_size - 1`
  // bytes. So, we can at least enforce that. And, thanks to doing multiple
  // iterations for each tests, the chance of false positives is extremely low.
  if (MatchingByteCount * 2 < FoundBytes.size())
    fail("The value was lost during the call: something went *very* wrong.");
}

namespace runtime_test = abi::runtime_test;
std::vector<std::byte>
VH::dropInterArgumentPadding(llvm::ArrayRef<std::byte> Bytes) const {
  std::vector<std::byte> Result;
  uint64_t PreviousArgumentEndsAt = 0;
  for (const auto &Argument : FunctionLayout.Arguments) {
    if (Argument.Stack.has_value()) {
      revng_assert(Argument.Stack->Size != 0);
      if (Argument.Stack->Offset < PreviousArgumentEndsAt)
        fail("Stack arguments must not overlap");

      auto PaddingSize = Argument.Stack->Offset - PreviousArgumentEndsAt;
      if (PaddingSize > ABI.getPointerSize()) {
        // TODO: this check can be improved quite a bit by taking
        // `abi::Definition::ScalarTypes()` into the account.
        fail("Padding exceeds the register size.\n"
             "Current argument is expected at offset "
             + std::to_string(Argument.Stack->Offset)
             + " while the previous one ends at "
             + std::to_string(PreviousArgumentEndsAt) + "\n");
      }

      // Since the view we want is fragmented, we'd have to keep a "view of
      // views" which is not easy to work with. Let's just copy the relevant
      // bytes out instead.
      llvm::copy(Bytes.slice(Argument.Stack->Offset, Argument.Stack->Size),
                 std::back_inserter(Result));

      PreviousArgumentEndsAt = Argument.Stack->Offset + Argument.Stack->Size;
    }
  }

  return Result;
}

VH::LeftToVerify VH::adjustForSPTAR(LeftToVerify Remaining) const {
  if (FunctionLayout.hasSPTAR()) {
    // Account for the shadow pointer to the return value.
    revng_assert(not FunctionLayout.Arguments.empty());
    auto &ShadowArgument = FunctionLayout.Arguments[0];
    using namespace abi::FunctionType::ArgumentKind;
    revng_assert(ShadowArgument.Kind == ShadowPointerToAggregateReturnValue);
    if (ShadowArgument.Registers.size() == 1) {
      // It's in a register, drop one if needed.
      model::Register::Values Register = *ShadowArgument.Registers.begin();
      revng_assert(Register == ABI.ReturnValueLocationRegister());
      if (!Remaining.Registers.empty())
        if (Remaining.Registers.front() == ABI.ReturnValueLocationRegister())
          Remaining.Registers = Remaining.Registers.drop_front();

    } else if (ShadowArgument.Stack.has_value()) {
      // It's on the stack, drop enough bytes for a pointer from the front.
      revng_assert(ShadowArgument.Stack->Offset == 0);
      revng_assert(ShadowArgument.Stack->Size == ABI.getPointerSize());
      Remaining.Stack = Remaining.Stack.drop_front(ABI.getPointerSize());
    } else {
      fail("Layout is not valid, does it verify?");
    }
  }

  return Remaining;
}

bool VH::tryToVerifyStack(llvm::ArrayRef<std::byte> &Bytes,
                          llvm::ArrayRef<std::byte> ExpectedBytes) const {
  if (Bytes.take_front(ExpectedBytes.size()).equals(ExpectedBytes)) {
    uint64_t BytesToDrop = ABI.paddedSizeOnStack(ExpectedBytes.size());
    if (Bytes.size() >= BytesToDrop)
      Bytes = Bytes.drop_front(BytesToDrop);
    else
      Bytes = {};

    return true;
  }

  return false;
}

VH::LeftToVerify VH::verifyAnArgument(const runtime_test::State &State,
                                      const runtime_test::Argument &Argument,
                                      LeftToVerify Remaining,
                                      uint64_t Index) const {
  // Make sure the value is the same both before and after the call.
  verifyValuePreservation(Argument.ExpectedBytes, Argument.FoundBytes);

  // Check for the "pointer-to-copy" style arguments first.
  if (!Remaining.Registers.empty()) {
    llvm::ArrayRef Bytes = State.Registers.at(Remaining.Registers[0]).Bytes;
    if (Bytes.equals(Argument.AddressBytes)) {
      Remaining.Registers = Remaining.Registers.drop_front();
      return Remaining;
    }
  }

  // Check bytes one piece at a time, consuming those that match.
  llvm::ArrayRef<std::byte> ArgumentBytes = Argument.FoundBytes;
  while (!ArgumentBytes.empty()) {
    // If there are still unverified registers, try to verify the next one.
    if (!Remaining.Registers.empty()) {
      llvm::ArrayRef Bytes = State.Registers.at(Remaining.Registers[0]).Bytes;
      if (ArgumentBytes.take_front(Bytes.size()).equals(Bytes)) {
        // Current register value matches: drop found bytes and start looking
        // for the rest.
        ArgumentBytes = ArgumentBytes.drop_front(Bytes.size());
        Remaining.Registers = Remaining.Registers.drop_front();
        continue;
      } else if (Bytes.take_front(ArgumentBytes.size()).equals(ArgumentBytes)) {
        // Current register matches all the remaining bytes.
        Remaining.Registers = Remaining.Registers.drop_front();
        ArgumentBytes = {};

        break;
      }
    }

    // We're out of registers, turn to the stack next.
    if (ABI.ArgumentsArePositionBased()
        && Argument.FoundBytes.size() > ABI.getPointerSize()) {
      // Position based ABIs use pointer-to-copy semantics for stack too.
      // This verifies whether that's the case here.
      revng_assert(ArgumentBytes.equals(Argument.FoundBytes),
                   "Only a part of the argument got consumed? That's weird.");
      if (tryToVerifyStack(Remaining.Stack, Argument.AddressBytes)) {
        // Stack matches the pointer to the argument: mark the entire argument
        // as found.
        ArgumentBytes = {};
        break;
      }
    } else {
      if (tryToVerifyStack(Remaining.Stack, ArgumentBytes)) {
        // Stack matches, go to the next argument.
        break;
      }

      revng_assert(!ABI.ScalarTypes().empty());
      auto &BiggestScalarType = *std::prev(ABI.ScalarTypes().end());
      if (BiggestScalarType.alignedAt() != ABI.getPointerSize()) {
        // If the ABI supports unusual alignment, try to account for it,
        // by dropping an conflicting part of the stack data.
        if (Remaining.Stack.size() < ABI.getPointerSize())
          Remaining.Stack = {};
        else
          Remaining.Stack = Remaining.Stack.drop_front(ABI.getPointerSize());
      }

      if (tryToVerifyStack(Remaining.Stack, ArgumentBytes)) {
        // Stack matches after accounting for alignment.
        break;
      }
    }

    fail("Argument #" + std::to_string(Index)
         + " uses neither the expected stack part nor the expected "
           "registers.");
  }

  return Remaining;
}

void VH::arguments(const abi::runtime_test::ArgumentTest &Test) const {
  // List all the things (currently registers and stack bytes) that are still
  // pending verification.
  // NOTE: they are going to be consumed piece by piece during the verification
  //       process.
  auto Registers = ABI.sortArguments(FunctionLayout.argumentRegisters());
  auto Stack = dropInterArgumentPadding(Test.StateBeforeTheCall.Stack);
  LeftToVerify Remaining{ .Registers = Registers, .Stack = Stack };

  // In case of SPTAR, handle the "extra" argument.
  Remaining = adjustForSPTAR(Remaining);

  // Verify each argument separately.
  for (uint64_t Index = 0; Index < Test.Arguments.size(); ++Index) {
    Remaining = verifyAnArgument(Test.StateBeforeTheCall,
                                 Test.Arguments[Index],
                                 Remaining,
                                 Index);
  }

  // Do final checks to make sure no unverified state is still remaining.
  if (!Remaining.Registers.empty()) {
    fail("There are leftover registers: the argument type size is inconsistent "
         "(model value differs from the real one) or the layout is straight up "
         "broken.");
  }

  if (!Remaining.Stack.empty()) {
    fail("There are " + std::to_string(Remaining.Stack.size())
         + " unconsumed stack bytes: the layout shows the need for more "
           "stack bytes than necessary.");
  }
}

uint64_t VH::valueFromBytes(llvm::ArrayRef<std::byte> Input) const {
  uint64_t PointerSize = model::Architecture::getPointerSize(Architecture);
  revng_assert(Input.size() <= PointerSize);

  uint64_t Result = 0;
  for (uint64_t I = 0; I < PointerSize; ++I) {
    if (I != 0)
      Result <<= 8;

    uint64_t Index = IsLittleEndian ? (PointerSize - I - 1) : I;
    if (Index < Input.size())
      Result += static_cast<uint64_t>(Input[Index]);
  }

  return Result;
}

void VH::returnValue(const abi::runtime_test::ReturnValueTest &Test) const {
  uint64_t PointerSize = model::Architecture::getPointerSize(Architecture);

  // Make sure the value is the same both before and after the call.
  verifyValuePreservation(Test.ReturnValue.ExpectedBytes,
                          Test.ReturnValue.FoundBytes);

  // Because we have no way to represent functions that use SPTAR in `rft`
  // format - they are misdetected here (and are tested as simple functions
  // that just happen to accept and return a pointer).
  //
  // To avoid that, manually flag such functions, which is safe because
  // we know for sure that a test function has EITHER arguments OR a return
  // value: never both. Because of which it's safe to assume that if
  // a function happen to have both - it's because of the `cft->rft`
  // conversion.
  bool UsesSPTAR = FunctionLayout.hasSPTAR();
  if (!UsesSPTAR) {
    bool SingleArgument = FunctionLayout.Arguments.size() == 1;
    bool SingleReturnValue = FunctionLayout.ReturnValues.size() == 1;
    if (SingleArgument && SingleReturnValue) {
      const auto &ArgumentType = FunctionLayout.Arguments[0].Type;
      const auto &ReturnValueType = FunctionLayout.ReturnValues[0].Type;
      bool TypesMatch = ArgumentType == ReturnValueType;
      bool ReturnValueIsAPointer = ReturnValueType.isPointer();

      // Case for a register SPTAR - types must match.
      if (TypesMatch && ReturnValueIsAPointer)
        UsesSPTAR = true;

      // Case for a stack SPTAR - allow any type as the argument as long as
      // it's pointer-sized.
      if (*ArgumentType.size() == PointerSize && ReturnValueIsAPointer)
        UsesSPTAR = true;
    }
  }

  if (UsesSPTAR) {
    // The return value location is passed in as a pointer.
    abi::FunctionType::Layout::Argument SPTAR = FunctionLayout.Arguments[0];

    // First, check whether the location (pointer to it) of the return value
    // matches the expected register or stack location.
    //
    // Layout represents such a location as the first argument.
    llvm::ArrayRef<std::byte> ReturnValueLocationBytes;
    uint64_t ReturnValueLocationValue;
    if (!SPTAR.Registers.empty()) {
      // The pointer is in a register.
      if (SPTAR.Registers.size() != 1) {
        fail("Multi-register pointers are not supported. Either a new obscure "
             "architecture was added, or something went *very* wrong.");
      }

      // Check if SPTAR is where we expect to be.
      const auto &Registers = Test.StateBeforeTheCall.Registers;
      llvm::ArrayRef RegisterBytes = Registers.at(SPTAR.Registers[0]).Bytes;
      if (RegisterBytes != llvm::ArrayRef(Test.ReturnValue.AddressBytes)) {
        fail("Verification of the return value location register ('"
             + model::Register::getName(SPTAR.Registers[0]).str()
             + "') failed.");
      }

      // Save the location to be used further up.
      ReturnValueLocationBytes = Test.ReturnValue.AddressBytes;
      ReturnValueLocationValue = Test.ReturnValue.Address;
    } else if (SPTAR.Stack.has_value()) {
      // The pointer is on the stack.
      if (SPTAR.Stack->Size != PointerSize)
        fail("Only pointer-sized return value locations are supported.");

      ReturnValueLocationBytes = llvm::ArrayRef(Test.StateBeforeTheCall.Stack)
                                   .slice(SPTAR.Stack->Offset,
                                          SPTAR.Stack->Size);
      ReturnValueLocationValue = valueFromBytes(ReturnValueLocationBytes);
    } else {
      fail("ABI definition for '" + std::string(ABI.getName())
           + "' does not define a return value location. Does the ABI support "
             "returning big values?");
    }

    // Only check the return value pointers if layout reports it to be there,
    // since there are architectures where its presence is not guaranteed,
    // for example AArch64.
    if (!FunctionLayout.ReturnValues.empty()) {
      if (FunctionLayout.ReturnValues.size() != 1
          && FunctionLayout.ReturnValues[0].Registers.size() != 1) {
        fail("At most one register is allowed as the return value for SPTAR "
             "functions.");
      }

      const auto &RVReg = FunctionLayout.ReturnValues[0].Registers[0];
      llvm::ArrayRef Bytes = Test.StateAfterTheReturn.Registers.at(RVReg).Bytes;
      if (ReturnValueLocationBytes != Bytes) {
        fail("Returned pointer ('" + model::Register::getName(RVReg).str()
             + "') doesn't match SPTAR value.");
      }
    }

    // The only thing that's left is to verify that the return value is
    // actually present at the location the pointer points to.
    auto StackPointer = model::Architecture::getStackPointer(Architecture);
    auto SPValue = Test.StateAfterTheReturn.Registers.at(StackPointer).Value;
    std::ptrdiff_t StackOffset = ReturnValueLocationValue - SPValue;
    revng_assert(StackOffset >= 0);
    llvm::ArrayRef OnStack = Test.StateAfterTheReturn.Stack;
    OnStack = OnStack.slice(StackOffset, Test.ReturnValue.FoundBytes.size());
    if (OnStack != llvm::ArrayRef(Test.ReturnValue.FoundBytes))
      fail("The return value doesn't match the one that was expected.");
  } else {
    // The value is returned normally.

    llvm::ArrayRef ReturnValueBytes = Test.ReturnValue.FoundBytes;

    for (const auto &ReturnValue : FunctionLayout.ReturnValues) {
      for (const auto &Register : ReturnValue.Registers) {
        const auto &RegState = Test.StateAfterTheReturn.Registers.at(Register);
        llvm::ArrayRef Bytes = RegState.Bytes;
        revng_assert(Bytes.size() <= PointerSize);
        Bytes = Bytes.take_front(ReturnValueBytes.size());
        if (!ReturnValueBytes.take_front(Bytes.size()).equals(Bytes)) {
          fail("A piece of the return value found in the '"
               + model::Register::getName(Register).str()
               + "' register doesn't match the expected value.");
        }

        ReturnValueBytes = ReturnValueBytes.drop_front(Bytes.size());
      }
    }

    if (!ReturnValueBytes.empty()) {
      fail("Unable to find some parts of the return value. Should some "
           "additional registers be mentioned in the definition of '"
           + std::string(ABI.getName()) + "' abi?");
    }
  }
}

static abi::FunctionType::Layout
getPrototypeLayout(const model::Function &Function, model::ABI::Values ABI) {
  const model::Type *Prototype = Function.Prototype().getConst();
  revng_assert(Prototype != nullptr);
  if (auto *CABI = llvm::dyn_cast<model::CABIFunctionType>(Prototype)) {
    if (ABI != CABI->ABI()) {
      std::string Error = "ABI mismatch. Passed argument indicates that "
                          "the ABI to use is '"
                          + serializeToString(ABI)
                          + "' while the function contains '"
                          + serializeToString(CABI->ABI()) + "'.";
      revng_abort(Error.c_str());
    }

    return abi::FunctionType::Layout(*CABI);
  } else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(Prototype)) {
    return abi::FunctionType::Layout(*Raw);
  } else {
    revng_abort("Layouts of non-function types are not supported.");
  }
}

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI) {
  model::Architecture::Values Architecture = model::ABI::getArchitecture(ABI);
  auto Parsed = abi::runtime_test::parse(RuntimeArtifact, Architecture);

  llvm::StringRef ArchitectureName = model::Architecture::getName(Architecture);
  revng_check(ArchitectureName == Parsed.Architecture);

  const abi::Definition &Def = abi::Definition::get(ABI);
  VerificationHelper Helper{ Architecture, Def, Parsed.IsLittleEndian };
  size_t ArgumentTestCount = 0, ReturnValueTestCount = 0;
  for (auto &Function : Binary->Functions()) {
    Helper.FunctionName = Function.OriginalName();
    if (Helper.FunctionName.take_front(5) == "test_")
      Helper.FunctionName = Helper.FunctionName.drop_front(5);
    if (auto Test = Parsed.ArgumentTests.find(Helper.FunctionName);
        Test != Parsed.ArgumentTests.end()) {
      Helper.FunctionLayout = getPrototypeLayout(Function, ABI);
      for (const abi::runtime_test::ArgumentTest &Iteration : Test->second)
        Helper.arguments(Iteration);
      ++ArgumentTestCount;
    } else if (auto Test = Parsed.ReturnValueTests.find(Helper.FunctionName);
               Test != Parsed.ReturnValueTests.end()) {
      Helper.FunctionLayout = getPrototypeLayout(Function, ABI);
      for (const abi::runtime_test::ReturnValueTest &Iteration : Test->second)
        Helper.returnValue(Iteration);
      ++ReturnValueTestCount;
    } else {
      // Ignore types from the model, that are not mentioned in
      // the runtime test artifact.
      continue;
    }
  }

  if (ArgumentTestCount != Parsed.ArgumentTests.size()) {
    std::string Error = std::to_string(ArgumentTestCount)
                        + " functions from the binary were tested, but "
                          "artifact contains "
                        + std::to_string(Parsed.ArgumentTests.size())
                        + " argument test functions.\n"
                          "Does the binary match the artifact?";
    revng_abort(Error.c_str());
  }

  if (ReturnValueTestCount != Parsed.ReturnValueTests.size()) {
    std::string Error = std::to_string(ReturnValueTestCount)
                        + " functions from the binary were tested, but "
                          "artifact contains "
                        + std::to_string(Parsed.ReturnValueTests.size())
                        + " return value test functions.\n"
                          "Does the binary match the artifact?";
    revng_abort(Error.c_str());
  }
}
