/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"

#include "ABIArtifactParser.h"

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI);

struct VerificationHelper {
public:
  void verify(const abi::FunctionType::Layout &FunctionLayout,
              const abi::artifact::FunctionArtifact &Artifact) const;

public:
  const model::Architecture::Values Architecture;
  const abi::Definition &ABI;
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
  const std::size_t RegisterSize;
};

using VH = VerificationHelper;
void VH::verify(const abi::FunctionType::Layout &FunctionLayout,
                const abi::artifact::FunctionArtifact &Artifact) const {
  for (std::size_t Index = 0; Index < Artifact.Iterations.size(); ++Index) {
    IterationAccessHelper Helper(Artifact.Iterations[Index], Architecture);

    // Sort the argument registers
    auto ArgumentRegisters = FunctionLayout.argumentRegisters();
    auto OrderedRegisterList = ABI.sortArguments(std::move(ArgumentRegisters));

    // Compute the relevant piece of the stack leaving padding behind.
    abi::artifact::Stack ExtractedStackBytes;
    std::size_t PreviousArgumentEndsAt = 0;
    std::size_t CombinedArgumentSize = 0;
    for (const auto &Argument : FunctionLayout.Arguments) {
      if (Argument.Stack.has_value()) {
        revng_assert(Argument.Stack->Size != 0);
        if (Argument.Stack->Offset < PreviousArgumentEndsAt) {
          std::string Error = "Layout of '" + FunctionName.str()
                              + "' is not valid: stack arguments must not "
                                "overlap.";
          revng_abort(Error.c_str());
        }

        auto PaddingSize = Argument.Stack->Offset - PreviousArgumentEndsAt;
        if (PaddingSize > ABI.getPointerSize()) {
          // TODO: this check can be improved quite a bit by taking
          // `ABI::Types()` into the account.
          std::string Error = "Layout of '" + FunctionName.str()
                              + "' is not valid: padding exceeds the register "
                                "size.\n";
          Error += "The current argument is expected at offset "
                   + std::to_string(Argument.Stack->Offset)
                   + " while the previous one ends at "
                   + std::to_string(PreviousArgumentEndsAt) + "\n";
          revng_abort(Error.c_str());
        }

        std::size_t CurrentSize = ExtractedStackBytes.size();
        ExtractedStackBytes.resize(CurrentSize + Argument.Stack->Size);

        llvm::ArrayRef<std::byte> Bytes = Helper.Iteration.Stack;
        Bytes = Bytes.slice(Argument.Stack->Offset, Argument.Stack->Size);
        llvm::copy(Bytes, std::next(ExtractedStackBytes.begin(), CurrentSize));
        PreviousArgumentEndsAt = Argument.Stack->Offset + Argument.Stack->Size;
        CombinedArgumentSize += (Argument.Stack->Size);
      }
    }
    llvm::ArrayRef<std::byte> StackBytes = ExtractedStackBytes;
    if (StackBytes.size() != CombinedArgumentSize) {
      std::string Error = "Verification of '" + FunctionName.str()
                          + "' failed: the piece of stack provided by the "
                            "artifact has a different size from what the "
                            "layout expects.";
      revng_abort(Error.c_str());
    }

    std::size_t CurrentRegisterIndex = 0;
    if (FunctionLayout.returnsAggregateType()) {
      // Account for the shadow pointer to the return value.
      revng_assert(not FunctionLayout.Arguments.empty());
      auto &ShadowArgument = FunctionLayout.Arguments[0];
      using namespace abi::FunctionType::ArgumentKind;
      revng_assert(ShadowArgument.Kind == ShadowPointerToAggregateReturnValue);
      if (ShadowArgument.Registers.size() == 1) {
        // It's in a register, drop one if needed.
        model::Register::Values Register = *ShadowArgument.Registers.begin();
        revng_assert(Register == ABI.ReturnValueLocationRegister());
        if (!OrderedRegisterList.empty())
          if (*OrderedRegisterList.begin() == ABI.ReturnValueLocationRegister())
            ++CurrentRegisterIndex;

      } else if (ShadowArgument.Stack.has_value()) {
        // It's on the stack, drop enough bytes for a pointer from the front.
        revng_assert(ShadowArgument.Stack->Offset == 0);
        revng_assert(ShadowArgument.Stack->Size == ABI.getPointerSize());
        StackBytes = StackBytes.drop_front(ABI.getPointerSize());
      } else {
        std::string Error = "Verification of the return value of '"
                            + FunctionName.str()
                            + "' failed: layout is not valid, did it verify?";
        revng_abort(Error.c_str());
      }
    }

    // Check the arguments.
    std::uint64_t ArgumentIndex = 0;
    for (const abi::artifact::Argument &Argument : Helper.Iteration.Arguments) {
      llvm::ArrayRef<std::byte> ArgumentBytes = Argument.Bytes;
      if (ArgumentBytes.empty()) {
        std::string Error = "The `Bytes` field of the artifact is empty for "
                            "the argument #"
                            + std::to_string(ArgumentIndex) + " of '"
                            + FunctionName.str() + "' function.";
        revng_abort(Error.c_str());
      }

      do {
        if (CurrentRegisterIndex < OrderedRegisterList.size()) {
          auto Current = OrderedRegisterList[CurrentRegisterIndex];
          auto RegValue = Helper.registerValue(Current);
          if (!RegValue.has_value()) {
            std::string Error = "Verification of '"
                                + model::Register::getName(Current).str()
                                + "' register failed: it's not specified by "
                                  "the artifact for '"
                                + FunctionName.str() + "'.";
            revng_abort(Error.c_str());
          }

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
          revng_abort("An argument cannot be found, it uses neither the "
                      "expected stack part nor the expected registers.");

        // Stack matches.
        auto Min = model::Architecture::getPointerSize(Architecture);
        StackBytes = StackBytes.drop_front(std::max(ArgumentBytes.size(), Min));
        ArgumentBytes = {};
      } while (!ArgumentBytes.empty());

      ++ArgumentIndex;
    }

    if (CurrentRegisterIndex != OrderedRegisterList.size()) {
      std::string Error = "'" + FunctionName.str()
                          + "' signature indicates the need for more registers "
                            "to pass an argument than the maximum allowed "
                            "count.";
      revng_abort(Error.c_str());
    }

    if (!StackBytes.empty()) {
      std::string Error = "The combined stack argument size of '"
                          + FunctionName.str()
                          + "' is different from what was "
                            "expected.";
      revng_abort(Error.c_str());
    }

    // Check the return value.
    auto RVR = FunctionLayout.returnValueRegisters();
    auto ReturnValueRegisterList = ABI.sortReturnValues(std::move(RVR));
    if (!Helper.Iteration.ReturnValue.Bytes.empty()) {
      if (ReturnValueRegisterList.size() == 0) {
        std::string Error = "Verification of the return value of '"
                            + FunctionName.str()
                            + "' failed: found a return value that should not "
                              "be there";
        revng_abort(Error.c_str());
      }

      std::size_t UsedRegisterCounter = 0;
      llvm::ArrayRef ReturnValueBytes = Helper.Iteration.ReturnValue.Bytes;
      for (auto &CurrentRegister : ReturnValueRegisterList) {
        auto RegValue = Helper.registerValue(CurrentRegister);
        if (!RegValue.has_value()) {
          std::string Error = "Verification of '"
                              + model::Register::getName(CurrentRegister).str()
                              + "' register failed: it's not specified by the "
                                "artifact.";
          revng_abort(Error.c_str());
        }

        if (ReturnValueBytes.take_front(RegValue->size()).equals(*RegValue)) {
          ReturnValueBytes = ReturnValueBytes.drop_front(RegValue->size());
          ++UsedRegisterCounter;
        } else if (auto Piece = RegValue->take_front(ReturnValueBytes.size());
                   Piece.equals(ReturnValueBytes)) {
          ReturnValueBytes = {};
          ++UsedRegisterCounter;
        } else {
          std::string Error = "Verification of the return value of '"
                              + FunctionName.str()
                              + "' failed: unable to locate it.";
          revng_abort(Error.c_str());
        }
      }

      if (UsedRegisterCounter != ReturnValueRegisterList.size()) {
        std::string Error = "'" + FunctionName.str()
                            + "' signature indicates the need for more "
                              "registers "
                              "to return a value than the maximum allowed "
                              "count.";
        revng_abort(Error.c_str());
      }
    } else if (ReturnValueRegisterList.size() != 0) {
      std::string Error = "'" + FunctionName.str()
                          + "' signature does not mention a return value even "
                            "though it's expected.";
      revng_abort(Error.c_str());
    }
  }
}

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI) {
  model::Architecture::Values Architecture = model::ABI::getArchitecture(ABI);
  auto ParsedArtifact = abi::artifact::parse(RuntimeArtifact, Architecture);

  llvm::StringRef ArchitectureName = model::Architecture::getName(Architecture);
  revng_assert(ArchitectureName == ParsedArtifact.Architecture);

  const abi::Definition &Def = abi::Definition::get(ABI);
  VerificationHelper Helper{ Architecture, Def, ParsedArtifact.IsLittleEndian };

  std::size_t TestedCount = 0;
  for (auto &Function : Binary->Functions()) {
    const std::string &Name = Function.OriginalName();
    auto CurrentFunction = ParsedArtifact.Functions.find(Name);
    if (CurrentFunction == ParsedArtifact.Functions.end()) {
      // Ignore types not present in the artifact.
      continue;
    }

    ++TestedCount;
    Helper.FunctionName = Name;

    using Layout = abi::FunctionType::Layout;
    const model::Type *Prototype = Function.Prototype().getConst();
    revng_assert(Prototype != nullptr);
    if (auto *CABI = llvm::dyn_cast<model::CABIFunctionType>(Prototype)) {
      // Copy the prototype since we might have to modify it before testing.
      model::CABIFunctionType PrototypeCopy = *CABI;

      if (ABI != PrototypeCopy.ABI()) {
        // Workaround for dwarf sometimes misdetecting a very specific ABI,
        // for example, it labels `SystemV_x86_regparam_N` as just `SystemV_x86`
        // despite them being incompatible.
        std::array AllowedABIs = { model::ABI::SystemV_x86_regparm_3,
                                   model::ABI::SystemV_x86_regparm_2,
                                   model::ABI::SystemV_x86_regparm_1 };
        revng_assert(llvm::is_contained(AllowedABIs, ABI));
        PrototypeCopy.ABI() = ABI;
      }

      Helper.verify(Layout(PrototypeCopy), *CurrentFunction);
    } else if (auto *Raw = llvm::dyn_cast<model::RawFunctionType>(Prototype)) {
      Helper.verify(Layout(*Raw), *CurrentFunction);
    } else {
      // Ignore non-function types.
    }
  }

  revng_check(TestedCount == ParsedArtifact.Functions.size(),
              "Not every function from the artifact was found in the binary. "
              "Does the binary match the artifact?");
}
