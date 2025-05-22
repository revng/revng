//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/DebugInfoHelpers.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/DecompilationHelpers.h"

RegisterIRHelper RenamableVariableHelper("renamable-variable",
                                         "absent in emitted c");

struct EmbedVariableNames {
public:
  static constexpr auto Name = "embed-variable-names";

private:
  llvm::FunctionCallee makeIRHelper(llvm::Module &M, uint64_t PointerSize) {
    // @renamable-variable(ptr @type, ptr @emitted-location)

    llvm::FunctionType *FT;
    if (PointerSize == 8) {
      using StringLiteral = const uint8_t *;
      FT = createFunctionType<uint64_t,
                              StringLiteral,
                              StringLiteral>(M.getContext());

    } else if (PointerSize == 4) {
      using StringLiteral = const uint8_t *;
      FT = createFunctionType<uint32_t,
                              StringLiteral,
                              StringLiteral>(M.getContext());

    } else {
      std::string Error = "Unsupported pointer size: "
                          + std::to_string(PointerSize);
      revng_abort(Error.c_str());
    }

    auto Result = getOrInsertIRHelper("renamable-variable", M, FT);
    auto &Callee = *llvm::cast<llvm::Function>(Result.getCallee());
    Callee.addFnAttr(llvm::Attribute::NoUnwind);
    Callee.addFnAttr(llvm::Attribute::WillReturn);
    Callee.addFnAttr(llvm::Attribute::NoMerge);
    Callee.setDoesNotAccessMemory();
    FunctionTags::RenamableVariable.addTo(&Callee);

    return Result;
  }

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::StackAccessesSegregated,
                                     0,
                                     revng::kinds::StackAccessesSegregated) };
  }

  // Unless this is set, if even a single variable user is missing debug
  // information, the entire variable is not considered for renaming.
  static constexpr bool ImpreciseMode = false;

  void run(pipeline::ExecutionContext &Context,
           pipeline::LLVMContainer &ModuleContainer) {
    llvm::Module &M = ModuleContainer.getModule();
    llvm::IRBuilder<> B(M.getContext());

    const model::Binary &Binary = *revng::getModelFromContext(Context);
    // model::CNameBuilder NameBuilder = Binary;

    auto PtrSize = model::Architecture::getPointerSize(Binary.Architecture());
    llvm::FunctionCallee IRHelper = makeIRHelper(M, PtrSize);

    using TaggedFunctionKind = revng::kinds::TaggedFunctionKind;
    llvm::StringRef ContainerName = ModuleContainer.name();

    for (auto &&[ModelFunction, LLVMFunction] :
         TaggedFunctionKind::getFunctionsAndCommit(Context, M, ContainerName)) {

      struct VariableMetadata {
        llvm::CallInst *Instruction;
        SortedVector<MetaAddress> UserAddressList = {};

        VariableMetadata(llvm::CallInst *Instruction) :
          Instruction(Instruction) {}
      };
      std::vector<VariableMetadata> Variables;

      // First, gather all the variable user addresses
      for (llvm::BasicBlock &BasicBlock : *LLVMFunction) {
        for (llvm::Instruction &Instruction : BasicBlock) {
          if (auto *LV = getCallToTagged(&Instruction,
                                         FunctionTags::LocalVariable)) {
            auto &[_, UserAddressList] = Variables.emplace_back(LV);
            for (const llvm::Value *UserV : LV->users()) {
              if (const auto *User = llvm::dyn_cast<llvm::Instruction>(UserV)) {
                if (std::optional MaybeMA = revng::tryExtractAddress(*User)) {
                  UserAddressList.emplace(std::move(*MaybeMA));

                } else if constexpr (ImpreciseMode) {
                  // Danger! Skipping a potential user!

                } else {
                  // Found a user without debug information,
                  // discard current variable.
                  UserAddressList.clear();
                  break;
                }
              }
            }
          }
        }
      }

      std::unordered_set<std::string> UsedLocations;
      for (const auto &[VariableInstruction, UserAddressList] : Variables) {
        revng_assert(VariableInstruction->getNumOperands() == 2,
                     "Did `@LocalVariable` signature change?");

        if (not UserAddressList.empty()) {
          std::string LocationString = toString(UserAddressList);
          auto [_, Inserted] = UsedLocations.emplace(LocationString);
          if (not Inserted) {
            // There is already a variable attached to the given location.
            // TODO: come up with a better solution than just disallowing
            //       subsequent variable renaming.
            continue;
          }

          B.SetInsertPoint(VariableInstruction->getNextNode());

          std::array<llvm::Value *, 2> Arguments = {
            VariableInstruction->getOperand(0),
            getUniqueString(&M, std::move(LocationString))
          };

          auto *RenamableVariableCall = B.CreateCall(IRHelper, Arguments);
          RenamableVariableCall->copyMetadata(*VariableInstruction);

          VariableInstruction->replaceAllUsesWith(RenamableVariableCall);
          VariableInstruction->eraseFromParent();
        }
      }
    }
  }
};

static pipeline::RegisterPipe<EmbedVariableNames> EVNPipe;
