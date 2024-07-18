/// \file Lift.cpp
/// Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Lift/IRAnnotators.h"
#include "revng/Lift/Lift.h"
#include "revng/Lift/LiftPipe.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;
using namespace pipeline;
using namespace ::revng::pipes;

class VerifyPass : public llvm::ModulePass {
public:
  static char ID;

public:
  VerifyPass() : llvm::ModulePass(ID) {}

public:
  bool runOnModule(Module &M) final {
    Function *RootFunction = M.getFunction("root");
    revng_assert(RootFunction != nullptr);

    llvm::ModuleSlotTracker MST(&M, false);
    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);

    for (BasicBlock &BB : *RootFunction) {
      // Ignore some special basic blocks
      if (BB.getName() == "dispatcher.default"
          or BB.getName() == "serialize_and_jump_out"
          or BB.getName() == "return_from_external" or BB.getName() == "setjmp"
          or BB.getName() == "dispatcher.external")
        continue;

      for (Instruction &I : BB) {
        bool Good = false;

        switch (I.getOpcode()) {
        case Instruction::Store:
        case Instruction::Load:
          Good = true;
          break;

        case Instruction::IntToPtr:
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::UDiv:
        case Instruction::SDiv:
        case Instruction::URem:
        case Instruction::SRem:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
        case Instruction::ZExt:
        case Instruction::Trunc:
        case Instruction::SExt:
        case Instruction::ICmp:
        case Instruction::LShr:
        case Instruction::AShr:
        case Instruction::Shl:
        case Instruction::Select:
          Good = true;
          break;

        case Instruction::Br:
        case Instruction::Switch:
        case Instruction::Unreachable:
          Good = true;
          break;

        case Instruction::PHI:
          Good = true;
          break;

        case Instruction::ExtractValue:
          Good = true;
          break;

        case Instruction::Call:
          // Make further checks
          auto *Call = cast<CallInst>(&I);
          Value *CalledOperand = Call->getCalledOperand();
          Function *Callee = dyn_cast_or_null<Function>(CalledOperand);
          StringRef CalleeName;
          if (Callee != nullptr)
            CalleeName = Callee->getName();

          Good = (CalleeName == "newpc" or CalleeName == "jump_to_symbol"
                  or CalleeName.startswith("helper_")
                  or CalleeName == "function_call"
                  or CalleeName == "helper_initialize_env"
                  or CalleeName == "abort");

          switch (Callee->getIntrinsicID()) {
          case Intrinsic::fshl:
          case Intrinsic::fshr:
          case Intrinsic::bswap:
          case Intrinsic::abs:
          case Intrinsic::umin:
          case Intrinsic::umax:
          case Intrinsic::smin:
          case Intrinsic::smax:
          case Intrinsic::ctlz:
            Good = true;
          }

          break;
        }

        if (not Good) {
          Stream << "Unexpected instruction: ";
          I.print(Stream, MST);
          Stream << "\n";
        }
      }
    }

    dbg << Buffer;

    return false;
  }
};

char VerifyPass::ID;

void Lift::run(ExecutionContext &EC,
               const BinaryFileContainer &SourceBinary,
               LLVMContainer &Output) {
  if (not SourceBinary.exists())
    return;

  const TupleTree<model::Binary> &Model = getModelFromContext(EC);

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(*SourceBinary.path());
  auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));
  RawBinaryView RawBinary(*Model, Buffer->getBuffer());

  // Perform lifting
  llvm::legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadExecutionContextPass(&EC, Output.name()));
  PM.add(new LoadBinaryWrapperPass(Buffer->getBuffer()));
  PM.add(new LiftPass);
  PM.add(new VerifyPass);
  PM.run(Output.getModule());

  EC.commitUniqueTarget(Output);
}

std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
Lift::invalidate(const BinaryFileContainer &SourceBinary,
                 const pipeline::LLVMContainer &ModuleContainer,
                 const GlobalTupleTreeDiff &Diff) const {
  // Prepare result in case of invalidation
  std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
    InvalidateResult;
  InvalidateResult[&ModuleContainer].push_back(pipeline::Target({},
                                                                kinds::Root));

  Function *Root = ModuleContainer.getModule().getFunction("root");
  const Function *NewPC = getIRHelper("newpc", ModuleContainer.getModule());

  if (Root == nullptr or NewPC == nullptr)
    return InvalidateResult;

  // Collect all jump targets by inspecting calls to newpc and record whether it
  // was found after adding entry addresses of functions
  std::map<MetaAddress, bool> JumpTargets;
  for (const CallBase *Call : callers(NewPC)) {
    bool IsJumpTarget = getLimitedValue(Call->getArgOperand(2)) == 1;

    if (IsJumpTarget) {
      auto Address = MetaAddress::fromValue(Call->getArgOperand(0));

      // Detect if this jump targets has been discovered *after* recording the
      // entry addresses of functions

      // Be conservative and assume it is, in absence of information
      bool DependsOnModelFunction = true;
      const Instruction *Terminator = Call->getParent()->getTerminator();
      if (Terminator->hasMetadata(JTReasonMDName)) {
        uint32_t Reasons = GeneratedCodeBasicInfo::getJTReasons(Terminator);
        DependsOnModelFunction = hasReason(Reasons,
                                           JTReason::DependsOnModelFunction);
      }

      JumpTargets.emplace(Address, DependsOnModelFunction);
    }
  }

  // Inspect the diff looking for newly added model::Functions
  auto *ModelDiff = Diff.getAs<model::Binary>();
  revng_assert(ModelDiff != nullptr);

  using Fields = TupleLikeTraits<model::Binary>::Fields;
  size_t FunctionsIndex = static_cast<size_t>(Fields::Functions);
  for (const auto &Change : ModelDiff->Changes) {
    bool IsAddition = not Change.Old.has_value() and Change.New.has_value();
    bool IsRemoval = Change.Old.has_value() and not Change.New.has_value();

    // Look for additions to /Functions
    auto &Path = Change.Path;
    if (Path.size() == 1) {
      if (Path[0].get<size_t>() == FunctionsIndex) {
        // Check the Entry address of the newly added model::Function
        MetaAddress ChangedAddress;
        if (IsAddition)
          ChangedAddress = std::get<model::Function>(*Change.New).Entry();
        else
          ChangedAddress = std::get<model::Function>(*Change.Old).Entry();

        auto It = JumpTargets.find(ChangedAddress);
        bool IsJumpTarget = It != JumpTargets.end();
        bool DependsOnModelFunction = IsJumpTarget and It->second;

        if (IsAddition and not IsJumpTarget) {
          // We're adding a function that was not a jump target
          return InvalidateResult;
        } else if (IsRemoval and DependsOnModelFunction) {
          // We're removing a function whose address was not discovered *before*
          // starting to take into account the entry addresses of model
          // functions
          return InvalidateResult;
        }
      }
    }
  }

  return {};
}

llvm::Error Lift::checkPrecondition(const pipeline::Context &Context) const {
  const auto &Model = *getModelFromContext(Context);

  if (Model.Architecture() == model::Architecture::Invalid) {
    return revng::createError("Cannot lift binary with architecture invalid.");
  }

  if (Model.DefaultABI() == model::ABI::Invalid
      and Model.DefaultPrototype().isEmpty()) {
    return revng::createError("Cannot lift binary without either a DefaultABI "
                              "or a DefaultPrototype.");
  }

  return llvm::Error::success();
}

static_assert(pipeline::HasInvalidate<Lift>);

static RegisterPipe<Lift> E1;
