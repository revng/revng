//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

/// \brief Pass that wraps Instructions in LLVM IR that must be serialized in
/// special marker calls.

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/MarkForSerialization/MarkAnalysis.h"
#include "revng-c/MarkForSerialization/MarkForSerializationFlags.h"
#include "revng-c/Support/Mangling.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

struct AddIRSerializationMarkersPass : public llvm::FunctionPass {
public:
  static char ID;

  AddIRSerializationMarkersPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};

static std::string makeTypeName(const llvm::Type *Ty) {
  std::string Name;
  if (auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty)) {
    Name = "ptr_to_" + makeTypeName(PtrTy->getElementType());
  } else if (auto *IntTy = llvm::dyn_cast<llvm::IntegerType>(Ty)) {
    Name = "i" + std::to_string(IntTy->getBitWidth());
  } else if (auto *StrucTy = llvm::dyn_cast<llvm::StructType>(Ty)) {
    Name = "struct";
    if (StrucTy->isLiteral()) {
      for (const auto *FieldTy : StrucTy->elements())
        Name += "_" + makeTypeName(FieldTy);
    } else {
      Name += "_" + makeCIdentifier(Ty->getStructName().str());
    }
  } else if (auto *FunTy = llvm::dyn_cast<llvm::FunctionType>(Ty)) {
    Name = "func_" + makeTypeName(FunTy->getReturnType());
    if (not FunTy->params().empty()) {
      Name += "_args";
      for (const auto &ArgT : FunTy->params())
        Name += "_" + makeTypeName(ArgT);
    }
  } else if (Ty->isVoidTy()) {
    Name += "void";
  } else {
    revng_unreachable("cannot build Type name");
  }
  return Name;
}

static std::string makeMarkerName(const llvm::Type *Ty) {
  return "revng_serialization_marker_" + makeTypeName(Ty);
}

bool AddIRSerializationMarkersPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // Mark instructions for serialization, and write the results in ToSerialize
  SerializationMap ToSerialize = {};
  MarkAnalysis::Analysis Mark(F, ToSerialize);
  Mark.initialize();
  Mark.run();

  llvm::Module *M = F.getParent();
  llvm::IRBuilder<> Builder(M->getContext());
  bool Changed = false;
  for (const auto &[I, Flag] : ToSerialize) {
    auto *IType = I->getType();

    // Never serialize the struct initializers.
    if (auto *Call = dyn_cast<llvm::CallInst>(I))
      if (auto *Callee = Call->getFunction())
        if (Callee->hasName()
            and Callee->getName().startswith("struct_initializer"))
          continue;

    // We cannot wrap void-typed things into wrappers.
    // We'll have to handle them in another way in the decompilation pipeline
    if (IType->isVoidTy())
      continue;

    if (bool(Flag)) {

      // Get the SCEV barrier function associated with IType
      auto MarkerName = makeMarkerName(IType);
      auto MarkerCallee = M->getOrInsertFunction(MarkerName, IType, IType);

      if (auto *MarkerF = dyn_cast<llvm::Function>(MarkerCallee.getCallee())) {
        MarkerF->addFnAttr(llvm::Attribute::NoUnwind);
        MarkerF->addFnAttr(llvm::Attribute::WillReturn);
        MarkerF->addFnAttr(llvm::Attribute::InaccessibleMemOnly);
      }

      // Insert a call to the SCEV barrier right after I. For now the call to
      // barrier has an undef argument, that will be fixed later.
      Builder.SetInsertPoint(I->getParent(), std::next(I->getIterator()));
      auto *Undef = llvm::UndefValue::get(IType);
      auto *Call = Builder.CreateCall(MarkerCallee, Undef);

      // Replace all uses of I with the new call.
      I->replaceAllUsesWith(Call);

      // Now Fix the call to use I as argument.
      Call->setArgOperand(0, I);
      Changed = true;
    }
  }
  return true;
}

char AddIRSerializationMarkersPass::ID = 0;

using Register = llvm::RegisterPass<AddIRSerializationMarkersPass>;
static Register X("add-ir-serialization-markers",
                  "Pass that adds serialization markers to the IR",
                  false,
                  false);
