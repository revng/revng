//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/ADT/Queue.h"
#include "revng/Canonicalize/FixPointerSize.h"
#include "revng/Model/Architecture.h"
#include "revng/PipeboxCommon/ObjectID.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static bool containsPointer(Type *T) {
  OnceQueue<Type *> Queue;
  Queue.insert(T);
  while (not Queue.empty()) {
    Type *Current = Queue.pop();
    if (Current->isPointerTy())
      return true;
    auto ContainedTypesCount = Current->getNumContainedTypes();
    for (unsigned I = 0; I < ContainedTypesCount; ++I)
      Queue.insert(Current->getContainedType(I));
  }

  return false;
}

// Rewrite the default-address-space pointer specification in a DataLayout
// string to the given pointer size in bits.
static std::string rewriteDefaultPointerSize(StringRef Original,
                                             unsigned Bits) {
  std::string Width = std::to_string(Bits);
  std::string NewSpec = "p:" + Width + ":" + Width;

  SmallVector<StringRef, 16> Parts;
  Original.split(Parts, '-');
  std::string Result;
  bool Replaced = false;
  for (StringRef Part : Parts) {
    if (not Result.empty())
      Result += '-';
    if (Part.starts_with("p:") or Part.starts_with("p0:")) {
      Result += NewSpec;
      Replaced = true;
    } else {
      Result += Part.str();
    }
  }
  if (not Replaced) {
    if (not Result.empty())
      Result += '-';
    Result += NewSpec;
  }

  return Result;
}

static void checkAndFixModule(llvm::Module &M, unsigned TargetPointerBits) {
  for (llvm::GlobalVariable &GV : M.globals()) {
    // Ignore unused globals
    if (GV.use_empty())
      continue;

    revng_assert(not containsPointer(GV.getValueType()),
                 ("used global variable's value type transitively contains a "
                  "pointer: "
                  + dumpToString(&GV))
                   .c_str());
  }

  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;

    for (llvm::Instruction &I : llvm::instructions(F)) {
      if (auto *Alloca = dyn_cast<llvm::AllocaInst>(&I)) {
        Type *AllocatedTy = Alloca->getAllocatedType();
        revng_assert(not containsPointer(AllocatedTy),
                     ("scalar alloca containing a pointer in @" + F.getName()
                      + ": " + dumpToString(Alloca))
                       .str()
                       .c_str());
      } else if (auto *GEP = dyn_cast<llvm::GetElementPtrInst>(&I)) {
        revng_assert(not containsPointer(GEP->getSourceElementType()),
                     ("GEP source type transitively contains a pointer in @"
                      + F.getName() + ": " + dumpToString(GEP))
                       .str()
                       .c_str());
      }
    }
  }

  auto &DL = M.getDataLayout();
  if (DL.getPointerSizeInBits(0) == TargetPointerBits)
    return;

  auto Original = DL.getStringRepresentation();
  M.setDataLayout(rewriteDefaultPointerSize(Original, TargetPointerBits));
}

namespace revng::pypeline::pipes {

PipeOutput FixPointerSize::run(const Model &TheModel,
                               const revng::pypeline::Request &Incoming,
                               const revng::pypeline::Request &Outgoing,
                               llvm::StringRef Configuration,
                               LLVMFunctionContainer &Container) {
  using namespace model::ABI;
  auto TargetPointerBits = getPointerSize(TheModel.get()->targetABI()) * 8;

  for (const ObjectID *Object : Outgoing.at(0)) {
    revng_assert(Object->kind() == Kinds::Function);
    checkAndFixModule(Container.getModule(*Object), TargetPointerBits);
  }

  return { {}, {} };
}

} // namespace revng::pypeline::pipes
