/// \file IRHelpers.cpp
/// \brief Implementation of IR helper functions

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/IRHelpers.h"

// TODO: including GeneratedCodeBasicInfo.h is not very nice

using namespace llvm;

void dumpModule(const Module *M, const char *Path) {
  std::ofstream FileStream(Path);
  raw_os_ostream Stream(FileStream);
  M->print(Stream, nullptr, true);
}

GlobalVariable *buildString(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  auto *Initializer = ConstantDataArray::getString(C, String, true);
  return new GlobalVariable(*M,
                            Initializer->getType(),
                            true,
                            GlobalVariable::InternalLinkage,
                            Initializer,
                            Name);
}

Constant *buildStringPtr(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  Type *Int8PtrTy = Type::getInt8Ty(C)->getPointerTo();
  GlobalVariable *NewVariable = buildString(M, String, Name);
  return ConstantExpr::getBitCast(NewVariable, Int8PtrTy);
}

Constant *getUniqueString(Module *M,
                          StringRef Namespace,
                          StringRef String,
                          const Twine &Name) {
  LLVMContext &C = M->getContext();
  Type *Int8PtrTy = Type::getInt8Ty(C)->getPointerTo();
  NamedMDNode *StringsList = M->getOrInsertNamedMetadata(Namespace);

  for (MDNode *Operand : StringsList->operands()) {
    auto *T = cast<MDTuple>(Operand);
    revng_assert(T->getNumOperands() == 1);
    auto *CAM = cast<ConstantAsMetadata>(T->getOperand(0).get());
    auto *GV = cast<GlobalVariable>(CAM->getValue());
    revng_assert(GV->isConstant() and GV->hasInitializer());

    const Constant *Initializer = GV->getInitializer();
    StringRef Content = cast<ConstantDataArray>(Initializer)->getAsString();

    // Ignore the terminator
    if (Content.drop_back() == String)
      return ConstantExpr::getBitCast(GV, Int8PtrTy);
  }

  GlobalVariable *NewVariable = buildString(M, String, Name);
  auto *CAM = ConstantAsMetadata::get(NewVariable);
  StringsList->addOperand(MDTuple::get(C, { CAM }));
  return ConstantExpr::getBitCast(NewVariable, Int8PtrTy);
}

std::pair<MetaAddress, uint64_t> getPC(Instruction *TheInstruction) {
  BasicBlock *Dispatcher = nullptr;
  CallInst *NewPCCall = nullptr;
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock::reverse_iterator> WorkList;
  if (TheInstruction->getIterator() == TheInstruction->getParent()->begin())
    WorkList.push(--TheInstruction->getParent()->rend());
  else
    WorkList.push(++TheInstruction->getReverseIterator());

  while (!WorkList.empty()) {
    auto I = WorkList.front();
    WorkList.pop();
    auto *BB = I->getParent();
    auto End = BB->rend();

    // Go through the instructions looking for calls to newpc
    for (; I != End; I++) {
      if (auto Marker = dyn_cast<CallInst>(&*I)) {
        // TODO: comparing strings is not very elegant
        auto *Callee = Marker->getCalledFunction();
        if (Callee != nullptr && Callee->getName() == "newpc") {

          // We found two distinct newpc leading to the requested instruction
          if (NewPCCall != nullptr)
            return { MetaAddress::invalid(), 0 };

          NewPCCall = Marker;
          break;
        }
      }
    }

    // If we haven't find a newpc call yet, continue exploration backward
    if (NewPCCall == nullptr) {
      // If one of the predecessors is the dispatcher, don't explore any further
      for (BasicBlock *Predecessor : predecessors(BB)) {

        // Lazily detect dispatcher
        using GCBI = GeneratedCodeBasicInfo;
        if (Dispatcher == nullptr
            and GCBI::getType(Predecessor) == BlockType::DispatcherBlock) {
          Dispatcher = Predecessor;
        }

        // Assert we didn't reach the almighty dispatcher
        revng_assert(!(NewPCCall == nullptr && Predecessor == Dispatcher));
        if (Predecessor == Dispatcher)
          continue;
      }

      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Ignore already visited or empty BBs
        if (!Predecessor->empty()
            && Visited.find(Predecessor) == Visited.end()) {
          WorkList.push(Predecessor->rbegin());
          Visited.insert(Predecessor);
        }
      }
    }
  }

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { MetaAddress::invalid(), 0 };

  auto PC = MetaAddress::fromConstant(NewPCCall->getArgOperand(0));
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(1));
  revng_assert(Size != 0);
  return { PC, Size };
}
