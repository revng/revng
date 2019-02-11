/// \file externaljumpsHandler.cpp
/// \brief Inject code to support jumping in non-translated code and handling
///        the comeback.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <string>

// LLVM includes
#include "llvm/ADT/Triple.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// Local libraries includes
#include "revng/Support/Debug.h"

// Local includes
#include "BinaryFile.h"
#include "ExternalJumpsHandler.h"
#include "JumpTargetManager.h"
#include "revng/Support/Debug.h"

using namespace llvm;
using std::string;

static string &
replace(string &Target, const StringRef Search, const StringRef Replace) {
  size_t Position = Target.find(Search.data());
  revng_assert(Position != string::npos);
  Target.replace(Position, Search.size(), Replace);
  return Target;
}

BasicBlock *ExternalJumpsHandler::createReturnFromExternal() {
  // Create return_from_call BasicBlock
  auto *ReturnFromExternal = BasicBlock::Create(Context,
                                                "return_from_external",
                                                &TheFunction);
  IRBuilder<> Builder(ReturnFromExternal);

  // Identify the global variables to be serialized
  Constant *SavedRegistersPtr = TheModule.getGlobalVariable("saved_registers");
  LoadInst *SavedRegisters = Builder.CreateLoad(SavedRegistersPtr);

  {
    // Deserialize the PC
    Value *GEP = Builder.CreateGEP(SavedRegisters,
                                   Builder.getInt32(Arch.pcMContextIndex()));
    LoadInst *RegisterValue = Builder.CreateLoad(GEP);
    Builder.CreateStore(RegisterValue, JumpTargets.pcReg());
  }

  // Deserialize the ABI registers
  for (const ABIRegister &Register : Arch.abiRegisters()) {
    GlobalVariable *CSV = TheModule.getGlobalVariable(Register.qemuName());
    // Not all the registers have a corresponding CSV
    if (CSV != nullptr) {

      if (Register.inMContext()) {

        Constant *RegisterIndex = Builder.getInt32(Register.mcontextIndex());
        Value *GEP = Builder.CreateGEP(SavedRegisters, RegisterIndex);
        LoadInst *RegisterValue = Builder.CreateLoad(GEP);
        Builder.CreateStore(RegisterValue, CSV);

      } else {

        std::string AsmString = Arch.readRegisterAsm();
        replace(AsmString, "REGISTER", Register.name());
        std::stringstream ConstraintStringStream;
        ConstraintStringStream << "*m,~{},~{dirflag},~{fpsr},~{flags}";
        InlineAsm *Asm = InlineAsm::get(AsmFunctionType,
                                        AsmString,
                                        ConstraintStringStream.str(),
                                        true,
                                        InlineAsm::AsmDialect::AD_ATT);
        Builder.CreateCall(Asm, CSV);
      }
    }
  }

  TerminatorInst *T = Builder.CreateBr(JumpTargets.dispatcher());
  T->setMetadata("revng.block.type",
                 QMD.tuple((uint32_t) ExternalJumpsHandlerBlock));

  return ReturnFromExternal;
}

ExternalJumpsHandler::ExternalJumpsHandler(BinaryFile &TheBinary,
                                           JumpTargetManager &JumpTargets,
                                           Function &TheFunction) :
  Context(getContext(&TheFunction)),
  QMD(Context),
  TheModule(*TheFunction.getParent()),
  TheFunction(TheFunction),
  TheBinary(TheBinary),
  Arch(TheBinary.architecture()),
  JumpTargets(JumpTargets),
  RegisterType(JumpTargets.pcReg()->getType()->getPointerElementType()) {

  AsmFunctionType = FunctionType::get(Type::getVoidTy(Context),
                                      { RegisterType->getPointerTo() },
                                      false);
}

BasicBlock *ExternalJumpsHandler::createSerializeAndJumpOut() {
  // Create the serialize and branch Basic Block
  BasicBlock *Result = BasicBlock::Create(Context,
                                          "serialize_and_jump_out",
                                          &TheFunction);
  IRBuilder<> Builder(Result);

  // Serialize ABI CSVs
  for (const ABIRegister &Register : Arch.abiRegisters()) {
    GlobalVariable *CSV = TheModule.getGlobalVariable(Register.qemuName());

    // Not all the registers have a corresponding CSV
    if (CSV == nullptr)
      continue;

    string AsmString = Arch.writeRegisterAsm();
    replace(AsmString, "REGISTER", Register.name());
    std::stringstream ConstraintStringStream;
    ConstraintStringStream << "*m,~{" << Register.name().data()
                           << "},~{dirflag},~{fpsr},~{flags}";
    InlineAsm *Asm = InlineAsm::get(AsmFunctionType,
                                    AsmString,
                                    ConstraintStringStream.str(),
                                    true,
                                    InlineAsm::AsmDialect::AD_ATT);
    Builder.CreateCall(Asm, CSV);
  }

  // Branch to the Program Counter address
  InlineAsm *Asm = InlineAsm::get(AsmFunctionType,
                                  Arch.jumpAsm(),
                                  "*m,~{dirflag},~{fpsr},~{flags}",
                                  true,
                                  InlineAsm::AsmDialect::AD_ATT);
  Value *PCReg = JumpTargets.pcReg();
  Builder.CreateCall(Asm, PCReg);

  TerminatorInst *T = Builder.CreateUnreachable();
  T->setMetadata("revng.block.type",
                 QMD.tuple((uint32_t) ExternalJumpsHandlerBlock));

  return Result;
}

llvm::BasicBlock *ExternalJumpsHandler::createSetjmp(BasicBlock *FirstReturn,
                                                     BasicBlock *SecondReturn) {
  using CE = ConstantExpr;
  using CI = ConstantInt;

  BasicBlock *SetjmpBB = BasicBlock::Create(Context, "setjmp", &TheFunction);
  IRBuilder<> Builder(SetjmpBB);

  // Call setjmp
  llvm::Constant *SetjmpFunction = TheModule.getFunction("setjmp");
  auto *SetJmpTy = SetjmpFunction->getType()->getPointerElementType();
  auto *JmpBuf = CE::getPointerCast(TheModule.getGlobalVariable("jmp_buffer"),
                                    SetJmpTy->getFunctionParamType(0));
  Value *SetjmpRes = Builder.CreateCall(SetjmpFunction, { JmpBuf });

  // Check if it's the first or second return
  auto *Zero = CI::get(cast<FunctionType>(SetJmpTy)->getReturnType(), 0);
  Value *BrCond = Builder.CreateICmpNE(SetjmpRes, Zero);

  TerminatorInst *T = Builder.CreateCondBr(BrCond, SecondReturn, FirstReturn);
  T->setMetadata("revng.block.type",
                 QMD.tuple((uint32_t) ExternalJumpsHandlerBlock));

  return SetjmpBB;
}

void ExternalJumpsHandler::buildExecutableSegmentsList() {
  SmallVector<Constant *, 10> ExecutableSegments;
  auto Int = [this](uint64_t V) { return ConstantInt::get(RegisterType, V); };
  for (auto &Segment : TheBinary.segments()) {
    if (Segment.IsExecutable) {
      ExecutableSegments.push_back(Int(Segment.StartVirtualAddress));
      ExecutableSegments.push_back(Int(Segment.EndVirtualAddress));
    }
  }

  auto *SegmentsType = ArrayType::get(RegisterType, ExecutableSegments.size());
  auto *SegmentsArray = ConstantArray::get(SegmentsType, ExecutableSegments);

  // Create the array (unnamed)
  auto *SegmentBoundaries = new GlobalVariable(TheModule,
                                               SegmentsArray->getType(),
                                               true,
                                               GlobalValue::InternalLinkage,
                                               SegmentsArray);

  // Create a pointer to the array (segment_boundaries) for support.c
  // consumption
  new GlobalVariable(TheModule,
                     RegisterType->getPointerTo(),
                     true,
                     GlobalValue::ExternalLinkage,
                     ConstantExpr::getPointerCast(SegmentBoundaries,
                                                  RegisterType->getPointerTo()),
                     "segment_boundaries");

  // Create a variable to hold the number of segments (segments_count)
  new GlobalVariable(TheModule,
                     RegisterType,
                     true,
                     GlobalValue::ExternalLinkage,
                     Int(ExecutableSegments.size() / 2),
                     "segments_count");
}

BasicBlock *
ExternalJumpsHandler::createExternalDispatcher(BasicBlock *IsExecutable,
                                               BasicBlock *IsNotExecutable) {
  buildExecutableSegmentsList();

  Constant *IsExecutableFunction = TheModule.getFunction("is_executable");
  BasicBlock *ExternalJumpHandler = BasicBlock::Create(Context,
                                                       "dispatcher.external",
                                                       &TheFunction);
  IRBuilder<> Builder(ExternalJumpHandler);
  Value *PC = Builder.CreateLoad(JumpTargets.pcReg());
  Value *IsExecutableResult = Builder.CreateCall(IsExecutableFunction, { PC });

  // If is_executable returns true go to default, otherwise setjmp
  TerminatorInst *T = Builder.CreateCondBr(IsExecutableResult,
                                           IsNotExecutable,
                                           IsExecutable);
  T->setMetadata("revng.block.type",
                 QMD.tuple((uint32_t) ExternalJumpsHandlerBlock));

  return ExternalJumpHandler;
}

void ExternalJumpsHandler::buildEmptyExecutableSegmentsList() {
  new GlobalVariable(TheModule,
                     RegisterType->getPointerTo(),
                     true,
                     GlobalValue::ExternalLinkage,
                     Constant::getNullValue(RegisterType->getPointerTo()),
                     "segment_boundaries");

  new GlobalVariable(TheModule,
                     RegisterType,
                     true,
                     GlobalValue::ExternalLinkage,
                     Constant::getNullValue(RegisterType),
                     "segments_count");
}

void ExternalJumpsHandler::createExternalJumpsHandler() {

  if (not Arch.isJumpOutSupported()) {
    buildEmptyExecutableSegmentsList();
    return;
  }

  BasicBlock *SerializeAndBranch = createSerializeAndJumpOut();
  BasicBlock *ReturnFromExternal = createReturnFromExternal();
  BasicBlock *SetjmpBB = createSetjmp(SerializeAndBranch, ReturnFromExternal);

  // Insert our BasicBlock as the default case of the dispatcher switch
  BasicBlock *Dispatcher = JumpTargets.dispatcher();
  auto *Switch = cast<SwitchInst>(Dispatcher->getTerminator());
  BasicBlock *DispatcherFail = Switch->getDefaultDest();

  // Replace the default case of the dispatcher with the external jump handler.
  // In practice, perfrom a blind jump, unless the target is within the
  // executable segment of the current module.
  BasicBlock *ExternalJumpHandler = createExternalDispatcher(SetjmpBB,
                                                             DispatcherFail);

  Switch->setDefaultDest(ExternalJumpHandler);
}
