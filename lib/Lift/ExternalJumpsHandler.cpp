/// \file externaljumpsHandler.cpp
/// Inject code to support jumping in non-translated code and handling the
/// comeback.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ProgramCounterHandler.h"

#include "ExternalJumpsHandler.h"

// This name corresponds to a function in `early-linked`.
RegisterIRHelper IsExecutableHelper("is_executable");

using namespace llvm;
using std::string;

static string &
replace(string &Target, const StringRef Search, const StringRef Replace) {
  size_t Position = Target.find(Search.str());
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
  GlobalVariable *SavedRegistersPtr = TheModule.getGlobalVariable("saved_"
                                                                  "registers");
  Instruction *SavedRegisters = createLoad(Builder, SavedRegistersPtr);

  // TODO: if we do not support this architecture, here things will be
  // completely broken
  using namespace model::Architecture;
  using namespace model::Register;
  auto Architecture = Model.Architecture();
  unsigned PCMContextIndex = getPCMContextIndex(Architecture).value_or(0);
  auto *RegisterType = IntegerType::get(Context,
                                        8 * getPointerSize(Architecture));
  Value *GEP = Builder.CreateGEP(RegisterType,
                                 SavedRegisters,
                                 Builder.getInt32(PCMContextIndex));
  Instruction *PCAddress = Builder.CreateLoad(RegisterType, GEP);
  PCH->deserializePCFromSignalContext(Builder, PCAddress, SavedRegisters);

  // Deserialize the ABI registers
  for (auto Register : registers(Model.Architecture())) {
    auto Name = getCSVName(Register);
    GlobalVariable *CSV = TheModule.getGlobalVariable(Name);

    // Not all the registers have a corresponding CSV
    if (CSV != nullptr) {

      auto MaybeMContextIndex = getMContextIndex(Register);
      if (MaybeMContextIndex) {

        Constant *RegisterIndex = Builder.getInt32(*MaybeMContextIndex);
        Value *GEP = Builder.CreateGEP(RegisterType,
                                       SavedRegisters,
                                       RegisterIndex);
        LoadInst *RegisterValue = Builder.CreateLoad(RegisterType, GEP);
        Builder.CreateStore(RegisterValue, CSV);

      } else {

        auto AsmString = getReadRegisterAssembly(Model.Architecture()).str();
        replace(AsmString, "REGISTER", getRegisterName(Register).str());
        std::stringstream ConstraintStringStream;
        ConstraintStringStream << "*m,~{},~{dirflag},~{fpsr},~{flags}";
        auto *FT = FunctionType::get(Type::getVoidTy(Context),
                                     { CSV->getType() },
                                     false);
        InlineAsm *Asm = InlineAsm::get(FT,
                                        AsmString,
                                        ConstraintStringStream.str(),
                                        true,
                                        InlineAsm::AsmDialect::AD_ATT);
        CallInst *AsmCall = Builder.CreateCall(Asm, CSV);
        AsmCall->addParamAttr(0,
                              Attribute::get(Context,
                                             Attribute::ElementType,
                                             CSV->getValueType()));
      }
    }
  }

  Instruction *T = Builder.CreateBr(Dispatcher);
  setBlockType(T, BlockType::ExternalJumpsHandlerBlock);

  return ReturnFromExternal;
}

ExternalJumpsHandler::ExternalJumpsHandler(const model::Binary &Model,
                                           BasicBlock *Dispatcher,
                                           Function &TheFunction,
                                           ProgramCounterHandler *PCH) :
  Model(Model),
  Context(getContext(&TheFunction)),
  QMD(Context),
  TheModule(*TheFunction.getParent()),
  TheFunction(TheFunction),
  Dispatcher(Dispatcher),
  PCH(PCH) {
}

BasicBlock *ExternalJumpsHandler::createSerializeAndJumpOut() {
  // Create the serialize and branch Basic Block
  BasicBlock *Result = BasicBlock::Create(Context,
                                          "serialize_and_jump_out",
                                          &TheFunction);
  IRBuilder<> Builder(Result);
  auto *PC = PCH->loadJumpablePC(Builder);
  auto *JumpablePC = new GlobalVariable(TheModule,
                                        PC->getType(),
                                        false,
                                        GlobalValue::InternalLinkage,
                                        ConstantInt::get(PC->getType(), 0),
                                        "jumpablepc");
  Builder.CreateStore(PC, JumpablePC);

  // Serialize ABI CSVs
  for (model::Register::Values Register : registers(Model.Architecture())) {
    using namespace model::Architecture;
    using namespace model::Register;
    GlobalVariable *CSV = TheModule.getGlobalVariable(getCSVName(Register));

    // Not all the registers have a corresponding CSV
    if (CSV == nullptr)
      continue;

    std::string AsmString = getWriteRegisterAssembly(Model.Architecture())
                              .str();
    StringRef RegisterName = getRegisterName(Register);
    replace(AsmString, "REGISTER", RegisterName);
    std::stringstream ConstraintStringStream;

    ConstraintStringStream << "*m,~{" << RegisterName.str()
                           << "},~{dirflag},~{fpsr},~{flags}";
    auto *FT = FunctionType::get(Type::getVoidTy(Context),
                                 { CSV->getType() },
                                 false);
    InlineAsm *Asm = InlineAsm::get(FT,
                                    AsmString,
                                    ConstraintStringStream.str(),
                                    true,
                                    InlineAsm::AsmDialect::AD_ATT);
    CallInst *AsmCall = Builder.CreateCall(Asm, CSV);
    AsmCall->addParamAttr(0,
                          Attribute::get(Context,
                                         Attribute::ElementType,
                                         CSV->getValueType()));
  }

  // Branch to the Program Counter address
  auto *FT = FunctionType::get(Type::getVoidTy(Context),
                               { JumpablePC->getType() },
                               false);
  InlineAsm *Asm = InlineAsm::get(FT,
                                  getJumpAssembly(Model.Architecture()),
                                  "*m,~{dirflag},~{fpsr},~{flags}",
                                  true,
                                  InlineAsm::AsmDialect::AD_ATT);
  CallInst *AsmCall = Builder.CreateCall(Asm, JumpablePC);
  AsmCall->addParamAttr(0,
                        Attribute::get(Context,
                                       Attribute::ElementType,
                                       JumpablePC->getValueType()));

  Instruction *T = Builder.CreateUnreachable();
  setBlockType(T, BlockType::ExternalJumpsHandlerBlock);

  return Result;
}

llvm::BasicBlock *ExternalJumpsHandler::createSetjmp(BasicBlock *FirstReturn,
                                                     BasicBlock *SecondReturn) {
  using CE = ConstantExpr;
  using CI = ConstantInt;

  BasicBlock *SetjmpBB = BasicBlock::Create(Context, "setjmp", &TheFunction);
  IRBuilder<> Builder(SetjmpBB);

  // Call setjmp
  llvm::Function *SetjmpFunction = TheModule.getFunction("setjmp");
  auto *SetJmpTy = SetjmpFunction->getValueType();
  auto *JmpBuf = CE::getPointerCast(TheModule.getGlobalVariable("jmp_buffer"),
                                    SetJmpTy->getFunctionParamType(0));
  Value *SetjmpRes = Builder.CreateCall(SetjmpFunction, { JmpBuf });

  // Check if it's the first or second return
  auto *Zero = CI::get(cast<FunctionType>(SetJmpTy)->getReturnType(), 0);
  Value *BrCond = Builder.CreateICmpNE(SetjmpRes, Zero);

  Instruction *T = Builder.CreateCondBr(BrCond, SecondReturn, FirstReturn);
  setBlockType(T, BlockType::ExternalJumpsHandlerBlock);

  return SetjmpBB;
}

void ExternalJumpsHandler::buildExecutableSegmentsList() {
  IRBuilder<> Builder(Context);
  IntegerType *Int64 = Builder.getInt64Ty();
  SmallVector<Constant *, 10> ExecutableSegments;
  auto Int = [Int64](uint64_t V) { return ConstantInt::get(Int64, V); };
  for (auto &Segment : Model.Segments()) {
    if (Segment.IsExecutable()) {
      ExecutableSegments.push_back(Int(Segment.StartAddress().address()));
      ExecutableSegments.push_back(Int(Segment.endAddress().address()));
    }
  }

  auto *SegmentsType = ArrayType::get(Int64, ExecutableSegments.size());
  auto *SegmentsArray = ConstantArray::get(SegmentsType, ExecutableSegments);

  // Create the array (unnamed)
  auto *SegmentBoundaries = new GlobalVariable(TheModule,
                                               SegmentsArray->getType(),
                                               true,
                                               GlobalValue::InternalLinkage,
                                               SegmentsArray,
                                               "segment_boundaries_data");

  // Create a pointer to the array (segment_boundaries) for support.c
  // consumption
  new GlobalVariable(TheModule,
                     Int64->getPointerTo(),
                     true,
                     GlobalValue::ExternalLinkage,
                     ConstantExpr::getPointerCast(SegmentBoundaries,
                                                  Int64->getPointerTo()),
                     "segment_boundaries");

  // Create a variable to hold the number of segments (segments_count)
  new GlobalVariable(TheModule,
                     Int64,
                     true,
                     GlobalValue::ExternalLinkage,
                     Int(ExecutableSegments.size() / 2),
                     "segments_count");
}

void ExternalJumpsHandler::createExternalJumpsHandler() {
  auto Assembly = model::Architecture::getJumpAssembly(Model.Architecture());

  if (Assembly.size() == 0) {
    buildExecutableSegmentsList();
    return;
  }

  BasicBlock *SerializeAndBranch = createSerializeAndJumpOut();
  BasicBlock *ReturnFromExternal = createReturnFromExternal();
  BasicBlock *SetjmpBB = createSetjmp(SerializeAndBranch, ReturnFromExternal);

  // Insert our BasicBlock as the default case of the dispatcher switch
  auto *Switch = cast<SwitchInst>(Dispatcher->getTerminator());
  BasicBlock *DispatcherFail = Switch->getDefaultDest();

  // Replace the default case of the dispatcher with the external jump handler.
  // In practice, it performs a blind jump, unless the target is within the
  // executable segment of the current module.
  BasicBlock *ExternalJumpHandler = BasicBlock::Create(Context,
                                                       "dispatcher.external",
                                                       &TheFunction);

  DispatcherFail->replaceAllUsesWith(ExternalJumpHandler);

  {
    BasicBlock *IsExecutable = SetjmpBB;
    BasicBlock *IsNotExecutable = DispatcherFail;
    buildExecutableSegmentsList();

    Function *IsExecutableFunction = getIRHelper("is_executable", TheModule);
    IRBuilder<> Builder(ExternalJumpHandler);
    Value *PC = PCH->loadJumpablePC(Builder);
    Value *IsExecutableResult = Builder.CreateCall(IsExecutableFunction,
                                                   { PC });

    // If is_executable returns true go to default, otherwise setjmp
    Instruction *T = Builder.CreateCondBr(IsExecutableResult,
                                          IsNotExecutable,
                                          IsExecutable);
    setBlockType(T, BlockType::ExternalJumpsHandlerBlock);
  }
}
