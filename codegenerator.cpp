/// \file codegenerator.cpp
/// \brief This file handles the whole translation process from the input
///        assembly to LLVM IR.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <fstream>
#include <queue>
#include <set>
#include <utility>

// LLVM includes
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// Boost includes
#include <boost/icl/interval_map.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/icl/right_open_interval.hpp>

// Local includes
#include "codegenerator.h"
#include "debug.h"
#include "debughelper.h"
#include "externaljumpshandler.h"
#include "functionboundariesdetection.h"
#include "instructiontranslator.h"
#include "jumptargetmanager.h"
#include "ptcinterface.h"
#include "revamb.h"
#include "variablemanager.h"

using namespace llvm;

using std::make_pair;

template<typename T, typename... Args>
inline std::array<T, sizeof...(Args)>
make_array(Args&&... args) {
  return { { std::forward<Args>(args)... } };
}

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

static std::unique_ptr<Module> parseIR(StringRef Path, LLVMContext &Context) {
  std::unique_ptr<Module> Result;
  SMDiagnostic Errors;
  Result = parseIRFile(Path, Errors, Context);

  if (Result.get() == nullptr) {
    Errors.print("revamb", dbgs());
    abort();
  }

  return Result;
}

CodeGenerator::CodeGenerator(BinaryFile &Binary,
                             Architecture& Target,
                             std::string Output,
                             std::string Helpers,
                             std::string EarlyLinked,
                             DebugInfoType DebugInfo,
                             std::string Debug,
                             std::string LinkingInfo,
                             std::string Coverage,
                             std::string BBSummary,
                             bool EnableOSRA,
                             bool DetectFunctionBoundaries,
                             bool EnableLinking,
                             bool ExternalCSVs) :
  TargetArchitecture(Target),
  Context(getGlobalContext()),
  TheModule((new Module("top", Context))),
  OutputPath(Output),
  Debug(new DebugHelper(Output, Debug, TheModule.get(), DebugInfo)),
  Binary(Binary),
  EnableOSRA(EnableOSRA),
  DetectFunctionBoundaries(DetectFunctionBoundaries),
  EnableLinking(EnableLinking),
  ExternalCSVs(ExternalCSVs)
{
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");

  HelpersModule = parseIR(Helpers, Context);
  EarlyLinkedModule = parseIR(EarlyLinked, Context);

  if (Coverage.size() == 0)
    Coverage = Output + ".coverage.csv";
  this->CoveragePath = Coverage;

  if (BBSummary.size() == 0)
    BBSummary = Output + ".bbsummary.csv";
  this->BBSummaryPath = BBSummary;

  // Prepare the linking info CSV
  if (LinkingInfo.size() == 0)
    LinkingInfo = OutputPath + ".li.csv";
  std::ofstream LinkingInfoStream(LinkingInfo);
  LinkingInfoStream << "name,start,end\n";

  auto *Uint8Ty = Type::getInt8Ty(Context);
  auto *ElfHeaderHelper = new GlobalVariable(*TheModule,
                                             Uint8Ty,
                                             true,
                                             GlobalValue::ExternalLinkage,
                                             ConstantInt::get(Uint8Ty, 0),
                                             "elfheaderhelper");
  ElfHeaderHelper->setAlignment(1);
  ElfHeaderHelper->setSection(".elfheaderhelper");

  auto *RegisterType = Type::getIntNTy(Context,
                                       Binary.architecture().pointerSize());
  auto createConstGlobal = [this, &RegisterType] (const Twine &Name,
                                                  uint64_t Value) {
    return new GlobalVariable(*TheModule,
                              RegisterType,
                              true,
                              GlobalValue::ExternalLinkage,
                              ConstantInt::get(RegisterType, Value),
                              Name);
  };

  // These values will be used to populate the auxiliary vectors
  createConstGlobal("e_phentsize", Binary.programHeaderSize());
  createConstGlobal("e_phnum", Binary.programHeadersCount());
  createConstGlobal("phdr_address", Binary.programHeadersAddress());

  for (SegmentInfo &Segment : Binary.segments()) {
    // If it's executable register it as a valid code area
    if (Segment.IsExecutable) {
      // We ignore possible p_filesz-p_memsz mismatches, zeros wouldn't be
      // useful code anyway
      ptc.mmap(Segment.StartVirtualAddress,
               static_cast<const void *>(Segment.Data.data()),
               static_cast<size_t>(Segment.Data.size()));
    }

    std::string Name = Segment.generateName();

    // Get data and size
    auto *DataType = ArrayType::get(Uint8Ty, Segment.size());

    Constant *TheData = nullptr;
    if (Segment.size() == Segment.Data.size()) {
      // Create the array directly from the mmap'd ELF
      TheData = ConstantDataArray::get(Context, Segment.Data);
    } else {
      // If we have extra data at the end we need to create a copy of the
      // segment and append the NULL bytes
      auto FullData = make_unique<uint8_t[]>(Segment.size());
      ::memcpy(FullData.get(),
               Segment.Data.data(),
               Segment.Data.size());
      ::bzero(FullData.get() + Segment.Data.size(),
              Segment.size() - Segment.Data.size());
      auto DataRef = ArrayRef<uint8_t>(FullData.get(), Segment.size());
      TheData = ConstantDataArray::get(Context, DataRef);
    }

    // Create a new global variable
    Segment.Variable = new GlobalVariable(*TheModule,
                                          DataType,
                                          !Segment.IsWriteable,
                                          GlobalValue::ExternalLinkage,
                                          TheData,
                                          Name);

    // Force alignment to 1 and assign the variable to a specific section
    Segment.Variable->setAlignment(1);
    Segment.Variable->setSection("." + Name);

    // Write the linking info CSV
    LinkingInfoStream << "." << Name
                      << ",0x" << std::hex << Segment.StartVirtualAddress
                      << ",0x" << std::hex << Segment.EndVirtualAddress
                      << "\n";

  }

  // Write needed libraries CSV
  std::string NeededLibs = OutputPath + ".need.csv";
  std::ofstream NeededLibsStream(NeededLibs);
  for (const std::string &Library : Binary.neededLibraryNames())
    NeededLibsStream << Library << "\n";
}

Function *CodeGenerator::importHelperFunctionDefinition(StringRef Name) {
  Function *HelperFunction = HelpersModule->getFunction(Name);
  FunctionType *HelperType = HelperFunction->getFunctionType();
  return cast<Function>(TheModule->getOrInsertFunction(Name, HelperType));
}

std::string SegmentInfo::generateName() {
  // Create name from start and size
  std::stringstream NameStream;
  NameStream << "o_"
             << (IsReadable ? "r" : "")
             << (IsWriteable ? "w" : "")
             << (IsExecutable ? "x" : "")
             << "_0x" << std::hex << StartVirtualAddress;

  return NameStream.str();
}

static BasicBlock *replaceFunction(Function *ToReplace) {
  ToReplace->setLinkage(GlobalValue::InternalLinkage);
  ToReplace->dropAllReferences();

  return BasicBlock::Create(ToReplace->getParent()->getContext(),
                            "",
                            ToReplace);
}

static void replaceFunctionWithRet(Function *ToReplace, uint64_t Result) {
  if (ToReplace == nullptr)
    return;

  BasicBlock *Body = replaceFunction(ToReplace);
  Value *ResultValue;

  if (ToReplace->getReturnType()->isVoidTy()) {
    assert(Result == 0);
    ResultValue = nullptr;
  } else if (ToReplace->getReturnType()->isIntegerTy()) {
    auto *ReturnType = cast<IntegerType>(ToReplace->getReturnType());
    ResultValue = ConstantInt::get(ReturnType, Result, false);
  } else {
    llvm_unreachable("No-op functions can only return void or an integer type");
  }

  ReturnInst::Create(ToReplace->getParent()->getContext(), ResultValue, Body);
}

class CpuLoopFunctionPass : public llvm::FunctionPass {
public:
  static char ID;

  CpuLoopFunctionPass() : llvm::FunctionPass(ID) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

char CpuLoopFunctionPass::ID = 0;

static RegisterPass<CpuLoopFunctionPass> X("cpu-loop",
                                           "cpu_loop FunctionPass",
                                           false,
                                           false);

void CpuLoopFunctionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
}

template<class Range, class UnaryPredicate>
auto find_unique(Range&& TheRange, UnaryPredicate Predicate)
  -> decltype(*TheRange.begin()) {

  const auto Begin = TheRange.begin();
  const auto End = TheRange.end();

  auto It = std::find_if(Begin, End, Predicate);
  auto Result = It;
  assert(Result != End);
  assert(std::find_if(++It, End, Predicate) == End);

  return *Result;
}

template<class Range>
auto find_unique(Range&& TheRange)
  -> decltype(*TheRange.begin()) {

  const auto Begin = TheRange.begin();
  const auto End = TheRange.end();

  auto Result = Begin;
  assert(Begin != End && ++Result == End);
  (void) Result;
  (void) End;

  return *Begin;
}

bool CpuLoopFunctionPass::runOnFunction(Function &F) {
  // cpu_loop must return void
  assert(F.getReturnType()->isVoidTy());

  Module *TheModule = F.getParent();

  // Part 1: remove the backedge of the main infinite loop
  const LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  const Loop *OutermostLoop = find_unique(LI);

  BasicBlock *Header = OutermostLoop->getHeader();

  // Check that the header has only one predecessor inside the loop
  auto IsInLoop = [&OutermostLoop] (BasicBlock *Predecessor) {
    return OutermostLoop->contains(Predecessor);
  };
  BasicBlock *Footer = find_unique(predecessors(Header), IsInLoop);

  // Assert on the type of the last instruction (branch or brcond)
  assert(Footer->end() != Footer->begin());
  Instruction *LastInstruction = &*--Footer->end();
  assert(isa<BranchInst>(LastInstruction));

  // Remove the last instruction and replace it with a ret
  LastInstruction->eraseFromParent();
  ReturnInst::Create(F.getParent()->getContext(), Footer);

  // Part 2: replace the call to cpu_*_exec with exception_index
  auto IsCpuExec = [] (Function& TheFunction) {
    StringRef Name = TheFunction.getName();
    return Name.startswith("cpu_") && Name.endswith("_exec");
  };
  Function& CpuExec = find_unique(F.getParent()->functions(), IsCpuExec);

  User *CallUser = find_unique(CpuExec.users(), [&F] (User *TheUser) {
      auto *TheInstruction = dyn_cast<Instruction>(TheUser);

      if (TheInstruction == nullptr)
        return false;

      return TheInstruction->getParent()->getParent() == &F;
    });

  auto *Call = cast<CallInst>(CallUser);
  assert(Call->getCalledFunction() == &CpuExec);
  Value *ExceptionIndex = TheModule->getOrInsertGlobal("exception_index",
                                                       CpuExec.getReturnType());
  Value *LoadExceptionIndex = new LoadInst(ExceptionIndex, "", Call);
  Call->replaceAllUsesWith(LoadExceptionIndex);
  Call->eraseFromParent();

  return true;
}

class CpuLoopExitPass : public llvm::ModulePass {
public:
  static char ID;

  CpuLoopExitPass() : llvm::ModulePass(ID), VM(0) { }
  CpuLoopExitPass(VariableManager *VM) :
    llvm::ModulePass(ID),
    VM(VM) { }

  bool runOnModule(llvm::Module& M) override;
private:
  VariableManager *VM;
};

char CpuLoopExitPass::ID = 0;

static RegisterPass<CpuLoopExitPass> Y("cpu-loop-exit",
                                       "cpu_loop_exit Pass",
                                       false,
                                       false);

static void purgeNoReturn(Function *F) {
  auto &Context = F->getParent()->getContext();

  if (F->hasFnAttribute(Attribute::NoReturn))
    F->removeFnAttr(Attribute::NoReturn);

  for (User *U : F->users())
    if (auto *Call = dyn_cast<CallInst>(U))
      if (Call->hasFnAttr(Attribute::NoReturn)) {
        auto OldAttr = Call->getAttributes();
        auto NewAttr = OldAttr.removeAttribute(Context,
                                               AttributeSet::FunctionIndex,
                                               Attribute::NoReturn);
        Call->setAttributes(NewAttr);
      }
}

static ReturnInst *createRet(Instruction *Position) {
  Function *F = Position->getParent()->getParent();
  purgeNoReturn(F);

  Type *ReturnType = F->getFunctionType()->getReturnType();
  if (ReturnType->isVoidTy()) {
    return ReturnInst::Create(F->getParent()->getContext(), nullptr, Position);
  } else if (ReturnType->isIntegerTy()) {
    auto *Zero = ConstantInt::get(static_cast<IntegerType*>(ReturnType), 0);
    return ReturnInst::Create(F->getParent()->getContext(), Zero, Position);
  } else {
    assert("Return type not supported");
  }

  return nullptr;
}

/// Find all calls to cpu_loop_exit and replace them with:
///
/// * call cpu_loop
/// * set cpu_loop_exiting = true
/// * return
///
/// Then look for all the callers of the function calling cpu_loop_exit and make
/// them check whether they should return immediately (cpu_loop_exiting == true)
/// or not.
/// Then when we reach the root function, set cpu_loop_exiting to false after
/// the call.
bool CpuLoopExitPass::runOnModule(llvm::Module& M) {
  Function *CpuLoopExit = M.getFunction("cpu_loop_exit");

  // Nothing to do here
  if (CpuLoopExit == nullptr)
    return false;

  purgeNoReturn(CpuLoopExit);

  Function *CpuLoop = M.getFunction("cpu_loop");
  LLVMContext &Context = M.getContext();
  IntegerType *BoolType = Type::getInt1Ty(Context);
  std::set<Function *> FixedCallers;
  Constant *CpuLoopExitingVariable = nullptr;
  CpuLoopExitingVariable = new GlobalVariable(M,
                                              BoolType,
                                              false,
                                              GlobalValue::CommonLinkage,
                                              ConstantInt::getFalse(BoolType),
                                              StringRef("cpu_loop_exiting"));

  assert(CpuLoop != nullptr);

  std::queue<User *> CpuLoopExitUsers;
  for (User *TheUser : CpuLoopExit->users())
    CpuLoopExitUsers.push(TheUser);

  while (!CpuLoopExitUsers.empty()) {
    auto *Call = cast<CallInst>(CpuLoopExitUsers.front());
    CpuLoopExitUsers.pop();
    assert(Call->getCalledFunction() == CpuLoopExit);

    // Call cpu_loop
    auto *EnvType = CpuLoop->getFunctionType()->getParamType(0);
    auto *AddressComputation = VM->computeEnvAddress(EnvType, Call);
    CallInst::Create(CpuLoop, { AddressComputation }, "", Call);

    // Set cpu_loop_exiting to true
    new StoreInst(ConstantInt::getTrue(BoolType), CpuLoopExitingVariable, Call);

    // Return immediately
    createRet(Call);
    auto *Unreach = cast<UnreachableInst>(&*++Call->getIterator());
    Unreach->eraseFromParent();

    Function *Caller = Call->getParent()->getParent();

    // Remove the call to cpu_loop_exit
    Call->eraseFromParent();

    if (FixedCallers.find(Caller) == FixedCallers.end()) {
      FixedCallers.insert(Caller);

      std::queue<Value *> WorkList;
      WorkList.push(Caller);

      while (!WorkList.empty()) {
        Value *F = WorkList.front();
        WorkList.pop();

        for (User *RecUser : F->users()) {
          auto *RecCall = dyn_cast<CallInst>(RecUser);
          if (RecCall == nullptr) {
            auto *Cast = dyn_cast<ConstantExpr>(RecUser);
            assert(Cast != nullptr && "Unexpected user");
            assert(Cast->getOperand(0) == F && Cast->isCast());
            WorkList.push(Cast);
            continue;
          }

          Function *RecCaller = RecCall->getParent()->getParent();

          // TODO: make this more reliable than using function name
          if (RecCaller->getName() == "root") {
            // If we got to the translated function, just reset cpu_loop_exiting
            // to false
            new StoreInst(ConstantInt::getFalse(BoolType),
                          CpuLoopExitingVariable,
                          &*++RecCall->getIterator());
          } else {
            // If the caller is a QEMU helper function make it check
            // cpu_loop_exiting and if it's true, make it return

            // Split BB
            BasicBlock *OldBB = RecCall->getParent();
            BasicBlock::iterator SplitPoint = ++RecCall->getIterator();
            assert(SplitPoint != OldBB->end());
            BasicBlock *NewBB = OldBB->splitBasicBlock(SplitPoint);

            // Add a BB with a ret
            BasicBlock *QuitBB = BasicBlock::Create(Context,
                                                    "cpu_loop_exit_return",
                                                    RecCaller,
                                                    NewBB);
            UnreachableInst *Temp = new UnreachableInst(Context, QuitBB);
            createRet(Temp);
            Temp->eraseFromParent();

            // Check value of cpu_loop_exiting
            auto *Branch = cast<BranchInst>(&*++(RecCall->getIterator()));
            auto *Compare = new ICmpInst(Branch,
                                         CmpInst::ICMP_EQ,
                                         new LoadInst(CpuLoopExitingVariable,
                                                      "",
                                                      Branch),
                                         ConstantInt::getTrue(BoolType));

            BranchInst::Create(QuitBB, NewBB, Compare,  Branch);
            Branch->eraseFromParent();

            // Add to the work list only if it hasn't been fixed already
            if (FixedCallers.find(RecCaller) == FixedCallers.end()) {
              FixedCallers.insert(RecCaller);
              WorkList.push(RecCaller);
            }
          }
        }
      }
    }
  }

  return true;
}

/// Removes all the basic blocks without predecessors from F
// TODO: this is not efficient, but it shouldn't be critical
static void purgeDeadBlocks(Function *F) {
  std::vector<BasicBlock *> Kill;
  do {
    for (BasicBlock *Dead : Kill)
      DeleteDeadBlock(Dead);
    Kill.clear();

    // Skip the first basic block
    for (BasicBlock &BB : make_range(++F->begin(), F->end()))
      if (pred_empty(&BB))
        Kill.push_back(&BB);

  } while (!Kill.empty());

}

void CodeGenerator::translate(uint64_t VirtualAddress) {
  using FT = FunctionType;

  // Declare useful functions
  auto *AbortTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *AbortFunction = TheModule->getOrInsertFunction("abort", AbortTy);

  importHelperFunctionDefinition("target_set_brk");
  TheModule->getOrInsertFunction("syscall_init",
                                 FT::get(Type::getVoidTy(Context), { }, false));

  // Instantiate helpers
  VariableManager Variables(*TheModule, *HelpersModule, TargetArchitecture);
  GlobalVariable *PCReg = Variables.getByEnvOffset(ptc.pc, "pc").first;
  GlobalVariable *SPReg = Variables.getByEnvOffset(ptc.sp, "sp").first;

  IRBuilder<> Builder(Context);

  // Create main function
  auto *MainType  = FT::get(Builder.getVoidTy(),
                            { SPReg->getType()->getPointerElementType() },
                            false);
  auto *MainFunction = Function::Create(MainType,
                                        Function::ExternalLinkage,
                                        "root",
                                        TheModule.get());

  // Create the first basic block and create a placeholder for variable
  // allocations
  BasicBlock *Entry = BasicBlock::Create(Context, "entrypoint", MainFunction);
  Builder.SetInsertPoint(Entry);

  // Create revamb.inputarch named metadata.
  QuickMetadata QMD(Context);
  NamedMDNode *InputArchMD;
  const char *MDName = "revamb.input.architecture";
  InputArchMD = TheModule->getOrInsertNamedMetadata(MDName);
  const Architecture &Arch = Binary.architecture();
  // Currently revamb.inputarch is composed as follows:
  //
  // revamb.inputarch = {
  //   InstructionAlignment,
  //   DelaySlotSize,
  //   PCRegisterName,
  //   SPRegisterName,
  //   ABIRegisters
  // }

  const SmallVector<ABIRegister, 20> &ABIRegisters = Arch.abiRegisters();
  SmallVector<Metadata*, 20> ABIRegMetadata;
  for (auto Register : ABIRegisters)
    ABIRegMetadata.push_back(MDString::get(Context, Register.name()));

  auto *Tuple = MDTuple::get(Context, {
    QMD.get(static_cast<uint32_t>(Arch.instructionAlignment())),
    QMD.get(static_cast<uint32_t>(Arch.delaySlotSize())),
    QMD.get("pc"),
    QMD.get(Arch.stackPointerRegister()),
    QMD.tuple(ArrayRef<Metadata*>(ABIRegMetadata)),
  });
  InputArchMD->addOperand(Tuple);

  // Create an instance of JumpTargetManager
  JumpTargetManager JumpTargets(MainFunction, PCReg, Binary, EnableOSRA);

  if (VirtualAddress == 0) {
    JumpTargets.harvestGlobalData();
    VirtualAddress = Binary.entryPoint();
  }
  JumpTargets.registerJT(VirtualAddress, JumpTargetManager::GlobalData);

  // Initialize the program counter
  auto *StartPC = ConstantInt::get(PCReg->getType()->getPointerElementType(),
                                   VirtualAddress);
  // Use this instruction as the delimiter for local variables
  auto *Delimiter = Builder.CreateStore(StartPC, PCReg);

  // We need to remember this instruction so we can later insert a call here.
  // The problem is that up until now we don't know where our CPUState structure is.
  // After the translation we will and use this information to create a call to
  // a helper function.
  // TODO: we need a more elegant solution here
  auto *InitEnvInsertPoint = Delimiter;
  Builder.CreateStore(&*MainFunction->arg_begin(), SPReg);

  // Fake jumps to the dispatcher-related basic blocks. This way all the blocks
  // are always reachable.
  auto *ReachSwitch = Builder.CreateSwitch(Builder.getInt8(0),
                                           JumpTargets.dispatcher());
  ReachSwitch->addCase(Builder.getInt8(1), JumpTargets.anyPC());
  ReachSwitch->addCase(Builder.getInt8(2), JumpTargets.unexpectedPC());

  std::tie(VirtualAddress, Entry) = JumpTargets.peek();

  std::vector<BasicBlock *> Blocks;

  InstructionTranslator Translator(Builder,
                                   Variables,
                                   JumpTargets,
                                   Blocks,
                                   Binary.architecture(),
                                   TargetArchitecture);

  while (Entry != nullptr) {
    Builder.SetInsertPoint(Entry);

    // TODO: what if create a new instance of an InstructionTranslator here?
    Translator.reset();

    // TODO: rename this type
    PTCInstructionListPtr InstructionList(new PTCInstructionList);
    size_t ConsumedSize = 0;

    ConsumedSize = ptc.translate(VirtualAddress, InstructionList.get());
    SmallSet<unsigned, 1> ToIgnore;
    ToIgnore = Translator.preprocess(InstructionList.get());

    DBG("ptc", dumpTranslation(dbg, InstructionList.get()));

    Variables.newFunction(Delimiter, InstructionList.get());
    unsigned j = 0;
    MDNode* MDOriginalInstr = nullptr;
    bool StopTranslation = false;
    uint64_t PC = VirtualAddress;
    uint64_t NextPC = 0;
    uint64_t EndPC = VirtualAddress + ConsumedSize;
    const auto InstructionCount = InstructionList->instruction_count;
    using IT = InstructionTranslator;
    IT::TranslationResult Result;
    bool ForceNewBlock = false;

    // Handle the first PTC_INSTRUCTION_op_debug_insn_start
    {
      PTCInstruction *NextInstruction = nullptr;
      for (unsigned k = 1; k < InstructionCount; k++) {
        PTCInstruction *I = &InstructionList->instructions[k];
        if (I->opc == PTC_INSTRUCTION_op_debug_insn_start
            && ToIgnore.count(k) == 0) {
          NextInstruction = I;
          break;
        }
      }
      PTCInstruction *Instruction = &InstructionList->instructions[j];
      std::tie(Result,
               MDOriginalInstr,
               PC,
               NextPC) = Translator.newInstruction(Instruction,
                                                   NextInstruction,
                                                   EndPC,
                                                   true,
                                                   false);
      j++;
    }

    // TODO: shall we move this whole loop in InstructionTranslator?
    for (; j < InstructionCount && !StopTranslation; j++) {
      if (ToIgnore.count(j) != 0)
        continue;

      PTCInstruction Instruction = InstructionList->instructions[j];
      PTCOpcode Opcode = Instruction.opc;

      Blocks.clear();
      Blocks.push_back(Builder.GetInsertBlock());

      switch(Opcode) {
      case PTC_INSTRUCTION_op_discard:
        // Instructions we don't even consider
        break;
      case PTC_INSTRUCTION_op_debug_insn_start:
        {
          // Find next instruction, if there is one
          PTCInstruction *NextInstruction = nullptr;
          for (unsigned k = j + 1; k < InstructionCount; k++) {
            PTCInstruction *I = &InstructionList->instructions[k];
            if (I->opc == PTC_INSTRUCTION_op_debug_insn_start
                && ToIgnore.count(k) == 0) {
              NextInstruction = I;
              break;
            }
          }

          std::tie(Result,
                   MDOriginalInstr,
                   PC,
                   NextPC) = Translator.newInstruction(&Instruction,
                                                       NextInstruction,
                                                       EndPC,
                                                       false,
                                                       ForceNewBlock);

          ForceNewBlock = false;
          break;
        }
      case PTC_INSTRUCTION_op_call:
        {
          Result = Translator.translateCall(&Instruction);

          // Sometimes libtinycode terminates a basic block with a call, in this
          // case force a fallthrough
          auto &IL = InstructionList;
          if (j == IL->instruction_count - 1) {
            using JTM = JumpTargetManager;
            Builder.CreateBr(notNull(JumpTargets.registerJT(EndPC,
                                                            JTM::PostHelper)));
          }

          break;
        }
      default:
        Result = Translator.translate(&Instruction, PC, NextPC);
      }

      switch (Result) {
      case IT::Success:
        // No-op
        break;
      case IT::Abort:
        Builder.CreateCall(AbortFunction);
        Builder.CreateUnreachable();
        StopTranslation = true;
        break;
      case IT::Stop:
        StopTranslation = true;
        break;
      case IT::ForceNewPC:
        ForceNewBlock = true;
        break;
      }

      // Create a new metadata referencing the PTC instruction we have just
      // translated
      std::stringstream PTCStringStream;
      dumpInstruction(PTCStringStream, InstructionList.get(), j);
      std::string PTCString = PTCStringStream.str() + "\n";
      MDString *MDPTCString = MDString::get(Context, PTCString);
      MDNode* MDPTCInstr = MDNode::getDistinct(Context, MDPTCString);

      // Set metadata for all the new instructions
      for (BasicBlock *Block : Blocks) {
        BasicBlock::iterator I = Block->end();
        while (I != Block->begin() && !(--I)->hasMetadata()) {
          I->setMetadata(OriginalInstrMDKind, MDOriginalInstr);
          I->setMetadata(PTCInstrMDKind, MDPTCInstr);
        }
      }

    } // End loop over instructions

    if (ForceNewBlock)
      JumpTargets.registerJT(EndPC, JumpTargetManager::PostHelper);

    // We might have a leftover block, probably due to the block created after
    // the last call to exit_tb
    auto *LastBlock = Builder.GetInsertBlock();
    if (LastBlock->empty())
      LastBlock->eraseFromParent();
    else if (!LastBlock->rbegin()->isTerminator()) {
      // Something went wrong, probably a mistranslation
      Builder.CreateUnreachable();
    }

    // Obtain a new program counter to translate
    std::tie(VirtualAddress, Entry) = JumpTargets.peek();
  } // End translations loop

  importHelperFunctionDefinition("cpu_loop");
  Function *CpuLoop = HelpersModule->getFunction("cpu_loop");
  assert(CpuLoop != nullptr);

  legacy::FunctionPassManager CpuLoopPM(TheModule.get());
  CpuLoopPM.add(new LoopInfoWrapperPass());
  CpuLoopPM.add(new CpuLoopFunctionPass());
  CpuLoopPM.run(*CpuLoop);

  // CpuLoopFunctionPass expects a variable name exception_index to exist
  Variables.getByEnvOffset(ptc.exception_index, "exception_index");

  // Handle some specific QEMU functions as no-ops or abort
  auto NoOpFunctionNames = make_array<const char *>("qemu_log_mask",
                                                    "fprintf",
                                                    "cpu_dump_state",
                                                    "mmap_lock",
                                                    "mmap_unlock",
                                                    "pthread_cond_broadcast",
                                                    "pthread_mutex_unlock",
                                                    "pthread_mutex_lock",
                                                    "pthread_cond_wait",
                                                    "pthread_cond_signal",
                                                    "cpu_exit",
                                                    "start_exclusive",
                                                    "process_pending_signals",
                                                    "end_exclusive");
  auto AbortFunctionNames = make_array<const char *>("cpu_restore_state",
                                                     "gdb_handlesig",
                                                     "queue_signal",
                                                     "cpu_mips_exec",
                                                     // syscall.c
                                                     "print_syscall",
                                                     "print_syscall_ret",
                                                     "do_ioctl_dm",
                                                     // ARM cpu_loop
                                                     "EmulateAll",
                                                     "cpu_abort",
                                                     "do_arm_semihosting");

  // EmulateAll: requires access to the opcode
  // do_arm_semihosting: we don't care about semihosting

  // Initializes the CPUState which is important on x86 architecture.
  if (HelpersModule->getFunction("initialize_env") != nullptr) {
    Function *InitEnv = importHelperFunctionDefinition("initialize_env");
    auto *CPUStateType = InitEnv->getFunctionType()->getParamType(0);
    Instruction *InsertBefore = InitEnvInsertPoint;
    auto *AddressComputation = Variables.computeEnvAddress(CPUStateType, InsertBefore);
    CallInst::Create(InitEnv, { AddressComputation }, "", InsertBefore);
  }

  // From syscall.c
  new GlobalVariable(*TheModule,
                     Type::getInt32Ty(Context),
                     false,
                     GlobalValue::CommonLinkage,
                     ConstantInt::get(Type::getInt32Ty(Context), 0),
                     StringRef("do_strace"));

  for (auto Name : NoOpFunctionNames)
    replaceFunctionWithRet(HelpersModule->getFunction(Name), 0);

  for (auto Name : AbortFunctionNames) {
    Function *TheFunction = HelpersModule->getFunction(Name);
    if (TheFunction != nullptr) {
      assert(HelpersModule->getFunction("abort") != nullptr);
      BasicBlock *NewBody = replaceFunction(TheFunction);
      CallInst::Create(HelpersModule->getFunction("abort"), { }, NewBody);
      new UnreachableInst(Context, NewBody);
    }
  }

  replaceFunctionWithRet(HelpersModule->getFunction("page_check_range"), 1);
  replaceFunctionWithRet(HelpersModule->getFunction("page_get_flags"),
                         0xffffffff);

  // HACK: the LLVM linker does not import non-static functions anymore if
  //       LinkOnlyNeeded is specified. We don't want this so mark all the
  //       non-static symbols not directly imported as static.
  {
    std::set<StringRef> Declarations;
    for (auto& GV : TheModule->functions())
      if (GV.isDeclaration())
        Declarations.insert(GV.getName());

    for (auto& GV : TheModule->globals())
      if (GV.isDeclaration())
        Declarations.insert(GV.getName());

    for (auto& GV : HelpersModule->functions())
      if (!GV.isDeclaration()
          && Declarations.find(GV.getName()) == Declarations.end()
          && GV.hasExternalLinkage())
        GV.setLinkage(GlobalValue::InternalLinkage);

    for (auto& GV : HelpersModule->globals())
      if (!GV.isDeclaration()
          && Declarations.find(GV.getName()) == Declarations.end()
          && GV.hasExternalLinkage())
        GV.setLinkage(GlobalValue::InternalLinkage);
  }

  if (EnableLinking) {
    Linker TheLinker(*TheModule);
    bool Result = TheLinker.linkInModule(std::move(HelpersModule),
                                         Linker::LinkOnlyNeeded);
    assert(!Result && "Linking failed");
  }

  Variables.setDataLayout(&TheModule->getDataLayout());

  legacy::PassManager PM;
  PM.add(createSROAPass());
  PM.add(new CpuLoopExitPass(&Variables));
  PM.add(Variables.createCorrectCPUStateUsagePass());
  PM.add(createDeadCodeEliminationPass());
  PM.run(*TheModule);

  JumpTargets.finalizeJumpTargets();

  purgeDeadBlocks(MainFunction);

  if (DetectFunctionBoundaries) {
    legacy::FunctionPassManager FPM(&*TheModule);
    FPM.add(new FunctionBoundariesDetectionPass(&JumpTargets, ""));
    FPM.run(*MainFunction);
  }

  JumpTargets.createJTReasonMD();

  // Link early-linked.c
  // TODO: moving this too earlier seems to break things
  {
    Linker TheLinker(*TheModule);
    bool Result = TheLinker.linkInModule(std::move(EarlyLinkedModule),
                                         Linker::None);
    assert(!Result && "Linking failed");
  }

  ExternalJumpsHandler JumpOutHandler(Binary, JumpTargets, *MainFunction);
  JumpOutHandler.createExternalJumpsHandler();

  JumpTargets.noReturn().cleanup();

  Translator.finalizeNewPCMarkers(CoveragePath);

  Variables.finalize(ExternalCSVs);

  Debug->generateDebugInfo();

}

void CodeGenerator::serialize() {
  // Ask the debug handler if it already has a good copy of the IR, if not dump
  // it
  if (!Debug->copySource()) {
    std::ofstream Output(OutputPath);
    Debug->print(Output, false);
  }
}
