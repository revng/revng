/// \file
/// \brief This file handles the whole translation process from the input
///        assembly to LLVM IR.

// Standard includes
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>
#include <fstream>
#include <set>
#include <queue>

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
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar.h"

// Local includes
#include "codegenerator.h"
#include "debughelper.h"
#include "instructiontranslator.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"
#include "ptcinterface.h"
#include "variablemanager.h"

using namespace llvm;

template<typename T, typename... Args>
inline std::array<T, sizeof...(Args)>
make_array(Args&&... args)
{ return { std::forward<Args>(args)... };  }

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

CodeGenerator::CodeGenerator(std::string Input,
                             Architecture& Target,
                             std::string Output,
                             std::string Helpers,
                             DebugInfoType DebugInfo,
                             std::string Debug,
                             std::string LinkingInfoPath) :
  TargetArchitecture(Target),
  Context(getGlobalContext()),
  TheModule((new Module("top", Context))),
  OutputPath(Output),
  Debug(new DebugHelper(Output, Debug, TheModule.get(), DebugInfo))
{
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");

  SMDiagnostic Errors;
  HelpersModule = parseIRFile(Helpers, Errors, Context);

  auto BinaryOrErr = object::createBinary(Input);
  assert(BinaryOrErr && "Couldn't open the input file");

  BinaryHandle = std::move(BinaryOrErr.get());

  // We only support ELF for now
  auto *TheBinary = cast<object::ObjectFile>(BinaryHandle.getBinary());

  SourceArchitecture = Architecture(1,
                                    TheBinary->isLittleEndian(),
                                    TheBinary->getBytesInAddress() * 8);

  if (SourceArchitecture.pointerSize() == 32) {
    if (SourceArchitecture.isLittleEndian()) {
      importGlobalData<object::ELF32LE>(TheBinary, LinkingInfoPath);
    } else {
      importGlobalData<object::ELF32BE>(TheBinary, LinkingInfoPath);
    }
  } else if (SourceArchitecture.pointerSize() == 64) {
    if (SourceArchitecture.isLittleEndian()) {
      importGlobalData<object::ELF64LE>(TheBinary, LinkingInfoPath);
    } else {
      importGlobalData<object::ELF64BE>(TheBinary, LinkingInfoPath);
    }
  } else {
    assert("Unexpect address size");
  }
}

template<typename T>
void CodeGenerator::importGlobalData(object::ObjectFile *TheBinary,
                                     std::string LinkingInfoPath) {
  // Parse the ELF file
  std::error_code EC;
  object::ELFFile<T> TheELF(TheBinary->getData(), EC);
  assert(!EC && "Error while loading the ELF file");

  // Prepare the linking info CSV
  if (LinkingInfoPath.size() == 0)
    LinkingInfoPath = OutputPath + ".li.csv";
  std::ofstream LinkingInfoStream(LinkingInfoPath);
  LinkingInfoStream << "name,start,end" << std::endl;

  auto *Uint8Ty = Type::getInt8Ty(Context);
  auto *ElfHeaderHelper = new GlobalVariable(*TheModule,
                                             Uint8Ty,
                                             true,
                                             GlobalValue::InternalLinkage,
                                             ConstantInt::get(Uint8Ty, 0),
                                             "");
  ElfHeaderHelper->setAlignment(1);
  ElfHeaderHelper->setSection(".elfheaderhelper");


  // Loop over the program headers looking for PT_LOAD segments, read them out
  // and create a global variable for each one of them (writable or read-only),
  // assign them a section and output information about them in the linking info
  // CSV
  for (auto &ProgramHeader : TheELF.program_headers())
    if (ProgramHeader.p_type == ELF::PT_LOAD) {
      auto EndAddress = ProgramHeader.p_vaddr + ProgramHeader.p_memsz;

      // If it's executable register it as a valid code area
      if (ProgramHeader.p_flags & ELF::PF_X)
        ExecutableRanges.push_back(std::make_pair(ProgramHeader.p_vaddr,
                                                  EndAddress));

      // Create name from start and size
      std::stringstream NameStream;
      NameStream << ".o_"
                 << (ProgramHeader.p_flags & ELF::PF_R ? "r" : "")
                 << (ProgramHeader.p_flags & ELF::PF_W ? "w" : "")
                 << (ProgramHeader.p_flags & ELF::PF_X ? "x" : "")
                 << "_0x" << std::hex << ProgramHeader.p_vaddr;
      std::string Name = NameStream.str();

      // Get data and size
      auto *DataType = ArrayType::get(Uint8Ty,
                                      ProgramHeader.p_memsz);

      Constant *TheData = nullptr;
      if (ProgramHeader.p_memsz == ProgramHeader.p_filesz) {
        // Create the array directly from the mmap'd ELF
        auto FileData = ArrayRef<uint8_t>(TheELF.base() + ProgramHeader.p_offset,
                                          ProgramHeader.p_filesz);
        TheData = ConstantDataArray::get(Context, FileData);
      } else {
        // If we have extra data at the end we need to create a copy of the
        // segment and append the NULL bytes
        auto FullData = std::make_unique<uint8_t[]>(ProgramHeader.p_memsz);
        ::memcpy(FullData.get(),
                 TheELF.base() + ProgramHeader.p_offset,
                 ProgramHeader.p_filesz);
        ::bzero(FullData.get() + ProgramHeader.p_filesz,
                ProgramHeader.p_memsz - ProgramHeader.p_filesz);
        auto DataRef = ArrayRef<uint8_t>(FullData.get(), ProgramHeader.p_memsz);
        TheData = ConstantDataArray::get(Context, DataRef);
      }

      // Check if it's writable
      bool IsConstant = !(ProgramHeader.p_flags & ELF::PF_W);

      // Create a new global variable
      auto *GlobalDataVar = new GlobalVariable(*TheModule,
                                               DataType,
                                               IsConstant,
                                               GlobalValue::InternalLinkage,
                                               TheData,
                                               "");

      // Force alignment to 1 and assign the variable to a specific section
      GlobalDataVar->setAlignment(1);
      GlobalDataVar->setSection(Name);

      // Write the linking info CSV
      LinkingInfoStream << Name
                        << ",0x" << std::hex << ProgramHeader.p_vaddr
                        << ",0x" << std::hex << EndAddress
                        << std::endl;
    }
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
    assert("No-op functions can only return void or an integer type");
  }

  ReturnInst::Create(ToReplace->getParent()->getContext(), ResultValue, Body);
}

class CpuLoopFunctionPass : public llvm::FunctionPass {
public:
  static char ID;

  CpuLoopFunctionPass() : llvm::FunctionPass(ID) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

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

static ReturnInst *createRet(Instruction *Position) {
  Function *F = Position->getParent()->getParent();
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

  for (User *TheUser : CpuLoopExit->users()) {
    auto *Call = cast<CallInst>(TheUser);
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

    // Remove the call to cpu_loop_exit
    Function *Caller = Call->getParent()->getParent();
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


void CodeGenerator::translate(size_t LoadAddress,
                              ArrayRef<uint8_t> Code,
                              size_t VirtualAddress,
                              std::string Name) {
  const uint8_t *CodePointer = Code.data();
  const uint8_t *CodeEnd = CodePointer + Code.size();

  IRBuilder<> Builder(Context);

  // Create main function
  auto *MainType  = FunctionType::get(Builder.getVoidTy(), false);
  auto *MainFunction = Function::Create(MainType,
                                        Function::ExternalLinkage,
                                        Name,
                                        TheModule.get());

  Debug->newFunction(MainFunction);

  // Create the first basic block and create a placeholder for variable
  // allocations
  BasicBlock *Entry = BasicBlock::Create(Context,
                                         "entrypoint",
                                         MainFunction);
  Builder.SetInsertPoint(Entry);
  Instruction *Delimiter = Builder.CreateUnreachable();

  // Instantiate helpers
  VariableManager Variables(*TheModule,
                            *HelpersModule);

  GlobalVariable *PCReg = Variables.getByEnvOffset(ptc.pc, "pc");

  JumpTargetManager JumpTargets(*TheModule,
                                PCReg,
                                MainFunction,
                                ExecutableRanges);

  // Create a new block where the translation will start and emit a tautological
  // branch to it, with false part being the dispatcher, which is always
  // reachable
  Entry = BasicBlock::Create(Context,
                             "translation-start",
                             MainFunction);
  Builder.CreateCondBr(Builder.getTrue(), Entry, JumpTargets.dispatcher());


  std::map<std::string, BasicBlock *> LabeledBasicBlocks;
  std::vector<BasicBlock *> Blocks;

  InstructionTranslator Translator(Builder,
                                   Variables,
                                   JumpTargets,
                                   LabeledBasicBlocks,
                                   Blocks,
                                   *TheModule,
                                   MainFunction,
                                   SourceArchitecture,
                                   TargetArchitecture);

  ptc.mmap(LoadAddress, Code.data(), Code.size());

  while (Entry != nullptr) {
    Builder.SetInsertPoint(Entry);

    LabeledBasicBlocks.clear();

    // TODO: rename this type
    PTCInstructionListPtr InstructionList(new PTCInstructionList);
    size_t ConsumedSize = 0;

    assert(CodeEnd > CodePointer);

    ConsumedSize = ptc.translate(VirtualAddress,
                                 InstructionList.get());
    uint64_t NextPC = VirtualAddress + ConsumedSize;

    dumpTranslation(std::cerr, InstructionList.get());

    Variables.newFunction(Delimiter, InstructionList.get());
    unsigned j = 0;
    MDNode* MDOriginalInstr = nullptr;
    bool StopTranslation = false;

    // Handle the first PTC_INSTRUCTION_op_debug_insn_start
    {
      PTCInstruction *Instruction = &InstructionList->instructions[j];
      auto Result = Translator.newInstruction(Instruction, true);
      std::tie(StopTranslation, MDOriginalInstr) = Result;
      j++;
    }

    for (; j < InstructionList->instruction_count && !StopTranslation; j++) {
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
          std::tie(StopTranslation,
                   MDOriginalInstr) = Translator.newInstruction(&Instruction,
                                                                false);
          break;
        }
      case PTC_INSTRUCTION_op_call:
        Translator.translateCall(&Instruction);

        // Sometimes libtinycode terminates a basic block with a call, in this
        // case force a fallthrough
        // TODO: investigate why this happens
        if (j == InstructionList->instruction_count - 1)
          Builder.CreateBr(JumpTargets.getBlockAt(NextPC));

        break;
      default:
        Translator.translate(&Instruction);
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

    Translator.closeLastInstruction(NextPC);

    // If we're out of jump targets, try to harvest some more
    if (JumpTargets.empty()) {
      legacy::PassManager PM;
      // Before looking for writes to the PC, give a shot of SROA
      PM.add(createSROAPass());
      PM.add(Translator.createTranslateDirectBranchesPass());
      PM.run(*TheModule);
    }

    // Obtain a new program counter to translate
    uint64_t NewPC = 0;
    std::tie(NewPC, Entry) = JumpTargets.peek();
    assert(Entry == nullptr || Entry->empty());

    VirtualAddress = NewPC;
    CodePointer = Code.data() + (NewPC - LoadAddress);
  } // End translations loop

  Function *CpuLoop = HelpersModule->getFunction("cpu_loop");
  assert(CpuLoop != nullptr);
  legacy::FunctionPassManager CpuLoopPM(TheModule.get());
  CpuLoopPM.add(new LoopInfoWrapperPass());
  CpuLoopPM.add(new CpuLoopFunctionPass());
  CpuLoopPM.run(*CpuLoop);

  // Force linking of cpu_loop
  TheModule->getOrInsertFunction("cpu_loop", CpuLoop->getFunctionType());

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
                                                     // ARM cpu_loop
                                                     "EmulateAll",
                                                     "cpu_abort",
                                                     "do_arm_semihosting");

  // EmulateAll: requires access to the opcode
  // do_arm_semihosting: we don't care about semihosting

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

  Linker TheLinker(TheModule.get());
  bool Result = TheLinker.linkInModule(HelpersModule.get(),
                                       Linker::LinkOnlyNeeded);
  assert(!Result && "Linking failed");

  legacy::PassManager PM;
  PM.add(createSROAPass());
  PM.add(new CpuLoopExitPass(&Variables));
  PM.add(Variables.createCorrectCPUStateUsagePass());
  PM.add(createDeadCodeEliminationPass());
  PM.run(*TheModule);

  // TODO: we have around all the usages of the PC, shall we drop them?
  Delimiter->eraseFromParent();

  JumpTargets.translateIndirectJumps();

  Translator.removeNewPCMarkers();

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
