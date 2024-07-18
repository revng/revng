/// \file CodeGenerator.cpp
/// This file handles the whole translation process from the input assembly to
/// LLVM IR.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstring>
#include <fstream>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/STLExtras.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/FunctionCallIdentification/PruneRetSuccessors.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/ProgramCounterHandler.h"

#include "qemu/libtcg/libtcg.h"

#include "CodeGenerator.h"
#include "ExternalJumpsHandler.h"
#include "InstructionTranslator.h"
#include "JumpTargetManager.h"
#include "VariableManager.h"

using namespace llvm;

using std::make_pair;
using std::string;

// Register all the arguments
static cl::opt<bool> RecordTCG("record-tcg",
                               cl::desc("create metadata for TCG"),
                               cl::cat(MainCategory));

static Logger<> LibTcgLog("libtcg");
static Logger<> Log("lift");

template<typename T, typename... ArgTypes>
inline std::array<T, sizeof...(ArgTypes)> make_array(ArgTypes &&...Args) {
  return { { std::forward<ArgTypes>(Args)... } };
}

/// Wrap a value around a temporary opaque function
///
/// Useful to prevent undesired optimizations
class OpaqueIdentity {
private:
  std::map<Type *, Function *> Map;
  Module *M;

public:
  OpaqueIdentity(Module *M) : M(M) {}

  ~OpaqueIdentity() { revng_assert(Map.size() == 0); }

  void drop() {
    SmallVector<CallInst *, 16> ToErase;
    for (auto [T, F] : Map) {
      for (User *U : F->users()) {
        auto *Call = cast<CallInst>(U);
        Call->replaceAllUsesWith(Call->getArgOperand(0));
        ToErase.push_back(Call);
      }
    }

    for (CallInst *Call : ToErase)
      eraseFromParent(Call);

    for (auto [T, F] : Map)
      eraseFromParent(F);

    Map.clear();
  }

  Instruction *wrap(IRBuilder<> &Builder, Value *V) {
    Type *ResultType = V->getType();
    Function *F = nullptr;
    auto It = Map.find(ResultType);
    if (It == Map.end()) {
      auto *FT = FunctionType::get(ResultType, { ResultType }, false);
      F = Function::Create(FT, GlobalValue::ExternalLinkage, "id", *M);
      F->setOnlyReadsMemory();
      Map[ResultType] = F;
    } else {
      F = It->second;
    }

    return Builder.CreateCall(F, { V });
  }

  Instruction *wrap(Instruction *I) {
    IRBuilder<> Builder(I->getParent(), ++I->getIterator());
    return wrap(Builder, I);
  }
};

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

static std::unique_ptr<Module> parseIR(StringRef Path, LLVMContext &Context) {
  std::unique_ptr<Module> Result;
  SMDiagnostic Errors;
  Result = parseIRFile(Path, Errors, Context);

  if (Result.get() == nullptr) {
    Errors.print("revng", dbgs());
    revng_abort();
  }

  return Result;
}

CodeGenerator::CodeGenerator(const RawBinaryView &RawBinary,
                             llvm::Module *TheModule,
                             const TupleTree<model::Binary> &Model,
                             std::string Helpers,
                             std::string EarlyLinked,
                             model::Architecture::Values TargetArchitecture) :
  RawBinary(RawBinary),
  TheModule(TheModule),
  Context(TheModule->getContext()),
  Model(Model),
  TargetArchitecture(TargetArchitecture) {

  OriginalInstrMDKind = Context.getMDKindID("oi");
  LibTcgInstrMDKind = Context.getMDKindID("pi");

  HelpersModule = parseIR(Helpers, Context);

  TheModule->setDataLayout(HelpersModule->getDataLayout());

  // Tag all global objects in HelpersModule as QEMU
  for (GlobalVariable &G : HelpersModule->globals())
    FunctionTags::QEMU.addTo(&G);

  for (Function &F : HelpersModule->functions()) {
    if (F.isIntrinsic())
      continue;

    F.setDSOLocal(false);

    FunctionTags::QEMU.addTo(&F);

    if (F.hasFnAttribute(Attribute::NoReturn)
        or F.getSection() == "revng_exceptional")
      FunctionTags::Exceptional.addTo(&F);
  }

  EarlyLinkedModule = parseIR(EarlyLinked, Context);
  for (llvm::Function &F : *EarlyLinkedModule) {
    if (F.isIntrinsic())
      continue;

    FunctionTags::QEMU.addTo(&F);
  }

  auto *Uint8Ty = Type::getInt8Ty(Context);
  auto *ElfHeaderHelper = new GlobalVariable(*TheModule,
                                             Uint8Ty,
                                             true,
                                             GlobalValue::ExternalLinkage,
                                             ConstantInt::get(Uint8Ty, 0),
                                             "elfheaderhelper");
  ElfHeaderHelper->setAlignment(MaybeAlign(1));
  ElfHeaderHelper->setSection(".elfheaderhelper");

  for (auto &[Segment, Data] : RawBinary.segments()) {
    // If it's executable register it as a valid code area
    if (Segment.IsExecutable()) {
      bool Found = false;
      MetaAddress End = Segment.pagesRange().second;
      revng_assert(End.isValid() and End.address() % 4096 == 0);
      for (const model::Segment &Segment : Model->Segments()) {
        if (Segment.IsExecutable() and Segment.contains(End)) {
          Found = true;
          break;
        }
      }

      // NOTE(anjo): I assume this is something we need to keep
      // as NoMoreCodeBoundaries gets populated.

      // The next page is not mapped
      if (not Found) {
        revng_check(Segment.endAddress().address() != 0);
        NoMoreCodeBoundaries.insert(Segment.endAddress());
      }
    }
  }
}

static BasicBlock *replaceFunction(Function *ToReplace) {
  // Save metadata
  SmallVector<std::pair<unsigned, MDNode *>, 4> SavedMetadata;
  ToReplace->getAllMetadata(SavedMetadata);

  ToReplace->setLinkage(GlobalValue::InternalLinkage);
  ToReplace->dropAllReferences();

  // Restore metadata
  for (auto [Kind, MD] : SavedMetadata)
    ToReplace->setMetadata(Kind, MD);

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
    revng_assert(Result == 0);
    ResultValue = nullptr;
  } else if (ToReplace->getReturnType()->isIntegerTy()) {
    auto *ReturnType = cast<IntegerType>(ToReplace->getReturnType());
    ResultValue = ConstantInt::get(ReturnType, Result, false);
  } else {
    revng_unreachable("No-op functions can only return void or an integer "
                      "type");
  }

  ReturnInst::Create(ToReplace->getParent()->getContext(), ResultValue, Body);
}

class CpuLoopFunctionPass : public llvm::ModulePass {
private:
  intptr_t ExceptionIndexOffset;

public:
  static char ID;

  CpuLoopFunctionPass() : llvm::ModulePass(ID), ExceptionIndexOffset(0) {}

  CpuLoopFunctionPass(intptr_t ExceptionIndexOffset) :
    llvm::ModulePass(ID), ExceptionIndexOffset(ExceptionIndexOffset) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};

char CpuLoopFunctionPass::ID = 0;

using RegisterCLF = RegisterPass<CpuLoopFunctionPass>;
static RegisterCLF Y("cpu-loop", "cpu_loop FunctionPass", false, false);

void CpuLoopFunctionPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
}

template<class Range, class UnaryPredicate>
auto findUnique(Range &&TheRange, UnaryPredicate Predicate)
  -> decltype(*TheRange.begin()) {

  const auto Begin = TheRange.begin();
  const auto End = TheRange.end();

  auto It = std::find_if(Begin, End, Predicate);
  auto Result = It;
  revng_assert(Result != End);
  revng_assert(std::find_if(++It, End, Predicate) == End);

  return *Result;
}

template<class Range>
auto findUnique(Range &&TheRange) -> decltype(*TheRange.begin()) {

  const auto Begin = TheRange.begin();
  const auto End = TheRange.end();

  auto Result = Begin;
  revng_assert(Begin != End && ++Result == End);

  return *Begin;
}

bool CpuLoopFunctionPass::runOnModule(Module &M) {
  Function &F = *M.getFunction("cpu_loop");

  // cpu_loop must return void
  revng_assert(F.getReturnType()->isVoidTy());

  Module *TheModule = F.getParent();

  // Part 1: remove the backedge of the main infinite loop
  const LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  const Loop *OutermostLoop = findUnique(LI);

  BasicBlock *Header = OutermostLoop->getHeader();

  // Check that the header has only one predecessor inside the loop
  auto IsInLoop = [&OutermostLoop](BasicBlock *Predecessor) {
    return OutermostLoop->contains(Predecessor);
  };
  BasicBlock *Footer = findUnique(predecessors(Header), IsInLoop);

  // Assert on the type of the last instruction (branch or brcond)
  revng_assert(Footer->end() != Footer->begin());
  Instruction *LastInstruction = &*--Footer->end();
  revng_assert(isa<BranchInst>(LastInstruction));

  // Remove the last instruction and replace it with a ret
  eraseFromParent(LastInstruction);
  ReturnInst::Create(F.getParent()->getContext(), Footer);

  // Part 2: replace the call to cpu_*_exec with exception_index
  Function &CpuExec = findUnique(F.getParent()->functions(), [](Function &F) {
    return F.getName() == "cpu_exec";
  });

  User *CallUser = findUnique(CpuExec.users(), [&F](User *U) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      return I->getParent()->getParent() == &F;
    } else {
      return false;
    }
  });

  auto *Call = cast<CallInst>(CallUser);
  revng_assert(Call->getCalledFunction() == &CpuExec);
  Value *CPUState = Call->getArgOperand(0);
  Type *TargetType = CpuExec.getReturnType();

  IRBuilder<> Builder(Call);
  Type *IntPtrTy = Builder.getIntPtrTy(TheModule->getDataLayout());
  Value *CPUIntPtr = Builder.CreatePtrToInt(CPUState, IntPtrTy);
  using CI = ConstantInt;
  auto Offset = CI::get(IntPtrTy, ExceptionIndexOffset);
  Value *ExceptionIndexIntPtr = Builder.CreateAdd(CPUIntPtr, Offset);
  Value *ExceptionIndexPtr = Builder.CreateIntToPtr(ExceptionIndexIntPtr,
                                                    TargetType->getPointerTo());
  Value *ExceptionIndex = Builder.CreateLoad(TargetType, ExceptionIndexPtr);
  Call->replaceAllUsesWith(ExceptionIndex);
  eraseFromParent(Call);

  return true;
}

class CpuLoopExitPass : public llvm::ModulePass {
public:
  static char ID;

  CpuLoopExitPass() : llvm::ModulePass(ID), VM(nullptr) {}
  CpuLoopExitPass(VariableManager *VM) : llvm::ModulePass(ID), VM(VM) {}

  bool runOnModule(llvm::Module &M) override;

private:
  VariableManager *VM;
};

char CpuLoopExitPass::ID = 0;

using RegisterCLE = RegisterPass<CpuLoopExitPass>;
static RegisterCLE Z("cpu-loop-exit", "cpu_loop_exit Pass", false, false);

static void purgeNoReturn(Function *F) {
  auto &Context = F->getParent()->getContext();

  if (F->hasFnAttribute(Attribute::NoReturn))
    F->removeFnAttr(Attribute::NoReturn);

  for (User *U : F->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      if (Call->hasFnAttr(Attribute::NoReturn)) {
        auto OldAttr = Call->getAttributes();
        auto NewAttr = OldAttr.removeFnAttribute(Context, Attribute::NoReturn);
        Call->setAttributes(NewAttr);
      }
    }
  }
}

static ReturnInst *createRet(Instruction *Position) {
  Function *F = Position->getParent()->getParent();
  purgeNoReturn(F);

  Type *ReturnType = F->getFunctionType()->getReturnType();
  if (ReturnType->isVoidTy()) {
    return ReturnInst::Create(F->getParent()->getContext(), nullptr, Position);
  } else if (ReturnType->isIntegerTy() or ReturnType->isPointerTy()) {
    auto *Null = Constant::getNullValue(ReturnType);
    return ReturnInst::Create(F->getParent()->getContext(), Null, Position);
  } else {
    revng_abort("Return type not supported");
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
bool CpuLoopExitPass::runOnModule(llvm::Module &M) {
  LLVMContext &Context = M.getContext();
  Function *CpuLoopExitRestore = M.getFunction("cpu_loop_exit_restore");


  // Replace uses of cpu_loop_exit_restore with cpu_loop_exit, some targets
  // e.g. mips only use cpu_loop_exit_restore
  if (CpuLoopExitRestore != nullptr) {
    Function *CpuLoopExit = cast<Function>(M.getOrInsertFunction("cpu_loop_exit", Type::getVoidTy(Context), CpuLoopExitRestore->getArg(0)->getType()).getCallee());
    SmallVector<Value *, 8> ToErase;
    for (User *U : CpuLoopExitRestore->users()) {
      auto *Call = cast<CallInst>(U);
      Value *CpuStateArg = Call->getArgOperand(0);
      IRBuilder<> Builder(Call);
      Value *NewCall = Builder.CreateCall(CpuLoopExit, {CpuStateArg});
      Call->replaceAllUsesWith(NewCall);
      ToErase.push_back(Call);
    }

    for (auto &V : ToErase) {
      eraseFromParent(V);
    }
  }

  Function *CpuLoopExit = M.getFunction("cpu_loop_exit");
  // Nothing to do here
  if (CpuLoopExit == nullptr)
    return false;

  revng_assert(VM->hasEnv());

  purgeNoReturn(CpuLoopExit);

  IntegerType *BoolType = Type::getInt1Ty(Context);
  std::set<Function *> FixedCallers;
  GlobalVariable *CpuLoopExitingVariable = nullptr;
  CpuLoopExitingVariable = new GlobalVariable(M,
                                              BoolType,
                                              false,
                                              GlobalValue::CommonLinkage,
                                              ConstantInt::getFalse(BoolType),
                                              StringRef("cpu_loop_exiting"));

  Function *CpuLoop = M.getFunction("cpu_loop");
  revng_assert(CpuLoop != nullptr);

  for (User *U : CpuLoopExit->users()) {
    auto *Call = cast<CallInst>(U);
    revng_assert(Call->getCalledFunction() == CpuLoopExit);

    // Call cpu_loop
    auto *EnvPtr = VM->cpuStateToEnv(Call->getArgOperand(0), Call);

    auto *CallCpuLoop = CallInst::Create(CpuLoop, { EnvPtr }, "", Call);

    // In recent versions of LLVM you can no longer inject a CallInst in a
    // Function with debug location if the call itself has not a debug location
    // as well, otherwise module verification will fail
    CallCpuLoop->setDebugLoc(Call->getDebugLoc());

    // Set cpu_loop_exiting to true
    auto *Store [[maybe_unused]] = new StoreInst(ConstantInt::getTrue(BoolType),
                                                 CpuLoopExitingVariable, Call);

    // Return immediately
    createRet(Call);
    auto *Unreach = cast<UnreachableInst>(&*(++Call->getIterator()));
    eraseFromParent(Unreach);

    Function *Caller = Call->getParent()->getParent();

    // Remove the call to cpu_loop_exit
    eraseFromParent(Call);

    if (FixedCallers.contains(Caller)) {
      continue;
    }

    FixedCallers.insert(Caller);

    std::queue<Value *> WorkList;
    WorkList.push(Caller);

    while (!WorkList.empty()) {
      Value *F = WorkList.front();
      WorkList.pop();

      for (User *RecUser : F->users()) {
        auto *RecCall = dyn_cast<CallInst>(RecUser);
        if (RecCall == nullptr) {
          if (auto *Cast = dyn_cast<ConstantExpr>(RecUser)) {
            revng_assert(Cast->getOperand(0) == F && Cast->isCast());
            WorkList.push(Cast);
            continue;
          } else if (auto *Store = dyn_cast<StoreInst>(RecUser)) {
            // TODO(anjo): We encountered a store, what do?
            //WorkList.push(Store);
            continue;
          } else if (auto *Store = dyn_cast<Constant>(RecUser)) {
            continue;
          } else {
            revng_assert(false, "Unexpected user");
          }
        }

        Function *RecCaller = RecCall->getParent()->getParent();

        // TODO: make this more reliable than using function name
        // If the caller is a QEMU helper function make it check
        // cpu_loop_exiting and if it's true, make it return

        // Split BB
        BasicBlock *OldBB = RecCall->getParent();
        BasicBlock::iterator SplitPoint = ++RecCall->getIterator();
        revng_assert(SplitPoint != OldBB->end());
        BasicBlock *NewBB = OldBB->splitBasicBlock(SplitPoint);

        // Add a BB with a ret
        BasicBlock *QuitBB = BasicBlock::Create(Context,
                                                "cpu_loop_exit_return",
                                                RecCaller,
                                                NewBB);
        UnreachableInst *Temp = new UnreachableInst(Context, QuitBB);
        // TODO(anjo): Sometimes we crash here but not always
        createRet(Temp);
        eraseFromParent(Temp);

        // Check value of cpu_loop_exiting
        auto *Branch = cast<BranchInst>(&*++(RecCall->getIterator()));
        auto *PointeeTy = CpuLoopExitingVariable->getValueType();
        auto *Compare = new ICmpInst(Branch,
                                     CmpInst::ICMP_EQ,
                                     new LoadInst(PointeeTy,
                                                  CpuLoopExitingVariable,
                                                  "",
                                                  Branch),
                                     ConstantInt::getTrue(BoolType));

        BranchInst::Create(QuitBB, NewBB, Compare, Branch);
        eraseFromParent(Branch);

        // Add to the work list only if it hasn't been fixed already
        if (!FixedCallers.contains(RecCaller)) {
          FixedCallers.insert(RecCaller);
          WorkList.push(RecCaller);
        }
      }
    }
  }

  return true;
}

void CodeGenerator::translate(const LibTcgInterface &LibTcg,
                              std::optional<uint64_t> RawVirtualAddress) {
  using FT = FunctionType;

  Task T(12, "Translation");

  // Declare the abort function
  auto *AbortTy = FunctionType::get(Type::getVoidTy(Context), false);
  FunctionCallee AbortFunction = TheModule->getOrInsertFunction("abort",
                                                                AbortTy);
  {
    auto *Abort = cast<Function>(skipCasts(AbortFunction.getCallee()));
    FunctionTags::Exceptional.addTo(Abort);
  }

  // tcg_allowed is a global in qemu defined in accel/tcg/... indicating
  // wheter or not TCG is used as a backend or not. Some helpers/code may
  // branch on this global. As we do not include accel/tcg/... code in the
  // helpers module we get an undefined reference during linking.
  //
  // Here we set the constant to 1 to avoid undefined refnerences but also
  // ensure all code correctly assumes we are using TCG.
  if (auto *TcgAllowed = HelpersModule->getGlobalVariable("tcg_allowed")) {
      auto *Uint8Ty = Type::getInt8Ty(Context);
      TcgAllowed->setInitializer(ConstantInt::get(Uint8Ty, 1));
  }

  // Prepare the helper modules by transforming the cpu_loop function and
  // running SROA
  T.advance("Prepare helpers module", true);
  legacy::PassManager CpuLoopPM;
  CpuLoopPM.add(new LoopInfoWrapperPass());
  CpuLoopPM.add(new CpuLoopFunctionPass(LibTcg.exception_index));
  CpuLoopPM.add(createSROAPass());
  CpuLoopPM.run(*HelpersModule);

  // Drop the main
  eraseFromParent(HelpersModule->getFunction("main"));

  // From syscall.c
  new GlobalVariable(*TheModule,
                     Type::getInt32Ty(Context),
                     false,
                     GlobalValue::CommonLinkage,
                     ConstantInt::get(Type::getInt32Ty(Context), 0),
                     StringRef("do_strace"));

  //
  // Handle some specific QEMU functions as no-ops or abort
  //

  // Transform in no op
  auto NoOpFunctionNames = make_array<const char *>("cpu_dump_state",
                                                    "cpu_exit",
                                                    "end_exclusive"
                                                    "fprintf",
                                                    "mmap_lock",
                                                    "mmap_unlock",
                                                    "pthread_cond_broadcast",
                                                    "pthread_mutex_unlock",
                                                    "pthread_mutex_lock",
                                                    "pthread_cond_wait",
                                                    "pthread_cond_signal",
                                                    "process_pending_signals",
                                                    "qemu_log_mask",
                                                    "qemu_thread_atexit_init",
                                                    "qemu_log",
                                                    "qemu_loglevel_mask",
                                                    "qemu_thread_atexit_init",
                                                    "_nocheck__trace_user_do_sigreturn",
                                                    "_nocheck__trace_user_do_rt_sigreturn",
                                                    "start_exclusive");
  for (auto Name : NoOpFunctionNames)
    replaceFunctionWithRet(HelpersModule->getFunction(Name), 0);

  // Transform in abort

  // do_arm_semihosting: we don't care about semihosting
  // EmulateAll: requires access to the opcode
  auto AbortFunctionNames = make_array<const char *>("cpu_restore_state",
                                                     "cpu_mips_exec",
                                                     "gdb_handlesig",
                                                     "queue_signal",
                                                     // syscall.c
                                                     "do_ioctl_dm",
                                                     "print_syscall",
                                                     "print_syscall_ret",
                                                     "safe_syscall_base",
                                                     // ARM cpu_loop
                                                     "cpu_abort",
                                                     "do_arm_semihosting",
                                                     "EmulateAll");
  for (auto Name : AbortFunctionNames) {
    Function *TheFunction = HelpersModule->getFunction(Name);
    if (TheFunction != nullptr) {
      auto *Abort = HelpersModule->getFunction("abort");
      revng_assert(Abort != nullptr);
      BasicBlock *NewBody = replaceFunction(TheFunction);
      CallInst::Create(Abort, {}, NewBody);
      new UnreachableInst(Context, NewBody);
    }
  }

  replaceFunctionWithRet(HelpersModule->getFunction("page_check_range"), 1);
  replaceFunctionWithRet(HelpersModule->getFunction("page_get_flags"),
                         0xffffffff);

  //
  // Record globals for marking them as internal after linking
  //
  std::vector<std::string> HelperGlobals;
  for (GlobalVariable &GV : HelpersModule->globals())
    if (GV.hasName())
      HelperGlobals.push_back(GV.getName().str());

  std::vector<std::string> HelperFunctions;
  for (Function &F : HelpersModule->functions())
    if (F.hasName() and F.getName() != "target_set_brk"
        and F.getName() != "syscall_init")
      HelperFunctions.push_back(F.getName().str());

  //
  // Link helpers module into the main module
  //
  T.advance("Linking helpers module", true);
  Linker TheLinker(*TheModule);
  bool Result = TheLinker.linkInModule(std::move(HelpersModule));
  revng_assert(not Result, "Linking failed");

  //
  // Mark as internal all the imported globals
  //
  for (StringRef GlobalName : HelperGlobals)
    if (not GlobalName.startswith("llvm."))
      if (auto *GV = TheModule->getGlobalVariable(GlobalName))
        if (not GV->isDeclaration())
          GV->setLinkage(GlobalValue::InternalLinkage);

  for (StringRef FunctionName : HelperFunctions)
    if (auto *F = TheModule->getFunction(FunctionName))
      if (not F->isDeclaration() and not F->isIntrinsic())
        F->setLinkage(GlobalValue::InternalLinkage);

  //
  // Create the libtcg context
  //
  // TODO(anjo): Move to Lift.cpp?
  LibTcgDesc Desc = {};
  auto *LibTcgContext = LibTcg.context_create(&Desc);
  revng_assert(LibTcgContext != nullptr, "Failed to create libtcg context");

  //
  // Create the VariableManager
  //
  bool TargetIsLittleEndian;
  {
    using namespace model::Architecture;
    TargetIsLittleEndian = isLittleEndian(TargetArchitecture);
  }

  // TODO: this not very robust. We should have a function with a sensible name
  //       taking as argument ${ARCH}CPU so that we can easily identify the
  //       struct.
  auto *CPUStruct =
      StructType::getTypeByName(TheModule->getContext(), "struct.ArchCPU");
  revng_assert(CPUStruct != nullptr);
  VariableManager Variables(*TheModule, TargetIsLittleEndian, CPUStruct,
                            LibTcg.env_offset, LibTcg.env_ptr(LibTcgContext));
  auto CreateCPUStateAccessAnalysisPass = [&Variables]() {
    return new CPUStateAccessAnalysisPass(&Variables, true);
  };

  {
    legacy::PassManager PM;
    PM.add(new CpuLoopExitPass(&Variables));
    PM.run(*TheModule);
  }

  std::set<Function *> CpuLoopExitingUsers;
  GlobalVariable *CpuLoopExiting =
      TheModule->getGlobalVariable("cpu_loop_exiting");
  revng_assert(CpuLoopExiting != nullptr);
  for (User *U : CpuLoopExiting->users())
    if (auto *I = dyn_cast<Instruction>(U))
      CpuLoopExitingUsers.insert(I->getParent()->getParent());

  //
  // Create well-known CSVs
  //
  auto SP = model::Architecture::getStackPointer(Model->Architecture());
  std::string SPName = model::Register::getCSVName(SP).str();
  GlobalVariable *SPReg = Variables.getByEnvOffset(LibTcg.sp, SPName).first;

  using PCHOwner = std::unique_ptr<ProgramCounterHandler>;
  auto Factory = [&Variables, &LibTcg](PCAffectingCSV::Values CSVID,
                                       llvm::StringRef Name) -> GlobalVariable * {
    intptr_t Offset = 0;

    switch (CSVID) {
    case PCAffectingCSV::PC:
      Offset = LibTcg.pc;
      break;

    case PCAffectingCSV::IsThumb:
      Offset = LibTcg.is_thumb;
      break;

    default:
      revng_abort();
    }

    return Variables.getByEnvOffset(Offset, Name.str()).first;
  };

  auto Architecture = toLLVMArchitecture(Model->Architecture());
  PCHOwner PCH = ProgramCounterHandler::create(Architecture,
                                               TheModule,
                                               Factory);

  IRBuilder<> Builder(Context);

  // Create main function
  auto *MainType = FT::get(Builder.getVoidTy(),
                           { SPReg->getValueType() },
                           false);
  auto *MainFunction = Function::Create(MainType,
                                        Function::ExternalLinkage,
                                        "root",
                                        TheModule);
  FunctionTags::Root.addTo(MainFunction);

  // Create the first basic block and create a placeholder for variable
  // allocations
  BasicBlock *Entry = BasicBlock::Create(Context, "entrypoint", MainFunction);
  Builder.SetInsertPoint(Entry);

  // We need to remember this instruction so we can later insert a call here.
  // The problem is that up until now we don't know where our CPUState structure
  // is.
  // After the translation we will and use this information to create a call to
  // a helper function.
  // TODO: we need a more elegant solution here
  auto *Delimiter = Builder.CreateStore(&*MainFunction->arg_begin(), SPReg);
  Variables.setAllocaInsertPoint(Delimiter);
  auto *InitEnvInsertPoint = Delimiter;

  QuickMetadata QMD(Context);

  // Link early-linked.c
  T.advance("Link early-linked.c", true);
  {
    Linker TheLinker(*TheModule);
    bool Result = TheLinker.linkInModule(std::move(EarlyLinkedModule),
                                         Linker::None);
    revng_assert(!Result, "Linking failed");
  }

  // Create an instance of JumpTargetManager
  JumpTargetManager JumpTargets(MainFunction,
                                PCH.get(),
                                CreateCPUStateAccessAnalysisPass,
                                Model,
                                RawBinary);

  MetaAddress VirtualAddress = MetaAddress::invalid();
  if (RawVirtualAddress) {
    VirtualAddress = JumpTargets.fromPC(*RawVirtualAddress);
  } else {
    JumpTargets.harvestGlobalData();
    VirtualAddress = Model->EntryPoint();
  }

  if (VirtualAddress.isValid()) {
    revng_assert(VirtualAddress.isCode());
    JumpTargets.registerJT(VirtualAddress, JTReason::GlobalData);

    // Initialize the program counter
    PCH->initializePC(Builder, VirtualAddress);
  }

  OpaqueIdentity OI(TheModule);

  // Fake jumps to the dispatcher-related basic blocks. This way all the blocks
  // are always reachable.
  auto *ReachSwitch = Builder.CreateSwitch(OI.wrap(Builder, Builder.getInt8(0)),
                                           JumpTargets.dispatcher());
  ReachSwitch->addCase(Builder.getInt8(1), JumpTargets.anyPC());
  ReachSwitch->addCase(Builder.getInt8(2), JumpTargets.unexpectedPC());

  JumpTargets.setCFGForm(CFGForm::SemanticPreserving);

  std::vector<BasicBlock *> Blocks;

  bool EndianessMismatch;
  {
    using namespace model::Architecture;
    bool SourceIsLittleEndian = isLittleEndian(Model->Architecture());
    EndianessMismatch = TargetIsLittleEndian != SourceIsLittleEndian;
  }

  T.advance("Lifting code", true);
  Task LiftTask({}, "Lifting");
  LiftTask.advance("Initial address peeking", false);
  InstructionTranslator Translator(LibTcg, Builder, Variables, JumpTargets,
                                   Blocks, EndianessMismatch, PCH.get());

  std::tie(VirtualAddress, Entry) = JumpTargets.peek();

  auto MaybeData = RawBinary.getFromAddressOn(VirtualAddress);
  revng_assert(MaybeData);
  llvm::ArrayRef<uint8_t> CodeBuffer = *MaybeData;
  MetaAddress CodeBufferStartAddress = VirtualAddress;

  while (Entry != nullptr) {
    size_t Offset = VirtualAddress.address()
                    - CodeBufferStartAddress.address();

    // Make sure VirtualAddress points to an executable segment, otherwise
    // look up a new one.
    if (Offset >= CodeBuffer.size()) {
      auto MaybeData = RawBinary.getFromAddressOn(VirtualAddress);
      revng_assert(MaybeData);
      CodeBuffer = *MaybeData;
      CodeBufferStartAddress = VirtualAddress;
    }

    LiftTask.advance(VirtualAddress.toString(), true);

    Task TranslateTask(3, "Translate");
    TranslateTask.advance("Lift to PTC", true);

    Builder.SetInsertPoint(Entry);

    // TODO: what if create a new instance of an InstructionTranslator here?
    Translator.reset();

    uint32_t TranslateFlags = 0;
    if (VirtualAddress.type() ==  MetaAddressType::Code_arm_thumb) {
      TranslateFlags |= LIBTCG_TRANSLATE_ARM_THUMB;
    }

    auto NewInstructionList =
      LibTcg.translate(LibTcgContext,
                       CodeBuffer.data() + Offset,
                       CodeBuffer.size() - Offset,
                       VirtualAddress.address(),
                       TranslateFlags);

    // TODO: rename this type
    const size_t ConsumedSize = NewInstructionList.size_in_bytes;
    revng_assert(ConsumedSize > 0);

    SmallSet<unsigned, 1> ToIgnore;
    // Handles writes to btarget, represents branching for microblaze/mips/cris
    ToIgnore = Translator.preprocess(NewInstructionList);

    if (LibTcgLog.isEnabled()) {
      static std::array<char, 128> DumpBuf{0};
      LibTcgLog << "[Translation starting from "
                << std::hex << VirtualAddress.address()
                << "]" << DoLog;
      for (size_t I = 0; I < NewInstructionList.instruction_count; ++I) {
        LibTcg.dump_instruction_to_buffer(&NewInstructionList.list[I],
                                          DumpBuf.data(),
                                          DumpBuf.size());
        LibTcgLog << DumpBuf.data() << DoLog;
      }
    }

    Variables.newTranslationBlock(&NewInstructionList);
    MDNode *MDOriginalInstr = nullptr;
    bool StopTranslation = false;

    MetaAddress PC = VirtualAddress;
    MetaAddress NextPC = MetaAddress::invalid();
    MetaAddress EndPC = VirtualAddress + ConsumedSize;

    const auto InstructionCount = NewInstructionList.instruction_count;
    using IT = InstructionTranslator;
    IT::TranslationResult Result;

    unsigned J = 0;

    // Handle the first LIBTCG_op_insn_start
    {
      LibTcgInstruction *NextInstruction = nullptr;
      for (unsigned K = 1; K < InstructionCount; K++) {
        LibTcgInstruction *I = &NewInstructionList.list[K];
        if (I->opcode == LIBTCG_op_insn_start && ToIgnore.count(K) == 0) {
          NextInstruction = I;
          break;
        }
      }
      LibTcgInstruction *Instruction = &NewInstructionList.list[J];
      std::tie(Result, MDOriginalInstr, PC, NextPC) =
        Translator.newInstruction(Instruction, NextInstruction,
                                  VirtualAddress, EndPC, true);
      J++;
    }

    // TODO: shall we move this whole loop in InstructionTranslator?
    for (; J < InstructionCount && !StopTranslation; J++) {
      if (ToIgnore.count(J) != 0)
        continue;

      LibTcgInstruction *Instruction = &NewInstructionList.list[J];
      auto Opcode = Instruction->opcode;

      Blocks.clear();
      Blocks.push_back(Builder.GetInsertBlock());

      switch (Opcode) {
      case LIBTCG_op_discard:
        // Instructions we don't even consider
        break;
      case LIBTCG_op_insn_start: {
        // Find next instruction, if there is one
        LibTcgInstruction *NextInstruction = nullptr;
        for (unsigned K = J + 1; K < InstructionCount; K++) {
          LibTcgInstruction *I = &NewInstructionList.list[K];
          if (I->opcode == LIBTCG_op_insn_start && ToIgnore.count(K) == 0) {
            NextInstruction = I;
            break;
          }
        }

        std::tie(Result, MDOriginalInstr, PC, NextPC) =
          Translator.newInstruction(Instruction, NextInstruction,
                                    VirtualAddress, EndPC, false);
      } break;
      case LIBTCG_op_call: {
        // TODO(anjo): Move to default
        Result = Translator.translateCall(Instruction);
      } break;
      default:
        Result = Translator.translate(Instruction, PC, NextPC);
        break;
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
      }

      // Create a new metadata referencing the TCG instruction we have just
      // translated
      MDNode *MDLibTcgInstr = nullptr;
      if (RecordTCG) {
        static std::array<char, 128> DumpBuf{0};
        LibTcg.dump_instruction_to_buffer(&NewInstructionList.list[J],
                                          DumpBuf.data(), DumpBuf.size());

        // Eh not very nice to strlen in construction of the StringRef,
        // maybe we can get the length from the LibTcg call above?
        StringRef Str{DumpBuf.data()};
        MDString *MDLibTcgString = MDString::get(Context, Str);
        MDLibTcgInstr = MDNode::getDistinct(Context, MDLibTcgString);
      }

      // Set metadata for all the new instructions
      for (BasicBlock *Block : Blocks) {
        BasicBlock::iterator I = Block->end();
        while (I != Block->begin() && !(--I)->hasMetadata()) {
          if (MDOriginalInstr != nullptr)
            I->setMetadata(OriginalInstrMDKind, MDOriginalInstr);
          if (MDLibTcgInstr != nullptr)
            I->setMetadata(LibTcgInstrMDKind, MDLibTcgInstr);
        }
      }

    } // End loop over instructions

    TranslateTask.complete();
    TranslateTask.advance("Finalization", true);
    LibTcg.instruction_list_destroy(LibTcgContext, NewInstructionList);

    // We might have a leftover block, probably due to the block created after
    // the last call to exit_tb
    auto *LastBlock = Builder.GetInsertBlock();
    //errs() << *LastBlock << "\n";
    if (LastBlock->empty()) {
      eraseFromParent(LastBlock);
    } else if (!LastBlock->rbegin()->isTerminator()) {
      // Something went wrong, probably a mistranslation
      errs() << *LastBlock->rbegin() << "\n";
      errs() << "MISTRANSLATION??\n";
      Builder.CreateUnreachable();
    }

    Translator.registerDirectJumps();
    // Obtain a new program counter to translate
    TranslateTask.complete();
    LiftTask.advance("Peek new address", true);
    std::tie(VirtualAddress, Entry) = JumpTargets.peek();
  } // End translations loop

  LiftTask.complete();

  // TODO(anjo): Destroy the libtcg context, note this
  // might be nice to move out of this translate function.
  LibTcg.context_destroy(LibTcgContext);

  OI.drop();

  // Reorder basic blocks in RPOT
  T.advance("Reordering basic blocks", true);
  {
    BasicBlock *Entry = &MainFunction->getEntryBlock();
    ReversePostOrderTraversal<BasicBlock *> RPOT(Entry);
    std::set<BasicBlock *> SortedBasicBlocksSet;
    std::vector<BasicBlock *> SortedBasicBlocks;
    for (BasicBlock *BB : RPOT) {
      SortedBasicBlocksSet.insert(BB);
      SortedBasicBlocks.push_back(BB);
    }

    std::vector<BasicBlock *> Unreachable;
    for (BasicBlock &BB : *MainFunction)
      if (!SortedBasicBlocksSet.contains(&BB))
        Unreachable.push_back(&BB);

    auto Size = MainFunction->size();
    for (unsigned I = 0; I < Size; ++I)
      MainFunction->begin()->removeFromParent();
    for (BasicBlock *BB : SortedBasicBlocks)
      MainFunction->insert(MainFunction->end(), BB);
    for (BasicBlock *BB : Unreachable)
      MainFunction->insert(MainFunction->end(), BB);
  }

  //
  // At this point we have all the code, add store false to cpu_loop_exiting in
  // root
  //
  T.advance("IR finalization", true);
  auto *BoolType = CpuLoopExiting->getValueType();
  std::queue<User *> WorkList;
  for (Function *Helper : CpuLoopExitingUsers)
    for (User *U : Helper->users())
      WorkList.push(U);

  while (not WorkList.empty()) {
    User *U = WorkList.front();
    WorkList.pop();

    if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->isCast())
        for (User *UCE : CE->users())
          WorkList.push(UCE);
    } else if (auto *Call = dyn_cast<CallInst>(U)) {
      if (Call->getParent()->getParent() == MainFunction) {
        new StoreInst(ConstantInt::getFalse(BoolType),
                      CpuLoopExiting,
                      Call->getNextNode());
      }
    }
  }

  // Add a call to the function to initialize the CPUState, if present.
  // This is important on x86 architecture.
  // We only add the call after the Linker has imported the
  // initialize_env function from the helpers, because the declaration
  // imported before with importHelperFunctionDeclaration() only has
  // stub types and injecting the CallInst earlier would break
  if (Function *InitEnv = TheModule->getFunction("initialize_env")) {
    revng_assert(not InitEnv->getFunctionType()->isVarArg());
    revng_assert(InitEnv->getFunctionType()->getNumParams() == 1);
    auto *CPUStateType = InitEnv->getFunctionType()->getParamType(0);
    Instruction *InsertBefore = InitEnvInsertPoint;
    auto *AddressComputation = Variables.computeEnvAddress(CPUStateType,
                                                           InsertBefore);
    CallInst::Create(InitEnv, { AddressComputation }, "", InsertBefore);
  }

  Variables.setDataLayout(&TheModule->getDataLayout());

  T.advance("Finalize newpc markers", true);
  Translator.finalizeNewPCMarkers();

  T.advance("Optimize lifted IR");
  // SROA must run before InstCombine because in this way InstCombine has many
  // more elementary operations to combine
  legacy::PassManager PreInstCombinePM;
  PreInstCombinePM.add(createSROAPass());
  PreInstCombinePM.run(*TheModule);

  // InstCombine must run before CPUStateAccessAnalysis (CSAA) because, if it
  // runs after it, it removes all the useful metadata attached by CSAA.
  legacy::FunctionPassManager InstCombinePM(&*TheModule);
  InstCombinePM.add(createInstructionCombiningPass());
  InstCombinePM.doInitialization();
  InstCombinePM.run(*MainFunction);
  InstCombinePM.doFinalization();

  legacy::PassManager PostInstCombinePM;
  PostInstCombinePM.add(new LoadModelWrapperPass(Model));
  PostInstCombinePM.add(new CPUStateAccessAnalysisPass(&Variables, false));
  PostInstCombinePM.add(createDeadCodeEliminationPass());
  PostInstCombinePM.add(new PruneRetSuccessors);
  PostInstCombinePM.run(*TheModule);

  T.advance("Finalize jump targets", true);
  JumpTargets.finalizeJumpTargets();

  T.advance("Purge dead code", true);
  EliminateUnreachableBlocks(*MainFunction, nullptr, false);

  T.advance("Create revng.jt.reason", true);
  JumpTargets.createJTReasonMD();

  T.advance("Finalization", true);
  ExternalJumpsHandler JumpOutHandler(*Model,
                                      JumpTargets.dispatcher(),
                                      *MainFunction,
                                      PCH.get());
  JumpOutHandler.createExternalJumpsHandler();

  Variables.finalize();
}
