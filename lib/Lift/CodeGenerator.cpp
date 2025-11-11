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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/STLExtras.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/FunctionCallIdentification/PruneRetSuccessors.h"
#include "revng/Lift/VariableManager.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/ProgramCounterHandler.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "CodeGenerator.h"
#include "ExternalJumpsHandler.h"
#include "InstructionTranslator.h"
#include "JumpTargetManager.h"

RegisterIRHelper CPULoopExitHelper("cpu_loop_exit");
RegisterIRHelper InitializeEnvHelper("helper_initialize_env");

using namespace llvm;

using std::make_pair;
using std::string;

// Register all the arguments
static cl::opt<bool> RecordTCG("record-tcg",
                               cl::desc("create metadata for TCG"),
                               cl::cat(MainCategory));

static Logger LibTcgLog("libtcg");
static Logger Log("lift");

template<typename T, typename... ArgTypes>
constexpr std::array<T, sizeof...(ArgTypes)> make_array(ArgTypes &&...Args) {
  return { { std::forward<ArgTypes>(Args)... } };
}

/// Wrap a value around a temporary opaque function
///
/// Useful to prevent undesired optimizations
class OpaqueIdentity {
private:
  std::map<Type *, Function *> Map;
  Module *M = nullptr;

public:
  OpaqueIdentity(Module *M) : M(M) {}

  ~OpaqueIdentity() { revng_assert(Map.size() == 0); }

  void drop() {
    SmallVector<CallInst *, 16> ToErase;
    for (auto &&[T, F] : Map) {
      for (User *U : F->users()) {
        auto *Call = cast<CallInst>(U);
        Call->replaceAllUsesWith(Call->getArgOperand(0));
        ToErase.push_back(Call);
      }
    }

    for (CallInst *Call : ToErase)
      eraseFromParent(Call);

    for (auto &&[T, F] : Map)
      eraseFromParent(F);

    Map.clear();
  }

  Instruction *wrap(revng::IRBuilder &Builder, Value *V) {
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
    revng::NonDebugInfoCheckingIRBuilder Builder(I->getParent(),
                                                 ++I->getIterator());
    return wrap(Builder, I);
  }
};

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

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

  LibTcgInstrMDKind = Context.getMDKindID("pi");

  HelpersModule = parseIR(Context, Helpers);
  revng_assert(HelpersModule->getGlobalVariable("cpu_loop_exiting") != nullptr);

  legacy::PassManager OptimizingPM;
  OptimizingPM.add(createSROAPass());
  OptimizingPM.add(createInstSimplifyLegacyPass());
  OptimizingPM.run(*HelpersModule);

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

  EarlyLinkedModule = parseIR(Context, EarlyLinked);
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

      // The next page is not mapped
      if (not Found) {
        revng_check(Segment.endAddress().address() != 0);
        NoMoreCodeBoundaries.insert(Segment.endAddress());
      }
    }
  }
}

void CodeGenerator::translate(LibTcg &LibTcg,
                              std::optional<uint64_t> RawVirtualAddress) {
  using FT = FunctionType;

  Task T(12, "Translation");

  //
  // Link helpers module into the main module
  //
  T.advance("Linking helpers module", true);
  Linker TheLinker(*TheModule);
  bool Result = TheLinker.linkInModule(std::move(HelpersModule));
  revng_assert(not Result, "Linking failed");

  //
  // Create the VariableManager
  //
  bool TargetIsLittleEndian;
  {
    using namespace model::Architecture;
    TargetIsLittleEndian = isLittleEndian(TargetArchitecture);
  }

  const auto &ArchInfo = LibTcg.archInfo();
  VariableManager Variables(*TheModule,
                            TargetIsLittleEndian,
                            ArchInfo.env_offset,
                            LibTcg.envPointer(),
                            LibTcg.globalNames());
  //
  // Create well-known CSVs
  //
  auto SP = model::Architecture::getStackPointer(Model->Architecture());
  std::string SPName = model::Register::getCSVName(SP);
  GlobalVariable *SPReg = Variables.getByEnvOffset(ArchInfo.sp).first;

  using PCHOwner = std::unique_ptr<ProgramCounterHandler>;
  auto Factory = [&Variables,
                  &ArchInfo](PCAffectingCSV::Values CSVID) -> GlobalVariable * {
    intptr_t Offset = 0;

    switch (CSVID) {
    case PCAffectingCSV::PC:
      Offset = ArchInfo.pc;
      break;

    case PCAffectingCSV::IsThumb:
      Offset = ArchInfo.is_thumb;
      break;

    default:
      revng_abort();
    }

    return Variables.getByEnvOffset(Offset).first;
  };

  PCHOwner PCH = ProgramCounterHandler::create(Model->Architecture(),
                                               TheModule,
                                               Factory);

  revng::NonDebugInfoCheckingIRBuilder Builder(Context);

  // Create main function
  auto *MainType = FT::get(Builder.getVoidTy(),
                           { SPReg->getValueType() },
                           false);
  auto *RootFunction = Function::Create(MainType,
                                        Function::ExternalLinkage,
                                        "root",
                                        TheModule);
  FunctionTags::Root.addTo(RootFunction);
  RootFunction->addFnAttr(Attribute::NullPointerIsValid);

  // Create the first basic block and create a placeholder for variable
  // allocations
  BasicBlock *Entry = BasicBlock::Create(Context, "entrypoint", RootFunction);
  Builder.SetInsertPoint(Entry);

  // We need to remember this instruction so we can later insert a call here.
  // The problem is that up until now we don't know where our CPUState structure
  // is.
  // After the translation we will and use this information to create a call to
  // a helper function.
  // TODO: we need a more elegant solution here
  auto *Delimiter = Builder.CreateStore(&*RootFunction->arg_begin(), SPReg);
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
  JumpTargetManager JumpTargets(RootFunction, PCH.get(), Model, RawBinary);

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
  InstructionTranslator Translator(LibTcg,
                                   Builder,
                                   Variables,
                                   JumpTargets,
                                   Blocks,
                                   EndianessMismatch,
                                   PCH.get());

  std::tie(VirtualAddress, Entry) = JumpTargets.peek();

  while (Entry != nullptr) {
    llvm::ArrayRef<uint8_t> CodeBuffer;
    CodeBuffer = RawBinary.getFromAddressOn(VirtualAddress).value();

    LiftTask.advance(VirtualAddress.toString(), true);

    Task TranslateTask(3, "Translate");
    TranslateTask.advance("Lift to PTC", true);

    Builder.SetInsertPoint(Entry);

    // TODO: what if create a new instance of an InstructionTranslator here?
    Translator.reset();

    uint32_t TranslateFlags = 0;
    if (VirtualAddress.type() == MetaAddressType::Code_arm_thumb) {
      TranslateFlags |= LIBTCG_TRANSLATE_ARM_THUMB;
    }

    auto TranslationBlock = LibTcg.translateBlock(CodeBuffer.data(),
                                                  CodeBuffer.size(),
                                                  VirtualAddress.address(),
                                                  TranslateFlags);

    // TODO: rename this type
    const size_t ConsumedSize = TranslationBlock->size_in_bytes;
    revng_assert(ConsumedSize > 0);

    SmallSet<unsigned, 1> ToIgnore;
    // Handles writes to btarget, represents branching for microblaze/mips/cris
    ToIgnore = Translator.preprocess(*TranslationBlock);

    if (LibTcgLog.isEnabled()) {
      static std::array<char, 128> DumpBuf{ 0 };
      LibTcgLog << "Translation starting from " << VirtualAddress.toGeneric()
                << " (size: " << ConsumedSize << " bytes)" << DoLog;
      LoggerIndent Indent(LibTcgLog);
      LibTcgLog.indent();
      for (size_t I = 0; I < TranslationBlock->instruction_count; ++I) {
        auto Opcode = TranslationBlock->list[I].opcode;
        bool IsInstructionStart = Opcode == LIBTCG_op_insn_start;
        if (IsInstructionStart)
          LibTcgLog.unindent();
        LibTcg.dumpInstructionToBuffer(&TranslationBlock->list[I],
                                       DumpBuf.data(),
                                       DumpBuf.size());
        LibTcgLog << StringRef(DumpBuf.data()).trim();
        if (ToIgnore.contains(I))
          LibTcgLog << " (ignored)";
        LibTcgLog << DoLog;
        if (IsInstructionStart)
          LibTcgLog.indent();
      }
      LibTcgLog.unindent();
    }

    Variables.newTranslationBlock();
    bool StopTranslation = false;

    MetaAddress PC = VirtualAddress;
    MetaAddress NextPC = MetaAddress::invalid();
    MetaAddress EndPC = VirtualAddress + ConsumedSize;

    const auto InstructionCount = TranslationBlock->instruction_count;
    using IT = InstructionTranslator;
    IT::TranslationResult Result;

    unsigned J = 0;

    // Handle the first LIBTCG_op_insn_start
    {
      LibTcgInstruction *NextInstruction = nullptr;
      for (unsigned K = 1; K < InstructionCount; K++) {
        LibTcgInstruction *I = &TranslationBlock->list[K];
        if (I->opcode == LIBTCG_op_insn_start && ToIgnore.count(K) == 0) {
          NextInstruction = I;
          break;
        }
      }
      LibTcgInstruction *Instruction = &TranslationBlock->list[J];
      std::tie(Result, PC, NextPC) = Translator.newInstruction(Instruction,
                                                               NextInstruction,
                                                               VirtualAddress,
                                                               EndPC,
                                                               true);
      J++;
    }

    unsigned SinceInstructionStart = 0;

    // TODO: shall we move this whole loop in InstructionTranslator?
    for (; J < InstructionCount && !StopTranslation; J++) {
      if (ToIgnore.count(J) != 0)
        continue;

      LibTcgInstruction *Instruction = &TranslationBlock->list[J];
      auto Opcode = Instruction->opcode;

      Blocks.clear();
      Blocks.push_back(Builder.GetInsertBlock());

      ++SinceInstructionStart;

      switch (Opcode) {
      case LIBTCG_op_discard:
        // Instructions we don't even consider
        break;
      case LIBTCG_op_insn_start: {
        SinceInstructionStart = 0;

        // Find next instruction, if there is one
        LibTcgInstruction *NextInstruction = nullptr;
        for (unsigned K = J + 1; K < InstructionCount; K++) {
          LibTcgInstruction *I = &TranslationBlock->list[K];
          if (I->opcode == LIBTCG_op_insn_start && ToIgnore.count(K) == 0) {
            NextInstruction = I;
            break;
          }
        }

        std::tie(Result,
                 PC,
                 NextPC) = Translator.newInstruction(Instruction,
                                                     NextInstruction,
                                                     VirtualAddress,
                                                     EndPC,
                                                     false);
      } break;
      case LIBTCG_op_call: {
        Result = Translator.translateCall(Instruction,
                                          PC,
                                          SinceInstructionStart);

        // Sometimes libtinycode terminates a basic block with a call, in this
        // case force a fallthrough
        if (J == TranslationBlock->instruction_count - 1) {
          BasicBlock *Target = JumpTargets.registerJT(EndPC,
                                                      JTReason::PostHelper);
          if (Target != nullptr) {
            Builder.CreateBr(&notNull(Target));
          } else {
            emitAbort(Builder, "");
          }
        }
      } break;
      default:
        Result = Translator.translate(Instruction,
                                      PC,
                                      SinceInstructionStart,
                                      NextPC);
        break;
      }

      switch (Result) {
      case IT::Success:
        // No-op
        break;
      case IT::Abort:
        emitAbort(Builder, "");
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
        static std::array<char, 128> DumpBuf{ 0 };
        LibTcg.dumpInstructionToBuffer(&TranslationBlock->list[J],
                                       DumpBuf.data(),
                                       DumpBuf.size());

        // Eh not very nice to strlen in construction of the StringRef,
        // maybe we can get the length from the LibTcg call above?
        StringRef Str{ DumpBuf.data() };
        MDString *MDLibTcgString = MDString::get(Context, Str);
        MDLibTcgInstr = MDNode::getDistinct(Context, MDLibTcgString);
      }

      // Set metadata for all the new instructions
      for (BasicBlock *Block : Blocks) {
        BasicBlock::iterator I = Block->end();
        while (I != Block->begin() && !(--I)->hasMetadata()) {
          if (MDLibTcgInstr != nullptr)
            I->setMetadata(LibTcgInstrMDKind, MDLibTcgInstr);
        }
      }

    } // End loop over instructions

    TranslateTask.complete();
    TranslateTask.advance("Finalization", true);

    // We might have a leftover block, probably due to the block created after
    // the last call to exit_tb
    auto *LastBlock = Builder.GetInsertBlock();
    if (LastBlock->empty()) {
      eraseFromParent(LastBlock);
    } else if (!LastBlock->rbegin()->isTerminator()) {
      // Something went wrong, probably a mistranslation
      Builder.CreateUnreachable();
    }

    Translator.registerDirectJumps();
    // Obtain a new program counter to translate
    TranslateTask.complete();
    LiftTask.advance("Peek new address", true);
    std::tie(VirtualAddress, Entry) = JumpTargets.peek();
  } // End translations loop

  LiftTask.complete();

  OI.drop();

  // Reorder basic blocks in RPOT
  T.advance("Reordering basic blocks", true);
  {
    BasicBlock *Entry = &RootFunction->getEntryBlock();
    ReversePostOrderTraversal<BasicBlock *> RPOT(Entry);
    std::set<BasicBlock *> SortedBasicBlocksSet;
    std::vector<BasicBlock *> SortedBasicBlocks;
    for (BasicBlock *BB : RPOT) {
      SortedBasicBlocksSet.insert(BB);
      SortedBasicBlocks.push_back(BB);
    }

    std::vector<BasicBlock *> Unreachable;
    for (BasicBlock &BB : *RootFunction)
      if (!SortedBasicBlocksSet.contains(&BB))
        Unreachable.push_back(&BB);

    auto Size = RootFunction->size();
    for (unsigned I = 0; I < Size; ++I)
      RootFunction->begin()->removeFromParent();
    for (BasicBlock *BB : SortedBasicBlocks)
      RootFunction->insert(RootFunction->end(), BB);
    for (BasicBlock *BB : Unreachable)
      RootFunction->insert(RootFunction->end(), BB);
  }

  T.advance("IR finalization", true);

  // Remove the "helpers_list" variable, whose purpose is to keep alive helpers
  // who would otherwise get DCE'd away due to their linkage.
  // At this point we know which ones we want, and we're OK with the dead ones
  // to be dropped.
  TheModule->getGlobalVariable("helpers_list")->eraseFromParent();

  //
  // Look for calls to functions that might exit and reset cpu_loop_exiting
  //
  auto *CpuLoopExiting = TheModule->getGlobalVariable("cpu_loop_exiting", true);
  auto *BoolType = CpuLoopExiting->getValueType();
  for (BasicBlock &BB : *RootFunction) {
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      if (Call == nullptr)
        continue;

      auto *Callee = getCalledFunction(Call);
      if (Callee == nullptr or not Callee->hasMetadata("revng.cpu_loop_exits"))
        continue;

      new StoreInst(ConstantInt::getFalse(BoolType),
                    CpuLoopExiting,
                    Call->getNextNode());
      // TODO: are we guaranteed to have a check for the PC and go back to the
      //       dispatcher here if there's a mismatch?
    }
  }

  // Add a call to the function to initialize the CPUState, if present.
  // This is important on x86 architecture.
  // We only add the call after the Linker has imported the
  // helper_initialize_env function from the helpers, because the declaration
  // imported before with importHelperFunctionDeclaration() only has
  // stub types and injecting the CallInst earlier would break
  if (Function *InitEnv = getIRHelper("helper_initialize_env", *TheModule)) {
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

  legacy::FunctionPassManager InstCombinePM(&*TheModule);
  InstCombinePM.add(createSROAPass());
  InstCombinePM.add(createInstructionCombiningPass());
  InstCombinePM.add(createDeadCodeEliminationPass());
  InstCombinePM.doInitialization();
  InstCombinePM.run(*RootFunction);
  InstCombinePM.doFinalization();

  // Ensure we don't have phis in the dispatcher. This can happen if a tiny code
  // local variable has an uninitialized read, which is usually a bug on our
  // part.
  auto Phis = JumpTargets.dispatcher()->phis();
  revng_assert(Phis.begin() == Phis.end(),
               "A phi has appeared in the dispatcher");

  legacy::PassManager PostInstCombinePM;
  PostInstCombinePM.add(new LoadModelWrapperPass(Model));
  PostInstCombinePM.add(new PruneRetSuccessors);
  PostInstCombinePM.add(createGlobalDCEPass());
  PostInstCombinePM.run(*TheModule);

  T.advance("Finalize jump targets", true);
  JumpTargets.finalizeJumpTargets();

  T.advance("Purge dead code", true);
  EliminateUnreachableBlocks(*RootFunction, nullptr, false);

  T.advance("Create revng.jt.reason", true);
  JumpTargets.createJTReasonMD();

  T.advance("Finalization", true);
  ExternalJumpsHandler JumpOutHandler(*Model,
                                      JumpTargets.dispatcher(),
                                      *RootFunction,
                                      PCH.get());
  JumpOutHandler.createExternalJumpsHandler();

  Variables.finalize();
}
