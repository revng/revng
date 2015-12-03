/// \file
/// \brief This file handles the whole translation process from the input
///        assembly to LLVM IR.

// Standard includes
#include <cstdint>
#include <sstream>
#include <vector>
#include <fstream>

// LLVM includes
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/LegacyPassManager.h"

// Local includes
#include "codegenerator.h"
#include "debughelper.h"
#include "instructiontranslator.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"
#include "ptcinterface.h"
#include "variablemanager.h"

using namespace llvm;

static bool startsWith(std::string String, std::string Prefix) {
  return String.substr(0, Prefix.size()) == Prefix;
}

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

CodeGenerator::CodeGenerator(Architecture& Source,
                             Architecture& Target,
                             std::string Output,
                             std::string Helpers,
                             DebugInfoType DebugInfo,
                             std::string Debug) :
  SourceArchitecture(Source),
  TargetArchitecture(Target),
  Context(getGlobalContext()),
  TheModule((new Module("top", Context))),
  OutputPath(Output),
  Debug(new DebugHelper(Output, Debug, TheModule.get(), DebugInfo)),
  CPUStateType(nullptr),
  HelpersModuleLayout(nullptr)
{
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");

  SMDiagnostic Errors;
  HelpersModule = parseIRFile(Helpers, Errors, Context);

  using ElectionMap = std::map<StructType *, unsigned>;
  using ElectionMapElement = std::pair<StructType * const, unsigned>;
  ElectionMap EnvElection;
  const std::string HelperPrefix = "helper_";
  for (Function& HelperFunction : *HelpersModule) {
    if (startsWith(HelperFunction.getName(), HelperPrefix)
        && HelperFunction.getFunctionType()->getNumParams() > 1) {

      for (Type *Candidate : HelperFunction.getFunctionType()->params()) {
        if (Candidate->isPointerTy()) {
          auto *PointeeType = Candidate->getPointerElementType();
          auto *EnvType = dyn_cast<StructType>(PointeeType);
          // Ensure it is a struct and not a union
          if (EnvType != nullptr && EnvType->getNumElements() > 1) {

            auto It = EnvElection.find(EnvType);
            if (It != EnvElection.end())
              EnvElection[EnvType]++;
            else
              EnvElection[EnvType] = 1;
          }
        }
      }
    }
  }

  assert(EnvElection.size() > 0);

  CPUStateType = std::max_element(EnvElection.begin(),
                                  EnvElection.end(),
                                  [] (ElectionMapElement& It1,
                                      ElectionMapElement& It2) {
                                    return It1.second < It2.second;
                                  })->first;

  HelpersModuleLayout = &HelpersModule->getDataLayout();
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
                            CPUStateType,
                            HelpersModuleLayout);

  GlobalVariable *PCReg = Variables.getByCPUStateOffset(ptc.get_pc(), "pc");

  JumpTargetManager JumpTargets(*TheModule, PCReg, MainFunction);
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

    Translator.closeLastInstruction(VirtualAddress + ConsumedSize);

    // Before looking for writes to the PC, give a shot of SROA
    legacy::PassManager PM;
    PM.add(createSROAPass());
    PM.add(Translator.createTranslateDirectBranchesPass());
    PM.run(*TheModule);

    // Obtain a new program counter to translate
    uint64_t NewPC = 0;
    std::tie(NewPC, Entry) = JumpTargets.peekJumpTarget();
    VirtualAddress = NewPC;
    CodePointer = Code.data() + (NewPC - LoadAddress);
  } // End translations loop

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
