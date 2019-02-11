/// \file debughelper.cpp
/// \brief This file handles debugging information generation.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <string>

// LLVM includes
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng/Support/CommandLine.h"
#include "revng/Support/DebugHelper.h"

using namespace llvm;

/// Boring code to get the text of the metadata with the specified kind
/// associated to the given instruction
static StringRef getText(const Instruction *Instruction, unsigned Kind) {
  revng_assert(Instruction != nullptr);

  Metadata *MD = Instruction->getMetadata(Kind);

  if (MD == nullptr)
    return StringRef();

  auto Node = dyn_cast<MDNode>(MD);

  revng_assert(Node != nullptr);

  const MDOperand &Operand = Node->getOperand(0);

  Metadata *MDOperand = Operand.get();

  if (MDOperand == nullptr)
    return StringRef();

  if (auto *String = dyn_cast<MDString>(MDOperand)) {
    return String->getString();
  } else if (auto *CAM = dyn_cast<ConstantAsMetadata>(MDOperand)) {
    auto *Cast = cast<ConstantExpr>(CAM->getValue());
    auto *GV = cast<GlobalVariable>(Cast->getOperand(0));
    auto *Initializer = GV->getInitializer();
    return cast<ConstantDataArray>(Initializer)->getAsString().drop_back();
  } else {
    revng_abort();
  }
}

static void
replaceAll(std::string &Input, const std::string &From, const std::string &To) {
  if (From.empty())
    return;

  size_t Start = 0;
  while ((Start = Input.find(From, Start)) != std::string::npos) {
    Input.replace(Start, From.length(), To);
    Start += To.length();
  }
}

/// Writes the text contained in the metadata with the specified kind ID to the
/// output stream, unless that metadata is exactly the same as in the previous
/// instruction.
static void writeMetadataIfNew(const Instruction *TheInstruction,
                               unsigned MDKind,
                               formatted_raw_ostream &Output,
                               StringRef Prefix) {
  auto BeginIt = TheInstruction->getParent()->begin();
  StringRef Text = getText(TheInstruction, MDKind);
  if (Text.size()) {
    StringRef LastText;

    do {
      if (TheInstruction->getIterator() == BeginIt) {
        TheInstruction = nullptr;
      } else {
        TheInstruction = TheInstruction->getPrevNode();
        LastText = getText(TheInstruction, MDKind);
      }
    } while (TheInstruction != nullptr && LastText.size() == 0);

    if (TheInstruction == nullptr or LastText != Text) {
      std::string TextToSerialize = Text.str();
      replaceAll(TextToSerialize, "\n", " ");
      Output << Prefix << TextToSerialize << "\n";
    }
  }
}

/// Add a module flag, if not already present, using name and value provided.
/// Used for creating the Dwarf compliant debug info.
static void addModuleFlag(Module *TheModule, StringRef Flag, uint32_t Value) {
  if (TheModule->getModuleFlag(Flag) == nullptr) {
    TheModule->addModuleFlag(Module::Warning, Flag, Value);
  }
}

using DAW = DebugAnnotationWriter;

DAW::DebugAnnotationWriter(LLVMContext &Context, bool DebugInfo) :
  Context(Context),
  DebugInfo(DebugInfo) {
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");
}

void DAW::emitInstructionAnnot(const Instruction *Instr,
                               formatted_raw_ostream &Output) {
  DISubprogram *Subprogram = Instr->getParent()->getParent()->getSubprogram();

  // Ignore whatever is outside the root and the isolated functions
  StringRef FunctionName = Instr->getParent()->getParent()->getName();
  if (Subprogram == nullptr
      or not(FunctionName == "root" or FunctionName.startswith("bb.")))
    return;

  writeMetadataIfNew(Instr, OriginalInstrMDKind, Output, "\n  ; ");
  writeMetadataIfNew(Instr, PTCInstrMDKind, Output, "\n  ; ");

  if (DebugInfo) {
    // If DebugInfo is activated the generated LLVM IR textual representation
    // will contain some reference to dangling pointers. So ignore the output
    // stream if you're using the annotator to generate debug info about the IR
    // itself.
    revng_assert(Subprogram != nullptr);

    // Flushing is required to have correct line and column numbers
    Output.flush();

    auto *Location = DILocation::get(Context,
                                     Output.getLine() + 1,
                                     Output.getColumn(),
                                     Subprogram);

    // Sorry Bjarne
    auto *NonConstInstruction = const_cast<Instruction *>(Instr);
    NonConstInstruction->setMetadata(DbgMDKind, Location);
  }
}

DebugHelper::DebugHelper(std::string Output,
                         Module *TheModule,
                         DebugInfoType::Values DebugInfo,
                         std::string DebugPath) :
  OutputPath(Output),
  Builder(*TheModule),
  TheModule(TheModule),
  DebugInfo(DebugInfo),
  DebugPath(DebugPath) {

  OriginalInstrMDKind = TheModule->getContext().getMDKindID("oi");
  PTCInstrMDKind = TheModule->getContext().getMDKindID("pi");
  DbgMDKind = TheModule->getContext().getMDKindID("dbg");

  // Generate automatically the name of the source file for debugging
  if (DebugPath.empty()) {
    if (DebugInfo == DebugInfoType::PTC)
      this->DebugPath = OutputPath + ".ptc";
    else if (DebugInfo == DebugInfoType::OriginalAssembly)
      this->DebugPath = OutputPath + ".S";
    else if (DebugInfo == DebugInfoType::LLVMIR)
      this->DebugPath = OutputPath;
  }

  if (DebugInfo != DebugInfoType::None) {
    auto File = Builder.createFile(this->DebugPath, "");
    CompileUnit = Builder.createCompileUnit(dwarf::DW_LANG_C,
                                            File,
                                            "revng",
                                            false,
                                            "",
                                            0 /* Runtime version */);

    // Add the current debug info version into the module after checking if it
    // is already present.
    addModuleFlag(TheModule, "Debug Info Version", DEBUG_METADATA_VERSION);
    addModuleFlag(TheModule, "Dwarf Version", 4);
  }
}

void DebugHelper::generateDebugInfo() {
  for (Function &F : TheModule->functions()) {
    // TODO: find a better way to identify root and the isolated functions
    if (F.getName() == "root" || F.getName().startswith("bb.")) {
      if (DebugInfo != DebugInfoType::None) {
        DISubroutineType *EmptyType = nullptr;
        DITypeRefArray EmptyArrayType = Builder.getOrCreateTypeArray({});
        EmptyType = Builder.createSubroutineType(EmptyArrayType);

        revng_assert(CompileUnit != nullptr);
        DISubprogram *Subprogram = nullptr;
        Subprogram = Builder.createFunction(CompileUnit->getFile(), // Scope
                                            F.getName(),
                                            StringRef(), // Linkage name
                                            CompileUnit->getFile(),
                                            1, // Line
                                            EmptyType, // Subroutine type
                                            false, // isLocalToUnit
                                            true, // isDefinition
                                            1, // ScopeLine
                                            DINode::FlagPrototyped,
                                            false /* isOptimized */);
        F.setSubprogram(Subprogram);
      }
    }
  }

  switch (DebugInfo) {
  case DebugInfoType::PTC:
  case DebugInfoType::OriginalAssembly: {
    // Generate the source file and the debugging information in tandem

    unsigned LineIndex = 1;
    unsigned MetadataKind = DebugInfo == DebugInfoType::PTC ?
                              PTCInstrMDKind :
                              OriginalInstrMDKind;

    StringRef Last;
    std::ofstream Source(DebugPath);
    for (Function &F : TheModule->functions()) {

      if (not(F.getName() == "root" || F.getName().startswith("bb.")))
        continue;

      if (DISubprogram *CurrentSubprogram = F.getSubprogram()) {
        for (BasicBlock &Block : F) {
          for (Instruction &Instruction : Block) {
            StringRef Body = getText(&Instruction, MetadataKind);

            if (Body.size() != 0 && Last != Body) {
              Last = Body;
              Source << Body.data();

              auto *Location = DILocation::get(TheModule->getContext(),
                                               LineIndex,
                                               0,
                                               CurrentSubprogram);
              Instruction.setMetadata(DbgMDKind, Location);
              LineIndex += std::count(Body.begin(), Body.end(), '\n');
            }
          }
        }
      }
    }

    Builder.finalize();
  } break;
  case DebugInfoType::LLVMIR: {
    // Use the annotator to obtain line and column of the textual LLVM IR for
    // each instruction. Discard the output since it will contain errors,
    // regenerating it later will give a correct result.
    Builder.finalize();

    raw_null_ostream NullStream;
    TheModule->print(NullStream, annotator(true /* DebugInfo */));

    std::ofstream Output(DebugPath);
    raw_os_ostream Stream(Output);
    TheModule->print(Stream, annotator(false));

  } break;
  default:
    break;
  }
}

void DebugHelper::print(std::ostream &Output, bool DebugInfo) {
  raw_os_ostream OutputStream(Output);
  TheModule->print(OutputStream, annotator(DebugInfo));
}

bool DebugHelper::copySource() {
  // If debug info refer to LLVM IR, just copy the output file
  if (DebugInfo == DebugInfoType::LLVMIR && DebugPath != OutputPath) {
    std::ifstream Source(DebugPath, std::ios::binary);
    std::ofstream Destination(OutputPath, std::ios::binary);

    Destination << Source.rdbuf();

    return true;
  }

  return false;
}

DAW *DebugHelper::annotator(bool DebugInfo) {
  Annotator.reset(new DAW(TheModule->getContext(), DebugInfo));
  return Annotator.get();
}
