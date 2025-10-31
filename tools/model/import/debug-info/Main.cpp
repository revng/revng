/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress.h"

namespace cl = llvm::cl;

static Logger Log("import-debug-info");

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::cat(ThisToolCategory),
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::cat(ThisToolCategory),
                                           llvm::cl::desc("Override output "
                                                          "filename"),
                                           llvm::cl::init("-"),
                                           llvm::cl::value_desc("filename"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  // Open output.
  llvm::ExitOnError ExitOnError;
  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  llvm::sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));

  auto BinaryOrErr = llvm::object::createBinary(InputFilename);
  if (not BinaryOrErr) {
    revng_log(Log, "Unable to create binary: " << BinaryOrErr.takeError());
    llvm::consumeError(BinaryOrErr.takeError());
    return 1;
  }

  using ObjectFileType = llvm::object::ObjectFile;
  using COFFObjectFileType = llvm::object::COFFObjectFile;
  auto &ObjectFile = *llvm::cast<ObjectFileType>(BinaryOrErr->getBinary());

  const ImporterOptions &Options = importerOptions();

  // Import debug info from both PE and ELF.
  TupleTree<model::Binary> Model;
  if (llvm::isa<llvm::object::ELFObjectFileBase>(&ObjectFile)) {
    DwarfImporter Importer(Model);
    Importer.import(InputFilename, Options);
  } else if (auto *Binary = llvm::dyn_cast<COFFObjectFileType>(&ObjectFile)) {
    MetaAddress ImageBase = MetaAddress::invalid();
    auto LLVMArch = ObjectFile.makeTriple().getArch();
    Model->Architecture() = model::Architecture::fromLLVMArchitecture(LLVMArch);

    if (Model->DefaultABI() == model::ABI::Invalid) {
      revng_assert(Model->Architecture() != model::Architecture::Invalid);
      if (auto ABI = model::ABI::getDefaultForPECOFF(Model->Architecture())) {
        Model->DefaultABI() = ABI.value();
      } else {
        auto AName = model::Architecture::getName(Model->Architecture()).str();
        revng_abort(("Unsupported architecture for PECOFF: " + AName).c_str());
      }
    }

    // Create a default prototype.
    Model->DefaultPrototype() = abi::registerDefaultFunctionPrototype(*Model);

    const llvm::object::pe32_header *PE32Header = Binary->getPE32Header();
    auto Architecture = model::Architecture::fromLLVMArchitecture(LLVMArch);
    if (PE32Header) {
      ImageBase = MetaAddress::fromPC(Architecture, PE32Header->ImageBase);
    } else {
      const llvm::object::pe32plus_header
        *PE32PlusHeader = Binary->getPE32PlusHeader();
      if (not PE32PlusHeader)
        return EXIT_FAILURE;

      // PE32+ Header.
      ImageBase = MetaAddress::fromPC(Architecture, PE32PlusHeader->ImageBase);
    }
    PDBImporter Importer(Model, ImageBase);
    Importer.import(*Binary, Options);
  }

  // Serialize the model.
  Model.serialize(OutputFile.os());

  OutputFile.keep();

  return EXIT_SUCCESS;
}
