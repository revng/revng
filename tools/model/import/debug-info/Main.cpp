/// \file Main.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

static Logger<> Log("import-debug-info");

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
  ExitOnError ExitOnError;
  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));

  auto BinaryOrErr = object::createBinary(InputFilename);
  if (not BinaryOrErr) {
    revng_log(Log, "Unable to create binary: " << BinaryOrErr.takeError());
    llvm::consumeError(BinaryOrErr.takeError());
    return 1;
  }
  auto &ObjectFile = *cast<llvm::object::ObjectFile>(BinaryOrErr->getBinary());

  const ImporterOptions &Options = importerOptions();

  // Import debug info from both PE and ELF.
  TupleTree<model::Binary> Model;
  if (isa<llvm::object::ELFObjectFileBase>(&ObjectFile)) {
    DwarfImporter Importer(Model);
    Importer.import(InputFilename, Options);
  } else if (auto *TheBinary = dyn_cast<object::COFFObjectFile>(&ObjectFile)) {
    MetaAddress ImageBase = MetaAddress::invalid();
    auto LLVMArchitecture = ObjectFile.makeTriple().getArch();
    using namespace model::Architecture;
    using namespace model::ABI;
    Model->Architecture() = fromLLVMArchitecture(LLVMArchitecture);
    if (Model->DefaultABI() == model::ABI::Invalid)
      Model->DefaultABI() = getDefaultMicrosoftABI(Model->Architecture());

    // Create a default prototype.
    Model->DefaultPrototype() = abi::registerDefaultFunctionPrototype(*Model);

    const llvm::object::pe32_header *PE32Header = TheBinary->getPE32Header();
    if (PE32Header) {
      ImageBase = MetaAddress::fromPC(LLVMArchitecture, PE32Header->ImageBase);
    } else {
      const llvm::object::pe32plus_header
        *PE32PlusHeader = TheBinary->getPE32PlusHeader();
      if (not PE32PlusHeader)
        return EXIT_FAILURE;

      // PE32+ Header.
      ImageBase = MetaAddress::fromPC(LLVMArchitecture,
                                      PE32PlusHeader->ImageBase);
    }
    PDBImporter Importer(Model, ImageBase);
    Importer.import(*TheBinary, Options);
  }

  // Serialize the model.
  Model.serialize(OutputFile.os());

  OutputFile.keep();

  return EXIT_SUCCESS;
}
