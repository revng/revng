//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <string>
#include <system_error>

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

static Logger Log("import-debug-info");

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::cat(ThisToolCategory),
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> OutputPath("o",
                                       cl::cat(ThisToolCategory),
                                       cl::desc("Override output "
                                                "filename"),
                                       cl::init("-"),
                                       cl::value_desc("filename"));

static cl::opt<std::string> Root("root",
                                 cl::cat(ThisToolCategory),
                                 cl::desc("Root where to look for debug "
                                          "symbols"),
                                 cl::init("/"),
                                 cl::value_desc("ROOT_PATH"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  // Open output.
  ExitOnError ExitOnError;
  std::error_code EC;
  ToolOutputFile OutputFile(OutputPath, EC, sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(createStringError(EC, EC.message()));

  const ImporterOptions &Options = importerOptions();

  // Import debug info from both PE and ELF.
  TupleTree<model::Binary> Model;

  auto MaybeBinary = object::createBinary(InputFilename);

  using ObjectFileType = object::ObjectFile;
  using COFFObjectFileType = object::COFFObjectFile;

  ObjectFileType *ObjectFile = nullptr;
  if (MaybeBinary) {
    ObjectFile = cast<ObjectFileType>(MaybeBinary->getBinary());
  } else {
    consumeError(MaybeBinary.takeError());
  }

  file_magic FileType;
  EC = identify_magic(InputFilename, FileType);

  if (EC) {
    dbg << "Couldn't identify file type: " << EC.message() << "\n";
    return EXIT_FAILURE;
  }

  if (dyn_cast_or_null<object::ELFObjectFileBase>(ObjectFile)) {
    DwarfImporter Importer(Model, std::nullopt);
    Importer.import(InputFilename, Options);
  } else if (auto *Binary = dyn_cast_or_null<COFFObjectFileType>(ObjectFile)) {
    MetaAddress ImageBase = MetaAddress::invalid();
    auto LLVMArch = ObjectFile->makeTriple().getArch();
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

    const object::pe32_header *PE32Header = Binary->getPE32Header();
    auto Architecture = model::Architecture::fromLLVMArchitecture(LLVMArch);
    if (PE32Header) {
      ImageBase = MetaAddress::fromPC(Architecture, PE32Header->ImageBase);
    } else {
      const auto *PE32PlusHeader = Binary->getPE32PlusHeader();
      if (not PE32PlusHeader)
        return EXIT_FAILURE;

      // PE32+ Header.
      ImageBase = MetaAddress::fromPC(Architecture, PE32PlusHeader->ImageBase);
    }

    PDBImporter Importer(Model, ImageBase, std::nullopt);
    Importer.importPDB(InputFilename, Options);
  } else if (FileType == file_magic::pdb) {
    PDBImporter Importer(Model, MetaAddress::invalid(), std::nullopt);
    Importer.importPDB(InputFilename, Options);
  } else {
    dbg << "Unexpected file format\n";
    return EXIT_FAILURE;
  }

  // Serialize the model.
  Model.serialize(OutputFile.os());

  OutputFile.keep();

  return EXIT_SUCCESS;
}
