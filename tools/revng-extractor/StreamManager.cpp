//
// This file is distributed under the MIT License. See LICENSE.md for details.
// \file StreamManager.cpp
// \brief Handles the stream of symbols in a pdb object
//
// revng includes
#include "InputFile.h"
#include "ParametersExtractor.h"
#include "StreamManager.h"

// llvm includes
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/DebugSymbolsSubsection.h"
#include "llvm/DebugInfo/CodeView/Formatters.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/ISectionContribVisitor.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/PublicsStream.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiHashing.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

// standard includes
#include <cctype>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

StreamManager::StreamManager(InputFile &File,
                             std::map<std::string, FunctionDecl> &FunctionMap,
                             const std::string &LibName) :
  LibName(LibName), FunctionMap(FunctionMap), File(File) {
}

PDBFile &StreamManager::getPdb() {
  return File.pdb();
}
object::COFFObjectFile &StreamManager::getObj() {
  return File.obj();
}

Error StreamManager::dump() {

  auto EC = File.isPdb() ? dumpModuleSymsForPdb() : dumpModuleSymsForObj();
  if (EC)
    return EC;

  return Error::success();
}

static Expected<ModuleDebugStreamRef>
getModuleDebugStream(PDBFile &File, uint32_t Index) {
  ExitOnError Err("Unexpected error: ");

  auto &Dbi = Err(File.getPDBDbiStream());
  const auto &Modules = Dbi.modules();
  auto Modi = Modules.getModuleDescriptor(Index);

  uint16_t ModiStream = Modi.getModuleStreamIndex();
  if (ModiStream == kInvalidStreamIndex)
    return make_error<RawError>(raw_error_code::no_stream,
                                "Module stream not present");

  auto ModStreamData = File.createIndexedStream(ModiStream);

  ModuleDebugStreamRef ModS(Modi, std::move(ModStreamData));
  if (auto EC = ModS.reload())
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid module stream");

  return std::move(ModS);
}

template<typename SubsecT>
using Cb = function_ref<void(uint32_t, const SymbolGroup &, SubsecT &)>;

template<typename SubsecT>
static void iterateModuleSubsections(InputFile &File, Cb<SubsecT> Callback) {

  uint32_t Modi = 0;
  for (const auto &SG : File.symbol_groups()) {
    for (const auto &SS : SG.getDebugSubsections()) {
      SubsecT Subsection;

      if (SS.kind() != Subsection.kind())
        continue;

      BinaryStreamReader Reader(SS.getRecordData());
      if (auto EC = Subsection.initialize(Reader))
        continue;
      Callback(Modi, SG, Subsection);
    }

    Modi++;
  }
}

Error StreamManager::dumpModuleSymsForObj() {

  ExitOnError Err("Unexpected error processing symbols: ");

  auto &Types = File.types();

  SymbolVisitorCallbackPipeline Pipeline;
  SymbolDeserializer Deserializer(nullptr, CodeViewContainer::ObjectFile);
  ParametersExtractor Dumper(Types, Types, FunctionMap, LibName);

  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Dumper);
  CVSymbolVisitor Visitor(Pipeline);

  std::unique_ptr<llvm::Error> SymbolError;

  auto SymbolIterator = [&](uint32_t Modi,
                            const SymbolGroup &Strings,
                            DebugSymbolsSubsectionRef &Symbols) {
    for (auto Symbol : Symbols) {
      if (auto EC = Visitor.visitSymbolRecord(Symbol)) {
        SymbolError = std::make_unique<Error>(std::move(EC));
        return;
      }
    }
  };

  iterateModuleSubsections<DebugSymbolsSubsectionRef>(File, SymbolIterator);

  if (SymbolError)
    return std::move(*SymbolError);

  return Error::success();
}

Error StreamManager::dumpModuleSymsForPdb() {

  if (!getPdb().hasPDBDbiStream()) {
    return Error::success();
  }

  ExitOnError Err("Unexpected error processing symbols: ");

  auto &Ids = File.ids();
  auto &Types = File.types();

  uint32_t I = 0;
  for (const auto &SG : File.symbol_groups()) {
    printf("%s\n", SG.name().str().c_str());
    auto ExpectedModS = getModuleDebugStream(File.pdb(), I);
    if (!ExpectedModS) {
      ++I;
      continue;
    }
    ModuleDebugStreamRef &ModS = *ExpectedModS;

    SymbolVisitorCallbackPipeline Pipeline;
    SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
    ParametersExtractor Dumper(Ids, Types, FunctionMap, LibName);

    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Dumper);
    CVSymbolVisitor Visitor(Pipeline);
    auto SS = ModS.getSymbolsSubstream();
    if (auto EC = Visitor.visitSymbolStream(ModS.getSymbolArray(), SS.Offset)) {
      ++I;
      continue;
    }
    ++I;
  }
  return Error::success();
}
