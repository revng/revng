//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"

#include "DwarfDebugInfoFinder.h"
#include "DwarfToModelConverter.h"
#include "ImportDebugInfoHelper.h"

using namespace llvm;
using namespace llvm::dwarf;

size_t DwarfImporter::import(StringRef FileName,
                             const ImporterOptions &Options) {
  revng_assert(not FileName.empty());
  revng_log(DILogger, "Importing DWARF information for " << FileName);
  LoggerIndent Indent(DILogger);

  auto MaybeBinary = object::createBinary(FileName);

  if (not MaybeBinary) {
    std::string Message = llvm::toString(MaybeBinary.takeError());
    revng_log(DILogger, "Can't create binary: " << Message);
    return -1;
  }

  return import(*MaybeBinary->getBinary(), FileName, Options.BaseAddress, -1);
}

size_t DwarfImporter::import(const revng::RootEntry *Root,
                             const ELFBinary &Binary,
                             const ImporterOptions &Options) {
  using namespace llvm::object;

  size_t AltIndex = -1;

  if (Root != nullptr) {
    auto Handler =
      [this, &Root, &Binary, &Options](const auto *ObjectFile) -> size_t {
      using T = std::remove_cvref_t<decltype(*ObjectFile)>;
      std::string Path = DwarfDebugInfoFinder<T>::find(*Root,
                                                       Binary,
                                                       *ObjectFile);
      if (Path.size() != 0) {
        // Note: we recur at most once on this path
        revng_log(DILogger, "Importing DWARF file: " << Path);
        return import(Path, Options);
      } else {
        revng_log(DILogger, "DebugInfoFinder failed to find debug info");
        return -1;
      }
    };

    if (auto *ELF = dyn_cast<ELF32LEObjectFile>(&Binary.ObjectFile)) {
      AltIndex = Handler(ELF);
    } else if (auto *ELF = dyn_cast<ELF32BEObjectFile>(&Binary.ObjectFile)) {
      AltIndex = Handler(ELF);
    } else if (auto *ELF = dyn_cast<ELF64LEObjectFile>(&Binary.ObjectFile)) {
      AltIndex = Handler(ELF);
    } else if (auto *ELF = dyn_cast<ELF64BEObjectFile>(&Binary.ObjectFile)) {
      AltIndex = Handler(ELF);
    } else {
      revng_abort();
    }
  }

  return import(Binary.ObjectFile,
                Binary.canonicalPath(),
                Options.BaseAddress,
                AltIndex);
}

auto zipPairs(auto &&R) {
  auto BeginIt = R.begin();
  auto EndIt = R.end();

  if (BeginIt == EndIt)
    return zip(make_range(EndIt, EndIt), make_range(EndIt, EndIt));

  auto First = BeginIt;
  auto Second = ++BeginIt;

  if (Second == EndIt)
    return zip(make_range(EndIt, EndIt), make_range(EndIt, EndIt));

  auto End = EndIt;
  auto Last = --EndIt;

  return zip(make_range(First, Last), make_range(Second, End));
}

/// This function considers all symbols with name of type STT_FUNC and clusters
/// them by address/type
static EquivalenceClasses<StringRef>
computeEquivalentSymbols(const llvm::object::ObjectFile &ELF) {
  using namespace llvm::object;

  EquivalenceClasses<StringRef> Result;

  struct SymbolDescriptor {
    uint64_t Address = 0;
    // TODO: one day we will want to consider STT_OBJECT too
    SymbolRef::Type Type = SymbolRef::ST_Unknown;
    /// \note we ignore this field for comparison purposes
    StringRef Name;

    auto key() const { return std::tie(Address, Type); }

    bool operator<(const SymbolDescriptor &Other) const {
      return key() < Other.key();
    }

    bool operator==(const SymbolDescriptor &Other) const {
      return key() == Other.key();
    }
  };

  std::vector<SymbolDescriptor> Symbols;
  for (const object::SymbolRef &Symbol : ELF.symbols()) {
    SymbolDescriptor NewSymbol;

    auto MaybeType = Symbol.getType();
    // Note: on ARM, LLVM's SymbolRef::getAddress clears the LSB of st_value for
    // STT_FUNC symbols (the Thumb indicator bit), so this address is the
    // Thumb-aligned base with bit 0 cleared.
    // This is a problem, but given that we use that value just to check if it's
    // non-zero, this is not a problem.
    auto MaybeAddress = Symbol.getAddress();
    auto MaybeName = Symbol.getName();
    auto MaybeFlags = Symbol.getFlags();

    if (auto Error = MaybeType.takeError()) {
      revng_log(DILogger, "Cannot access symbol type: " << Error);
      consumeError(std::move(Error));
      continue;
    } else if (auto Error = MaybeAddress.takeError()) {
      revng_log(DILogger, "Cannot access symbol address: " << Error);
      consumeError(std::move(Error));
      continue;
    } else if (auto Error = MaybeName.takeError()) {
      revng_log(DILogger, "Cannot access symbol name: " << Error);
      consumeError(std::move(Error));
      continue;
    } else if (auto Error = MaybeFlags.takeError()) {
      revng_log(DILogger, "Cannot access symbol flags: " << Error);
      consumeError(std::move(Error));
      continue;
    }

    // Ignore unnamed and nullptr symbols
    if (MaybeName->size() == 0 or *MaybeAddress == 0)
      continue;

    // Consider only STT_FUNC symbols
    if (*MaybeType != SymbolRef::ST_Function)
      continue;

    // Consider only global symbols
    if (!((*MaybeFlags) & SymbolRef::SF_Global))
      continue;

    Symbols.push_back({ *MaybeAddress, *MaybeType, *MaybeName });
  }

  llvm::sort(Symbols);

  for (const auto &[Previous, Current] : zipPairs(Symbols))
    if (Previous == Current)
      Result.unionSets(Previous.Name, Current.Name);

  return Result;
}

// TODO: it would be beneficial to do this even at other levels
static void detectAliases(const llvm::object::ObjectFile &ELF,
                          TupleTree<model::Binary> &Model) {
  EquivalenceClasses<StringRef> Aliases = computeEquivalentSymbols(ELF);
  auto &ImportedDynamicFunctions = Model->ImportedDynamicFunctions();
  auto &Functions = Model->Functions();

  std::unordered_map<std::string, model::Function *> FunctionsByName;
  // Map functions by names, so we have faster lookup below.
  for (auto &Function : Functions) {
    if (Function.Name().size()) {
      FunctionsByName[Function.Name()] = &Function;
    }
  }

  for (auto AliasesIt = Aliases.begin(), E = Aliases.end(); AliasesIt != E;
       ++AliasesIt) {
    llvm::SmallVector<std::string, 4> CurrentAliases;
    if (AliasesIt->isLeader()) {
      SmallVector<std::string, 4> UnprototypedFunctionsNames;
      model::UpcastableType Prototype;
      for (auto AliasSetIt = Aliases.member_begin(AliasesIt);
           AliasSetIt != Aliases.member_end();
           ++AliasSetIt) {

        std::string Name = AliasSetIt->str();
        if (Name.size() == 0)
          continue;

        CurrentAliases.push_back(Name);

        // Create DynamicFunction, if it doesn't exist already
        auto It = ImportedDynamicFunctions.tryGet(Name);
        bool Found = It != nullptr;

        // If DynamicFunction doesn't have a prototype, register it for copying
        // it from the leader.
        // Otherwise, record the type as the leader.
        if (Found and not It->Prototype().isEmpty()) {
          Prototype = It->Prototype().copy();
        } else {
          UnprototypedFunctionsNames.push_back(Name);
        }
      }

      // Check if we should add an ExportedName for local Functions.
      llvm::SmallVector<std::string, 4> PotentialExportedNamesToBeAdded;
      bool IsLocalFunction = false;
      model::Function *TheFunction = nullptr;
      for (auto &Name : CurrentAliases) {
        auto It = FunctionsByName.find(Name);
        PotentialExportedNamesToBeAdded.push_back(Name);
        if (It != FunctionsByName.end()) {
          // We found a local function.
          // TODO: In some situations Name is not in the ExportedNames?
          // For example in the case of importing `__libc_calloc` from
          // libc.so.6.
          TheFunction = It->second;
        }
      }
      // It is a local function. Populate the ExportedNames.
      if (TheFunction) {
        for (auto &Name : PotentialExportedNamesToBeAdded)
          TheFunction->ExportedNames().insert(Name);
        continue;
      }

      // Consider it as a Dynamic function.
      if (not Prototype.isEmpty()) {
        for (const std::string &Name : UnprototypedFunctionsNames) {
          auto It = ImportedDynamicFunctions.find(Name);
          if (It == ImportedDynamicFunctions.end())
            It = ImportedDynamicFunctions.insert({ Name }).first;

          It->Prototype() = std::move(Prototype);
        }
      }
    }
  }
}

size_t DwarfImporter::import(const llvm::object::Binary &TheBinary,
                             StringRef CanonicalPath,
                             uint64_t PreferredBaseAddress,
                             size_t AltIndex) {
  using namespace llvm::object;

  revng_log(DILogger, "Parsing DWARF of " << CanonicalPath);
  LoggerIndent Indent(DILogger);

  if (auto *ELF = dyn_cast<ELFObjectFileBase>(&TheBinary)) {

    {
      using namespace model::Architecture;
      if (Model->Architecture() == Invalid)
        Model->Architecture() = fromLLVMArchitecture(ELF->getArch());
    }

    if (ELF->getEType() != ELF::ET_DYN)
      PreferredBaseAddress = 0;

    auto TheDWARFContext = DWARFContext::create(*ELF);
    DwarfToModelConverter Converter(*this,
                                    *TheDWARFContext,
                                    LoadedFiles.size(),
                                    AltIndex,
                                    PreferredBaseAddress);
    Converter.run();

    detectAliases(*ELF, Model);
  }

  LoadedFiles.push_back(sys::path::filename(CanonicalPath).str());
  return LoadedFiles.size() - 1;
}
