#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/Concepts.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/ImportLogger.h"
#include "revng/Model/Importer/PrototypeMatching.h"
#include "revng/Model/Importer/TypeCopier.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Model/Processing.h"
#include "revng/Support/LDDTree.h"
#include "revng/TupleTree/TupleTree.h"

using TypeCopierMap = std::map<std::string, std::unique_ptr<TypeCopier>>;

template<StrictSpecializationOf<BinaryDescriptor> BinaryDescriptor,
         typename ImporterType>
void findMissingTypes(LDDTree &Dependencies,
                      const ImporterOptions &Options,
                      Logger &Logger,
                      TupleTree<model::Binary> &Binary) {
  using namespace llvm;

  if (Options.DebugInfo != DebugInfoLevel::Yes)
    return;

  ModelMap ModelsOfLibraries;
  TypeCopierMap TypeCopiers;

  revng_log(Logger, "Importing dependencies");
  LoggerIndent Indent(Logger);

  for (auto &[DependencyName, Dependency] : Dependencies.Dependencies) {
    using namespace model;

    revng_log(Logger, "Importing debug info for: " << DependencyName);
    LoggerIndent Indent(Logger);
    if (Logger.isEnabled()) {
      Logger << "Dependency:\n";
      Dependency.dump(Logger, "  ");
      Logger << DoLog;
    }

    revng_assert(!ModelsOfLibraries.contains(DependencyName));

    // Craft a minimal (temporary) model for the sole purpose of being imported
    auto &DependencyModel = ModelsOfLibraries[DependencyName];
    DependencyModel->Architecture() = Binary->Architecture();
    BinaryIdentifierReference BinaryReference;
    BinaryIdentifier Identifier;
    Identifier.Index() = 0;
    Identifier.Hash() = ("00000000000000000000000000000000"
                         "00000000000000000000000000000000");
    Identifier.CanonicalPath() = Dependency.canonicalPath();
    DependencyModel->Binaries().insert(std::move(Identifier));
    BinaryReference = DependencyModel->getBinaryIdentifierReference(0);

    bool Is64 = Architecture::getPointerSize(Binary->Architecture()) == 8;
    Segment Universe(MetaAddress::fromGeneric(Binary->Architecture(), 0));
    Universe.VirtualSize() = Is64 ? std::numeric_limits<uint64_t>::max() :
                                    std::numeric_limits<uint32_t>::max();
    Universe.FileSize() = Universe.VirtualSize();
    Universe.IsExecutable() = true;
    Universe.IsReadable() = true;
    Universe.IsWriteable() = true;
    Universe.Binary() = BinaryReference;
    DependencyModel->Segments().insert(std::move(Universe));

    ImporterOptions AdjustedOptions{
      .BaseAddress = Options.BaseAddress,
      .DebugInfo = DebugInfoLevel::IgnoreLibraries,
      .EnableRemoteDebugInfo = Options.EnableRemoteDebugInfo
    };

    using ObjectFileType = typename BinaryDescriptor::ObjectFileType;
    BinaryDescriptor TheBinary(cast<ObjectFileType>(Dependency.objectFile()),
                               Dependency.fullPathForExternalTools(),
                               BinaryReference);

    std::optional<std::map<MetaAddress, LDDTree::Symbol>> Whitelist;
    Whitelist.emplace();
    for (const auto &[Name, Symbol] : Dependency.providedSymbols()) {
      // TODO: usually this reponsability is demanded to BinaryImporterHelper
      auto FunctionAddress = Symbol.Address + Options.BaseAddress;
      (*Whitelist)[FunctionAddress] = Symbol;

      auto &Function = DependencyModel->Functions()[FunctionAddress];
      Function.Name() = Name;
      Function.ExportedNames().insert(Name);
    }

    ImportLogger ImportLogger(DependencyModel,
                              Logger,
                              TheBinary.canonicalPath());
    ImporterType Importer(DependencyModel, std::move(Whitelist));
    Importer.import(Dependencies.Root, TheBinary, AdjustedOptions);

    revng_log(Logger, DependencyName << " imported successfully");
  }

  auto GetOrMakeACopier = [&](llvm::StringRef Name) -> TypeCopier & {
    if (auto It = TypeCopiers.find(Name.str()); It != TypeCopiers.end())
      return *It->second;

    auto Iterator = ModelsOfLibraries.find(Name.str());
    revng_assert(Iterator != ModelsOfLibraries.end());

    auto NewCopier = std::make_unique<TypeCopier>(Iterator->second, Binary);
    auto &&[Result, Success] = TypeCopiers.emplace(Name.str(),
                                                   std::move(NewCopier));
    revng_assert(Success);
    return *Result->second;
  };

  revng_log(Logger, "Importing prototypes for dynamic functions");
  LoggerIndent Indent2(Logger);

  for (auto &DynamicFunction : Binary->ImportedDynamicFunctions()) {
    revng_log(Logger, "Considering " << DynamicFunction.Name());
    LoggerIndent Indent(Logger);

    if (DynamicFunction.Name().empty()) {
      revng_log(Logger, "It has no name, bailing out");
      continue;
    }

    auto MaybeFunction = findPrototype(DynamicFunction.Name(),
                                       ModelsOfLibraries);

    if (not MaybeFunction.has_value()) {
      revng_log(Logger,
                "Prototype for " << DynamicFunction.Name() << " not found");
      continue;
    }

    if (not DynamicFunction.Prototype().isEmpty()) {
      // The prototype already exist, but maybe we can import some argument
      // names
      revng_log(Logger, "It already has a prototype");

      auto &Old = *DynamicFunction.Prototype()->getPrototype();
      auto &New = *MaybeFunction->Prototype.getPrototype();

      if (Old.Kind() != New.Kind()) {
        revng_log(Logger, "Old and new prototype have different kind");
        continue;
      }

      if (Old.Kind() != model::TypeDefinitionKind::CABIFunctionDefinition) {
        revng_log(Logger, "Old prototype is not CABIFunctionDefinition");
        continue;
      }

      auto &OldCABI = cast<model::CABIFunctionDefinition>(Old);
      auto &NewCABI = cast<model::CABIFunctionDefinition>(New);

      if (OldCABI.Arguments().size() != NewCABI.Arguments().size()) {
        revng_log(Logger,
                  "Old and new prototype have different number of "
                  "arguments");
        continue;
      }

      unsigned NamesGiven = 0;
      for (auto &&[OldArgument, NewArgument] :
           zip(OldCABI.Arguments(), NewCABI.Arguments())) {
        if (OldArgument.Name().empty() and not NewArgument.Name().empty()) {
          OldArgument.Name() = NewArgument.Name();
          ++NamesGiven;
        }
      }

      revng_log(Logger,
                "The name of " << NamesGiven
                               << " arguments have been imported");

      continue;
    }

    revng_assert(!MaybeFunction->ModuleName.empty());
    revng_assert(MaybeFunction->Prototype.verify(true));

    using model::UpcastableTypeDefinition;
    UpcastableTypeDefinition SerializablePrototype = MaybeFunction->Prototype;
    revng_log(Logger,
              "Found type for " << DynamicFunction.Name() << " in "
                                << MaybeFunction->ModuleName << ": "
                                << toString(SerializablePrototype));
    TypeCopier &TheTypeCopier = GetOrMakeACopier(MaybeFunction->ModuleName);
    DynamicFunction.Prototype() = TheTypeCopier
                                    .copyTypeInto(MaybeFunction->Prototype);

    // Copy all the Attributes except for the inlining-related ones, which
    // dynamic functions cannot have.
    for (auto &Attribute : MaybeFunction->Attributes)
      if (Attribute != model::FunctionAttribute::AlwaysInline
          and Attribute != model::FunctionAttribute::HasOneBrokenReturn)
        DynamicFunction.Attributes().insert(Attribute);
  }

  // Finalize the copies
  for (auto &[_, TC] : TypeCopiers)
    TC->finalize();

  // Purge cached references and update the reference to Root.
  Binary.disableReferenceCaching();
  Binary.initializeReferences();

  model::flattenPrimitiveTypedefs(Binary);
  deduplicateEquivalentTypes(Binary);
  model::deduplicateCollidingNames(Binary);
}
