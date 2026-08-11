//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Importer/ImportPrototypesFromDatabase.h"
#include "revng/Model/Importer/PrototypeMatching.h"
#include "revng/Model/Importer/TypeCopier.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/SQLite.h"

static Logger Log("import-prototypes-from-db");

/// Turn a TypeDefinition body (flat YAML mapping) into a YAML list item
/// indented for inclusion under a top-level key.
///
/// Input (each line at column 0):
///   ID: 123
///   Kind: CABIFunctionDefinition
///   ...
///
/// Output:
///   - ID: 123
///     Kind: CABIFunctionDefinition
///     ...
static std::string indentAsListItem(llvm::StringRef Body) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);

  bool FirstLine = true;
  while (not Body.empty()) {
    auto [Line, Rest] = Body.split('\n');
    if (FirstLine) {
      Stream << "  - " << Line << "\n";
      FirstLine = false;
    } else if (not Line.empty()) {
      Stream << "    " << Line << "\n";
    }
    Body = Rest;
    if (Body.empty() && Line.empty())
      break;
  }

  return Result;
}

/// Prepend \p Prefix to each non-empty line of \p Text.
static std::string addLinePrefix(llvm::StringRef Prefix, llvm::StringRef Text) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);

  while (not Text.empty()) {
    auto [Line, Rest] = Text.split('\n');
    if (not Line.empty())
      Stream << Prefix << Line << "\n";
    Text = Rest;
  }

  return Result;
}

struct LibraryInfo {
  int64_t LibraryID;
  std::string Header;
};

// model::Function is queried by its ExportedNames (its dynamic-symbol names).
// Name is a local identifier and is never used as a lookup key.
inline llvm::SmallVector<llvm::StringRef, 4>
lookupNames(const model::Function &Function) {
  llvm::SmallVector<llvm::StringRef, 4> Names;
  for (const auto &Name : Function.ExportedNames())
    Names.emplace_back(Name);
  return Names;
}

inline llvm::SmallVector<llvm::StringRef, 4>
lookupNames(const model::DynamicFunction &Function) {
  if (Function.Name().empty())
    return {};
  return { llvm::StringRef(Function.Name()) };
}

class PrototypeDatabase {
private:
  sqlite::Database Database;

public:
  PrototypeDatabase(llvm::StringRef Path) :
    Database(Path, sqlite::Database::OpenMode::ReadOnly) {}

  /// Find a platform by its exact name.
  /// Returns -1 if no platform with that name exists.
  int64_t findPlatformByName(llvm::StringRef PlatformName) {
    auto Statement = Database.makeStatement("SELECT PlatformID FROM Platform "
                                            "WHERE Name = ?1");
    Statement.bind(1, PlatformName);

    int64_t PlatformID = -1;
    for (auto [ID] : Statement.execute<int64_t>())
      PlatformID = ID;

    return PlatformID;
  }

  /// Find the platform that resolves the most symbols among those compatible
  /// with the given architecture and operating system.
  /// Returns -1 if no matching platform is found.
  int64_t electPlatform(const std::vector<std::string> &SymbolNames,
                        llvm::StringRef ArchitectureName,
                        llvm::StringRef OperatingSystemName) {
    std::string SymbolPlaceholders = sqlite::buildInClause(SymbolNames.size());
    int ArchitectureParamIndex = SymbolNames.size() + 1;
    int OperatingSystemParamIndex = SymbolNames.size() + 2;

    std::string Query;
    {
      llvm::raw_string_ostream Stream(Query);
      Stream << R"(
        SELECT p.PlatformID, COUNT(DISTINCT s.Name) AS SymbolCount
        FROM Platform p
        JOIN Library l ON l.PlatformID = p.PlatformID
        JOIN Symbol s ON s.LibraryID = l.LibraryID
        WHERE p.Architecture = ?)"
             << ArchitectureParamIndex << R"(
          AND p.OperatingSystem = ?)"
             << OperatingSystemParamIndex << R"(
          AND s.Name IN ()"
             << SymbolPlaceholders << R"()
        GROUP BY p.PlatformID
        ORDER BY SymbolCount DESC
        LIMIT 1
      )";
    }

    auto Statement = Database.makeStatement(Query);
    Statement.bind(SymbolNames);
    Statement.bind(ArchitectureParamIndex, ArchitectureName);
    Statement.bind(OperatingSystemParamIndex, OperatingSystemName);

    int64_t PlatformID = -1;
    for (auto [ID, Count] : Statement.execute<int64_t, int64_t>())
      PlatformID = ID;

    return PlatformID;
  }

  /// Enumerate libraries on the given platform that provide at least one of
  /// the requested symbols.
  std::vector<LibraryInfo>
  enumerateLibraries(int64_t PlatformID,
                     const std::vector<std::string> &SymbolNames) {
    std::string SymbolPlaceholders = sqlite::buildInClause(SymbolNames.size());
    int PlatformParamIndex = SymbolNames.size() + 1;

    std::string Query;
    {
      llvm::raw_string_ostream Stream(Query);
      Stream << R"(
        SELECT DISTINCT l.LibraryID, l.Header
        FROM Library l
        JOIN Symbol s ON s.LibraryID = l.LibraryID
        WHERE l.PlatformID = ?)"
             << PlatformParamIndex << R"(
          AND s.Name IN ()"
             << SymbolPlaceholders << R"()
      )";
    }

    auto Statement = Database.makeStatement(Query);
    Statement.bind(SymbolNames);
    Statement.bind(PlatformParamIndex, static_cast<int>(PlatformID));

    std::vector<LibraryInfo> Libraries;
    for (auto [LibraryID, HeaderText] :
         Statement.execute<int64_t, llvm::StringRef>())
      Libraries.push_back({ LibraryID, HeaderText.str() });

    return Libraries;
  }

  /// Build a model from a single library's symbols and type definitions,
  /// parse it, then use a TypeCopier to import the prototypes into Binary.
  ///
  /// OriginalIDs in type definition bodies are scoped per-library, so each
  /// library must be imported independently.
  void importLibrary(const LibraryInfo &Library,
                     const std::vector<std::string> &SymbolNames,
                     TupleTree<model::Binary> &Binary) {
    std::string SymbolPlaceholders = sqlite::buildInClause(SymbolNames.size());
    int LibraryParamIndex = SymbolNames.size() + 1;

    // Query type definitions for this library
    std::string TypeDefinitionsQuery;
    {
      llvm::raw_string_ostream Stream(TypeDefinitionsQuery);
      Stream << R"(
        WITH RECURSIVE DependentTypeDefinitions AS ( --
          SELECT td.TypeDefinitionID, td.Body, td.OriginalID
          FROM TypeDefinition td
          JOIN Symbol s ON s.TypeDefinitionID = td.TypeDefinitionID
          WHERE s.LibraryID = ?)"
             << LibraryParamIndex << R"(
            AND s.Name IN ()"
             << SymbolPlaceholders << R"()
          UNION
          SELECT t.TypeDefinitionID, t.Body, t.OriginalID
          FROM TypeDefinition t
          JOIN TypeDefinitionDependencies dependency
            ON dependency.SourceTypeDefinitionID = t.TypeDefinitionID
          JOIN DependentTypeDefinitions dt
            ON dependency.DestinationTypeDefinitionID = dt.TypeDefinitionID
        )
        SELECT DISTINCT Body, OriginalID
        FROM DependentTypeDefinitions
        ORDER BY OriginalID
      )";
    }

    auto TypeDefinitionsStatement = Database
                                      .makeStatement(TypeDefinitionsQuery);
    TypeDefinitionsStatement.bind(SymbolNames);
    TypeDefinitionsStatement.bind(LibraryParamIndex,
                                  static_cast<int>(Library.LibraryID));

    struct TypeInfo {
      std::string Body;
      int64_t OriginalID;
      std::string Kind;
    };

    std::vector<TypeInfo> TypeDefinitions;
    std::map<int64_t, std::string> OriginalIDToKind;

    for (auto [Body, OriginalID] :
         TypeDefinitionsStatement.execute<llvm::StringRef, int64_t>()) {
      std::string BodyString = Body.str();

      // Extract Kind from the YAML body
      std::string Kind;
      llvm::StringRef BodyReference(BodyString);
      while (not BodyReference.empty()) {
        auto [Line, Rest] = BodyReference.split('\n');
        if (Line.starts_with("Kind: ")) {
          Kind = Line.substr(6).str();
          break;
        }
        BodyReference = Rest;
      }

      OriginalIDToKind[OriginalID] = Kind;
      TypeDefinitions.push_back({ std::move(BodyString), OriginalID, Kind });
    }

    revng_log(Log, "Found " << TypeDefinitions.size() << " type definitions");

    // Query symbols for this library
    std::string SymbolsQuery;
    {
      llvm::raw_string_ostream Stream(SymbolsQuery);
      Stream << R"(
        SELECT s.Name, COALESCE(td.OriginalID, -1), s.Body
        FROM Symbol s
        LEFT JOIN TypeDefinition td
          ON s.TypeDefinitionID = td.TypeDefinitionID
        WHERE s.LibraryID = ?)"
             << LibraryParamIndex << R"(
          AND s.Name IN ()"
             << SymbolPlaceholders << R"()
      )";
    }

    auto SymbolsStatement = Database.makeStatement(SymbolsQuery);
    SymbolsStatement.bind(SymbolNames);
    SymbolsStatement.bind(LibraryParamIndex,
                          static_cast<int>(Library.LibraryID));

    struct SymbolInfo {
      std::string Name;
      int64_t OriginalID;
      bool HasType;
      std::string Body;
    };

    std::vector<SymbolInfo> Symbols;
    using llvm::StringRef;
    for (auto [Name, OriginalID, Body] :
         SymbolsStatement.execute<StringRef, int64_t, StringRef>()) {
      bool HasType = OriginalID >= 0 and OriginalIDToKind.count(OriginalID) > 0;
      Symbols.push_back({ Name.str(), OriginalID, HasType, Body.str() });
    }

    if (Symbols.empty()) {
      revng_log(Log, "No matching symbols found");
      return;
    }

    revng_log(Log, "Found " << Symbols.size() << " matching symbols");

    // Compose the model YAML for this library
    std::string ModelYAML;
    llvm::raw_string_ostream YAMLStream(ModelYAML);

    YAMLStream << "---\n";

    llvm::StringRef HeaderReference(Library.Header);
    HeaderReference = HeaderReference.rtrim();
    YAMLStream << HeaderReference << "\n";

    YAMLStream << "ImportedDynamicFunctions:\n";
    for (const auto &Symbol : Symbols) {
      YAMLStream << "  - Name:            " << Symbol.Name << "\n";
      if (Symbol.HasType) {
        auto KindIterator = OriginalIDToKind.find(Symbol.OriginalID);
        if (KindIterator != OriginalIDToKind.end()) {
          YAMLStream << "    Prototype:\n";
          YAMLStream << "      Kind:            DefinedType\n";
          YAMLStream << "      Definition:      \"/TypeDefinitions/"
                     << Symbol.OriginalID << "-" << KindIterator->second
                     << "\"\n";
        }
      }

      // The fields with no dedicated column (Comment, Attributes) travel as a
      // YAML mapping in Symbol.Body.
      if (not Symbol.Body.empty())
        YAMLStream << addLinePrefix("    ", Symbol.Body);
    }

    if (not TypeDefinitions.empty()) {
      YAMLStream << "TypeDefinitions:\n";
      for (const auto &TypeDefinition : TypeDefinitions)
        YAMLStream << indentAsListItem(TypeDefinition.Body);
    }

    YAMLStream << "...\n";
    YAMLStream.flush();

    // Parse the composed YAML into a model
    auto MaybeModel = TupleTree<model::Binary>::fromString(ModelYAML);
    if (not MaybeModel) {
      revng_log(Log, "Failed to parse composed model YAML");
      llvm::consumeError(MaybeModel.takeError());
      return;
    }

    // Use TypeCopier to import prototypes into the destination model.
    // The copier is fully created, used, and finalized within this scope.
    TupleTree<model::Binary> &ParsedModel = *MaybeModel;
    TypeCopier Copier(ParsedModel, Binary);

    unsigned ImportedCount = 0;

    auto ImportSymbol = [&](auto &Function) {
      auto &Source = ParsedModel->ImportedDynamicFunctions();
      for (llvm::StringRef Name : lookupNames(Function)) {
        auto Match = findPrototypeInDynamicFunctions(Source, Name, {});
        if (not Match.has_value())
          continue;

        // Only the prototype is guarded: an existing one comes from the user or
        // from a more precise importer and must not be overwritten. Attributes
        // and the comment are additive, so they are applied either way --
        // guarding them on the prototype too would mean a prototype imported
        // earlier silently suppresses them.
        if (Function.prototype() == nullptr) {
          Function.Prototype() = Copier.copyTypeInto(Match->Prototype);
          ++ImportedCount;
          revng_log(Log, "Imported prototype for " << Name);
        }

        for (const auto &Attribute : Match->Attributes) {
          if (Function.Attributes().contains(Attribute))
            continue;

          Function.Attributes().insert(Attribute);
          revng_log(Log,
                    "Imported attribute "
                      << model::FunctionAttribute::getName(Attribute).str()
                      << " for " << Name);
        }

        if (Function.Comment().empty() and not Match->Comment.empty()) {
          Function.Comment() = Match->Comment.str();
          revng_log(Log, "Imported comment for " << Name);
        }

        return;
      }
    };

    for (auto &Function : Binary->Functions())
      ImportSymbol(Function);

    for (auto &DynamicFunction : Binary->ImportedDynamicFunctions())
      ImportSymbol(DynamicFunction);

    Copier.finalize();

    revng_log(Log, "Imported " << ImportedCount << " prototypes");
  }

  /// Main entry point: find the database, collect symbols without prototypes,
  /// elect a platform, and import prototypes from each library.
  static void run(TupleTree<model::Binary> &Binary) {
    revng_log(Log, "Starting prototype import from database");
    LoggerIndent Indent(Log);

    // Find the database
    using revng::ResourceFinder;
    auto MaybeDbPath = ResourceFinder.findFile("share/revng/prototypes.sqlite");
    if (not MaybeDbPath) {
      revng_log(Log, "Database not found, skipping");
      return;
    }

    revng_log(Log, "Using database: " << *MaybeDbPath);

    PrototypeDatabase Database(*MaybeDbPath);

    std::vector<std::string> SymbolNames;
    // Collect symbol names from functions without a prototype
    auto PushNames = [&](const auto &Function) {
      if (Function.prototype() != nullptr)
        return;
      for (llvm::StringRef Name : lookupNames(Function))
        SymbolNames.emplace_back(Name);
    };

    for (const auto &Function : Binary->Functions())
      PushNames(Function);

    for (const auto &DynamicFunction : Binary->ImportedDynamicFunctions())
      PushNames(DynamicFunction);

    if (SymbolNames.empty()) {
      revng_log(Log, "No functions without prototypes, skipping");
      return;
    }

    revng_log(Log,
              "Collected " << SymbolNames.size()
                           << " symbols without prototypes");

    // Phase 1: elect the best platform.
    // If PlatformName is set, try to find it by name first.
    int64_t PlatformID = -1;

    if (not Binary->PlatformName().empty()) {
      revng_log(Log, "Looking up platform by name: " << Binary->PlatformName());
      PlatformID = Database.findPlatformByName(Binary->PlatformName());
    }

    if (PlatformID < 0) {
      using namespace model;
      auto ArchitectureName = Architecture::getName(Binary->Architecture());
      auto OperatingSystem = Binary->OperatingSystem();
      auto OperatingSystemName = OperatingSystem::getName(OperatingSystem);
      revng_log(Log,
                "Electing platform for " << ArchitectureName << "/"
                                         << OperatingSystemName);
      PlatformID = Database.electPlatform(SymbolNames,
                                          ArchitectureName,
                                          OperatingSystemName);
    }

    if (PlatformID < 0) {
      revng_log(Log, "No matching platform found, skipping");
      return;
    }

    revng_log(Log, "Selected platform ID: " << PlatformID);

    // Phase 2: enumerate libraries with matching symbols
    auto Libraries = Database.enumerateLibraries(PlatformID, SymbolNames);
    if (Libraries.empty()) {
      revng_log(Log, "No libraries with matching symbols found");
      return;
    }

    revng_log(Log, "Found " << Libraries.size() << " libraries to import");

    // Phase 3: import each library independently
    for (const auto &Library : Libraries) {
      revng_log(Log, "Importing library " << Library.LibraryID);
      LoggerIndent LibraryIndent(Log);
      Database.importLibrary(Library, SymbolNames, Binary);
    }

    model::flattenPrimitiveTypedefs(Binary);
    deduplicateEquivalentTypes(Binary);
    model::deduplicateCollidingNames(Binary);

    revng_log(Log, "Import complete");
  }
};

namespace revng::pypeline::analyses {

llvm::Error ImportPrototypesFromDatabase::run(Model &TheModel,
                                              const Request &Incoming,
                                              llvm::StringRef Configuration) {
  PrototypeDatabase::run(TheModel.get());
  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
