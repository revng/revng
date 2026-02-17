//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/Sqlite3.h"

namespace cl = llvm::cl;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> DatabasePath("db",
                                         cl::Required,
                                         cl::desc("Path to the SQLite "
                                                  "database"),
                                         cl::cat(ThisToolCategory));

static cl::opt<std::string> PlatformName("platform",
                                         cl::Required,
                                         cl::desc("Platform name"),
                                         cl::cat(ThisToolCategory));

static cl::opt<std::string> LibraryName("library",
                                        cl::Required,
                                        cl::desc("Library name"),
                                        cl::cat(ThisToolCategory));

static cl::list<std::string> SymbolNames("symbol",
                                         cl::desc("Symbol to export "
                                                  "(repeatable)"),
                                         cl::OneOrMore,
                                         cl::cat(ThisToolCategory));

static cl::opt<std::string> OutputFilename("o",
                                           cl::init("-"),
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(ThisToolCategory));

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
  while (!Body.empty()) {
    auto [Line, Rest] = Body.split('\n');
    if (FirstLine) {
      Stream << "  - " << Line << "\n";
      FirstLine = false;
    } else if (!Line.empty()) {
      Stream << "    " << Line << "\n";
    }
    Body = Rest;
    if (Body.empty() && Line.empty())
      break;
  }

  return Result;
}

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  llvm::ExitOnError ExitOnError;

  Sqlite3Db Database(DatabasePath);

  // 1. Fetch the library header
  auto HeaderStatement = Database.makeStatement(R"(
    SELECT l.Header
    FROM Library l
    JOIN Platform p ON l.PlatformID = p.PlatformID
    WHERE l.Name = ?1 AND p.Name = ?2
  )");
  HeaderStatement.bind(1, llvm::StringRef(LibraryName));
  HeaderStatement.bind(2, llvm::StringRef(PlatformName));

  std::string Header;
  bool Found = false;
  for (auto [HeaderText] :
       HeaderStatement.execute<llvm::StringRef>()) {
    Header = HeaderText.str();
    Found = true;
  }

  if (!Found) {
    dbg << "No matching library/platform found.\n";
    return EXIT_FAILURE;
  }

  // 2. Build the recursive CTE query for type definitions.
  //    We need to build the IN clause dynamically.
  std::string SymbolPlaceholders;
  for (size_t I = 0; I < SymbolNames.size(); ++I) {
    if (I > 0)
      SymbolPlaceholders += ",";
    SymbolPlaceholders += "?" + std::to_string(I + 1);
  }

  int LibraryParamIndex = SymbolNames.size() + 1;
  int PlatformParamIndex = SymbolNames.size() + 2;

  std::string TypeDefsSQL;
  {
    llvm::raw_string_ostream SQL(TypeDefsSQL);
    SQL << R"(
      WITH RECURSIVE DependentTypeDefinitions AS (
        SELECT td.TypeDefinitionID, td.Body, td.OriginalID
        FROM TypeDefinition td
        JOIN Symbol s ON s.TypeDefinitionID = td.TypeDefinitionID
        JOIN Library l ON s.LibraryID = l.LibraryID
        JOIN Platform p ON l.PlatformID = p.PlatformID
        WHERE s.Name IN ()" << SymbolPlaceholders << R"()
          AND l.Name = ?)" << LibraryParamIndex << R"(
          AND p.Name = ?)" << PlatformParamIndex << R"(
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

  auto TypeDefsStatement = Database.makeStatement(TypeDefsSQL);
  for (size_t I = 0; I < SymbolNames.size(); ++I)
    TypeDefsStatement.bind(I + 1, llvm::StringRef(SymbolNames[I]));
  TypeDefsStatement.bind(LibraryParamIndex, llvm::StringRef(LibraryName));
  TypeDefsStatement.bind(PlatformParamIndex, llvm::StringRef(PlatformName));

  // We need to extract the Kind from each body to build function references.
  // The Kind is on a line like "Kind: CABIFunctionDefinition".
  struct TypeInfo {
    std::string Body;
    int64_t OriginalID;
    std::string Kind;
  };

  std::vector<TypeInfo> TypeDefinitions;
  std::map<int64_t, std::string> OriginalIDToKind;

  for (auto [Body, OriginalID] :
       TypeDefsStatement.execute<llvm::StringRef, int64_t>()) {
    std::string BodyStr = Body.str();

    // Extract Kind from the YAML body
    std::string Kind;
    llvm::StringRef BodyRef(BodyStr);
    while (!BodyRef.empty()) {
      auto [Line, Rest] = BodyRef.split('\n');
      if (Line.starts_with("Kind: ")) {
        Kind = Line.substr(6).str();
        break;
      }
      BodyRef = Rest;
    }

    OriginalIDToKind[OriginalID] = Kind;
    TypeDefinitions.push_back({ std::move(BodyStr), OriginalID, Kind });
  }

  // 3. Fetch symbol rows
  std::string SymbolsSQL;
  {
    llvm::raw_string_ostream SQL(SymbolsSQL);
    SQL << R"(
      SELECT s.Name, COALESCE(td.OriginalID, -1)
      FROM Symbol s
      JOIN Library l ON s.LibraryID = l.LibraryID
      JOIN Platform p ON l.PlatformID = p.PlatformID
      LEFT JOIN TypeDefinition td
        ON s.TypeDefinitionID = td.TypeDefinitionID
      WHERE s.Name IN ()" << SymbolPlaceholders << R"()
        AND l.Name = ?)" << LibraryParamIndex << R"(
        AND p.Name = ?)" << PlatformParamIndex;
  }

  auto SymbolsStatement = Database.makeStatement(SymbolsSQL);
  for (size_t I = 0; I < SymbolNames.size(); ++I)
    SymbolsStatement.bind(I + 1, llvm::StringRef(SymbolNames[I]));
  SymbolsStatement.bind(LibraryParamIndex, llvm::StringRef(LibraryName));
  SymbolsStatement.bind(PlatformParamIndex, llvm::StringRef(PlatformName));

  struct SymbolInfo {
    std::string Name;
    int64_t OriginalID; // -1 if no type
    bool HasType;
  };

  std::vector<SymbolInfo> Symbols;
  for (auto [Name, OriginalID] :
       SymbolsStatement.execute<llvm::StringRef, int64_t>()) {
    bool HasType = OriginalID >= 0 && OriginalIDToKind.count(OriginalID) > 0;
    Symbols.push_back({ Name.str(), OriginalID, HasType });
  }

  // 4. Compose the model YAML via string concatenation
  std::string ModelYAML;
  llvm::raw_string_ostream YAMLStream(ModelYAML);

  YAMLStream << "---\n";

  // Strip trailing whitespace from header
  llvm::StringRef HeaderRef(Header);
  HeaderRef = HeaderRef.rtrim();
  YAMLStream << HeaderRef << "\n";

  // ImportedDynamicFunctions
  if (!Symbols.empty()) {
    YAMLStream << "ImportedDynamicFunctions:\n";
    for (const auto &Symbol : Symbols) {
      YAMLStream << "  - Name:            " << Symbol.Name << "\n";
      if (Symbol.HasType) {
        auto KindIt = OriginalIDToKind.find(Symbol.OriginalID);
        if (KindIt != OriginalIDToKind.end()) {
          YAMLStream << "    Prototype:\n";
          YAMLStream << "      Kind:            DefinedType\n";
          YAMLStream << "      Definition:      \"/TypeDefinitions/"
                     << Symbol.OriginalID << "-" << KindIt->second << "\"\n";
        }
      }
    }
  }

  // TypeDefinitions
  if (!TypeDefinitions.empty()) {
    YAMLStream << "TypeDefinitions:\n";
    for (const auto &TypeDef : TypeDefinitions) {
      YAMLStream << indentAsListItem(TypeDef.Body);
    }
  }

  YAMLStream << "...\n";
  YAMLStream.flush();

  // 5. Parse via TupleTree
  auto MaybeModel = TupleTree<model::Binary>::fromString(ModelYAML);
  if (!MaybeModel) {
    dbg << "Failed to parse composed model YAML.\n";

    // Dump the YAML for debugging
    dbg << "--- Composed YAML ---\n" << ModelYAML << "\n";

    llvm::consumeError(MaybeModel.takeError());
    return EXIT_FAILURE;
  }

  // 6. Verify
  if (!MaybeModel->verify()) {
    dbg << "Warning: model verification failed.\n";
  }

  // 7. Dump
  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  llvm::sys::fs::OpenFlags::OF_Text);
  if (EC) {
    ExitOnError(llvm::createStringError(EC, EC.message()));
  }

  MaybeModel->serialize(OutputFile.os());
  OutputFile.keep();

  return EXIT_SUCCESS;
}
