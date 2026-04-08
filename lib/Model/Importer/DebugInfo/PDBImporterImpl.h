#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <deque>
#include <map>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"

#include "revng/ADT/IndexedVector.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Model/Type.h"
#include "revng/Support/Debug.h"

namespace llvm::codeview {
class SymbolVisitorCallbacks;
}

inline Logger Log("pdb-importer");

inline std::string toString(llvm::codeview::TypeIndex Index) {
  return "0x" + llvm::utohexstr(Index.getIndex(), true);
}

struct Records {
  using TypeRecord = llvm::codeview::TypeRecord;
  using TypeIndex = llvm::codeview::TypeIndex;

  template<std::derived_from<TypeRecord> T>
  using TypeList = std::deque<std::pair<TypeIndex, T>>;

  llvm::DenseMap<TypeIndex, TypeRecord *> RecordsByIndex;

  TypeList<llvm::codeview::ClassRecord> ClassRecords;
  TypeList<llvm::codeview::EnumRecord> EnumRecords;
  TypeList<llvm::codeview::UnionRecord> UnionRecords;

  TypeList<llvm::codeview::ProcedureRecord> ProcedureRecords;
  TypeList<llvm::codeview::MemberFunctionRecord> MemberFunctionRecords;

  // Note: the following don' really need the TypeIndex, we never go through
  //       them linearly.
  TypeList<llvm::codeview::ArgListRecord> ArgListRecords;
  TypeList<llvm::codeview::FieldListRecord> FieldListRecords;
  TypeList<llvm::codeview::PointerRecord> PointerRecords;
  TypeList<llvm::codeview::ModifierRecord> ModifierRecords;
  TypeList<llvm::codeview::ArrayRecord> ArrayRecords;
  TypeList<llvm::codeview::FuncIdRecord> FuncIdRecords;
  TypeList<llvm::codeview::BitFieldRecord> BitFieldRecords;
  TypeList<llvm::codeview::AliasRecord> AliasRecords;

  IndexedVector<TypeIndex, llvm::codeview::EnumeratorRecord> EnumeratorRecords;
  IndexedVector<TypeIndex, llvm::codeview::DataMemberRecord> DataMemberRecords;

  void finalize() {
    EnumeratorRecords.finalize();
    DataMemberRecords.finalize();
  }
};

struct Symbols {
  std::vector<llvm::codeview::Compile2Sym> Compile2Records;
  std::vector<llvm::codeview::Compile3Sym> Compile3Records;
  std::vector<llvm::codeview::ProcSym> ProcSymRecords;
  std::vector<llvm::codeview::UDTSym> UDTSymRecords;
};

class ProcessedTypeMap {
public:
  using TypeIndex = llvm::codeview::TypeIndex;

  struct Entry {
    model::UpcastableType Type;

    /// True if and only if calling size() on Type is guaranteed to succeed.
    bool IsSizeAvailable = false;
  };

private:
  // Note: it would be nice to use a more compact data structure such as a
  //       DenseMap but we need the pointer to the value to be stable.
  std::map<TypeIndex, Entry> Map;

public:
  Entry *tryGetEntry(TypeIndex Index) {
    auto It = Map.find(Index);
    if (It != Map.end())
      return &It->second;
    return nullptr;
  }

  model::UpcastableType *tryGetType(TypeIndex Index) {
    if (auto *E = tryGetEntry(Index))
      return &E->Type;
    return nullptr;
  }

  bool hasBeenProcessed(TypeIndex Index) const { return Map.contains(Index); }

  auto begin() const { return Map.begin(); }
  auto end() const { return Map.end(); }

public:
  Entry &
  record(TypeIndex Index, model::UpcastableType &&Type, bool SizeAvailable) {
    auto [It,
          Inserted] = Map.insert({ Index,
                                   Entry{ std::move(Type), SizeAvailable } });
    revng_assert(Inserted);
    return It->second;
  }

  void markSizeAvailable(TypeIndex Index) {
    auto It = Map.find(Index);
    revng_assert(It != Map.end());
    It->second.IsSizeAvailable = true;
  }
};

class PDBImporterImpl {
public:
  using TypeIndex = llvm::codeview::TypeIndex;

private:
  PDBImporter &Importer;

  TupleTree<model::Binary> &Model;

  llvm::pdb::InputFile *TheInput = nullptr;

  /// A copy of everything in the TPI stream
  Records TypeRecords;

  /// A copy of everything in the IPI stream
  Records FunctionRecords;

  /// A copy of everything in the DBI stream
  Symbols Symbols;

  /// A list of all the processed types
  ///
  /// \note This might contain DefinedTypes that are not fully processed yet.
  ProcessedTypeMap ProcessedTypes;

  /// A set of invalid TypeDefinition that we'll need to purge at the end
  std::set<const model::TypeDefinition *> ToDrop;

  /// A list of UDT typedefs
  ///
  /// These are used to create a typedef for each TypeIndex that is referenced
  /// from one (and only one) UDT.
  ///
  /// Note that we also produce `typedef`s via handle AliasRecord, which is a
  /// more idiomatic, albeit less used, way of representing them.
  llvm::DenseMap<TypeIndex, llvm::codeview::UDTSym *> Typedefs;

  /// Set of names of symbols that we now to be local. All the others will be
  /// considered dynamic.
  llvm::StringSet<> LocalSymbolNames;

public:
  PDBImporterImpl(PDBImporter &Importer) :
    Importer(Importer), Model(Importer.getModel()) {}

public:
  void run();

private:
  bool loadInputFile();

  void collectRecords();

  void detectArchitecture();

  void createTypedefs();

  void resolveAllForwardReferences();

  void preregisterTypeDefinitions();

  void populateTypes();

  void populateSymbolsWithTypes();

public:
  template<std::derived_from<model::TypeDefinition> T>
  void preregisterTypeDefinition(TypeIndex TypeIndex) {
    revng_log(Log,
              "Recording "
                << toString(TypeIndex) << " as "
                << model::TypeDefinitionKind::getName(T::AssociatedKind));
    registerTypeDefinition<T>(TypeIndex);
  }

  template<std::derived_from<model::TypeDefinition> T>
  std::pair<T &, const model::UpcastableType &>
  registerTypeDefinition(TypeIndex TypeIndex) {
    auto [Definition, Type] = Model->makeTypeDefinition<T>();
    const auto &CachedType = recordType(TypeIndex, std::move(Type), false);
    return { Definition, CachedType };
  }

  void registerInvalidDefinition(const model::TypeDefinition *Definition) {
    ToDrop.insert(Definition);
  }

  const model::UpcastableType &recordType(TypeIndex Index,
                                          model::UpcastableType &&Result,
                                          bool SizeAvailable);

  ProcessedTypeMap::Entry *tryGetEntry(TypeIndex Index) {
    return ProcessedTypes.tryGetEntry(Index);
  }

  model::UpcastableType *tryGetType(TypeIndex Index) {
    return ProcessedTypes.tryGetType(Index);
  }

  void markSizeAvailable(TypeIndex Index) {
    ProcessedTypes.markSizeAvailable(Index);
  }

  Records::TypeRecord *getTypeRecord(TypeIndex Index) {
    auto It = TypeRecords.RecordsByIndex.find(Index);
    if (It == TypeRecords.RecordsByIndex.end())
      return nullptr;
    return It->second;
  }

  template<typename T>
  T *getTypeRecord(TypeIndex Index) {
    auto It = TypeRecords.RecordsByIndex.find(Index);
    if (It == TypeRecords.RecordsByIndex.end())
      return nullptr;

    llvm::codeview::TypeRecordKind Kind = It->second->Kind;

    if constexpr (false) {
#define TYPE_RECORD_ALIAS(lf_ename, value, name, alias_name)
#define MEMBER_RECORD(lf_ename, value, name)
#define TYPE_RECORD(lf_ename, value, name)                              \
  }                                                                     \
  else if constexpr (std::is_same_v<T, llvm::codeview::name##Record>) { \
    if (Kind != llvm::codeview::TypeRecordKind::name)                   \
      return nullptr;
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
#undef TYPE_RECORD
#undef TYPE_RECORD_ALIAS
#undef MEMBER_RECORD
    } else {
      return nullptr;
    }

    return reinterpret_cast<T *>(It->second);
  }

  auto getDataMemberRecords(TypeIndex Index) {
    return TypeRecords.DataMemberRecords.getRange(Index);
  }

  auto getEnumeratorRecords(TypeIndex Index) {
    return TypeRecords.EnumeratorRecords.getRange(Index);
  }

private:
  void visitSymbols(llvm::codeview::SymbolVisitorCallbacks &Visitor);

  template<typename T>
  void resolveForwardReferences(llvm::StringRef Name,
                                Records::TypeList<T> &ObjectList);

  void handleProcedureSymbol(llvm::codeview::ProcSym &Procedure);
};
