#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Model/UnionDefinition.h"

#include "PDBImporterImpl.h"

class TypeResolver {
public:
  using TypeIndex = llvm::codeview::TypeIndex;
  using TypeRecord = llvm::codeview::TypeRecord;

private:
  PDBImporterImpl &Importer;
  model::Architecture::Values Architecture = model::Architecture::Invalid;

  bool NeedsSize = true;
  llvm::DenseSet<const model::TypeDefinition *> CurrentlyProcessing;

public:
  TypeResolver(PDBImporterImpl &Importer,
               model::Architecture::Values Architecture) :
    Importer(Importer), Architecture(Architecture) {}

public:
  const model::UpcastableType &getTypeFor(TypeIndex Index) {
    return *getTypeForImpl(Index);
  }

private:
  RecursiveCoroutine<const model::UpcastableType *>
  getTypeForImpl(TypeIndex Index);

private:
  const model::UpcastableType *handleSimpleType(TypeIndex SimpleType);

private:
  //
  // Type modifiers
  //
  template<typename RecordType>
  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, TypeRecord *TheType) {
    // Check cache
    if (auto *Entry = Importer.tryGetEntry(Index)) {
      if (not NeedsSize or Entry->IsSizeAvailable) {
        revng_log(Log, toString(Index) + " found in cache");
        rc_return & Entry->Type;
      }
      revng_log(Log,
                toString(Index) + " cached but size unavailable, re-resolving");
    }

    // Resolve
    auto &Record = *reinterpret_cast<RecordType *>(TheType);
    rc_return rc_recur handle(Index, Record);
  }

  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, llvm::codeview::BitFieldRecord &Pointer);

  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, llvm::codeview::PointerRecord &Pointer);

  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, llvm::codeview::ModifierRecord &Pointer);

  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, llvm::codeview::ArrayRecord &Pointer);

private:
  //
  // Type definitions
  //
  template<std::derived_from<TypeRecord> RecordType,
           std::derived_from<model::TypeDefinition> DefinitionType>
  RecursiveCoroutine<const model::UpcastableType *>
  handle(TypeIndex Index, TypeRecord *TheType) {
    using namespace llvm;

    auto &Entry = notNull(Importer.tryGetEntry(Index));

    // If we don't need the size, no need to process the type definition.
    // Also skip if we've already processed it, or if this cache entry is not
    // backed by a pre-registered TypeDefinition (e.g. forward-ref fallbacks).
    if (NeedsSize and not Entry.IsSizeAvailable) {
      // Unfortunately, no LLVM-style RTTI here
      auto &Record = *reinterpret_cast<RecordType *>(TheType);
      revng_assert(Entry.Definition != nullptr,
                   "handle<RecordType, DefinitionType> requires a "
                   "pre-registered TypeDefinition");
      auto *Definition = cast<DefinitionType>(Entry.Definition);

      // Flip size-available up front so that if processDefinition transitively
      // reaches us via a non-pointer edge, the recursion check triggers
      // instead of re-entering processing.
      Importer.markSizeAvailable(Index);

      // Detect recursion
      if (CurrentlyProcessing.contains(Definition)) {
        revng_log(Log,
                  "Recursion found. Recursion is allowed only via pointers");
        Importer.registerInvalidDefinition(Definition);
        rc_return &Entry.Type;
      }

      // Register current type to detect recursion
      CurrentlyProcessing.insert(Definition);

      // Populate this type definition
      bool Result = rc_recur processDefinition(Record, *Definition);

      // Expunge current type from currently processing
      CurrentlyProcessing.erase(Definition);

      if (not Result) {
        revng_log(Log, "Registering type definition for being purged");
        Importer.registerInvalidDefinition(Definition);
      }
    }

    rc_return &Entry.Type;
  }

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::ClassRecord &ClassRecord,
                    model::StructDefinition &Struct);

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::UnionRecord &UnionRecord,
                    model::UnionDefinition &Union);

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::EnumRecord &UnionRecord,
                    model::EnumDefinition &Union);

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::ProcedureRecord &ProcedureRecord,
                    model::CABIFunctionDefinition &Prototype);

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::MemberFunctionRecord &MemberFunctionRecord,
                    model::CABIFunctionDefinition &Prototype);

  RecursiveCoroutine<bool>
  processFunctionDefinition(llvm::codeview::CallingConvention CallingConvention,
                            TypeIndex ReturnTypeIndex,
                            TypeIndex ArgumentListIndex,
                            model::CABIFunctionDefinition &Prototype);

  RecursiveCoroutine<bool>
  processDefinition(llvm::codeview::AliasRecord &AliasRecord,
                    model::TypedefDefinition &Typedef);

private:
  const model::UpcastableType *record(TypeIndex Index,
                                      model::UpcastableType &&Result,
                                      bool IsSizeAvailable) {
    return &Importer.recordType(Index, std::move(Result), IsSizeAvailable);
  }

  const model::UpcastableType *fail(TypeIndex Index) {
    return record(Index, model::UpcastableType::empty(), true);
  }
};
