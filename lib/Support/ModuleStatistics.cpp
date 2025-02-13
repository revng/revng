/// \file ModuleStatistics.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/TypeFinder.h"

#include "revng/ADT/ZipMapIterator.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ModuleStatistics.h"
#include "revng/Support/Tag.h"

using namespace llvm;

static void
emitIncrement(llvm::raw_ostream &Output, unsigned NewValue, unsigned OldValue) {
  if (OldValue == 0) {
    Output << " (+Inf%)";
    return;
  }
  unsigned Increment = 100 * NewValue / OldValue;
  Output << " (";
  if (Increment >= 0)
    Output << "+";
  Output << Increment;
  Output << "%)";
}

static void emitIndentation(llvm::raw_ostream &Output, unsigned Indent) {
  for (unsigned I = 0; I < Indent; ++I)
    Output << "  ";
}

void FunctionClass::dump(llvm::raw_ostream &Output,
                         unsigned Indent,
                         const FunctionClass *Old) const {
  static const FunctionClass Empty;
  bool HasOld = Old != nullptr;
  if (not HasOld)
    Old = &Empty;

  auto Emit = [&](StringRef Name, unsigned NewValue, unsigned OldValue) {
    emitIndentation(Output, Indent);

    Output << Name.str() << ": " << NewValue;
    if (HasOld)
      emitIncrement(Output, NewValue, OldValue);
    Output << "\n";
  };

  Emit("DeclarationsCount", DeclarationsCount, Old->DeclarationsCount);
  Emit("DefinitionsCount", DefinitionsCount, Old->DefinitionsCount);

  emitIndentation(Output, Indent);
  Output << "InstructionsStatistics: ";
  Output << InstructionsStatistics.toString();
  if (HasOld) {
    emitIncrement(Output,
                  InstructionsStatistics.sum(),
                  Old->InstructionsStatistics.sum());
  }
  Output << "\n";
}

void ModuleStatistics::dump(llvm::raw_ostream &Output,
                            unsigned Indent,
                            const ModuleStatistics *Old) const {
  static const ModuleStatistics Empty;
  bool HasOld = Old != nullptr;
  if (not HasOld)
    Old = &Empty;

  auto Emit = [&](StringRef Name, unsigned NewValue, unsigned OldValue) {
    emitIndentation(Output, Indent);
    Output << Name.str() << ": " << NewValue;
    if (HasOld)
      emitIncrement(Output, NewValue, OldValue);
    Output << "\n";
  };

#define EMIT(FieldName)                          \
  do {                                           \
    Emit(#FieldName, FieldName, Old->FieldName); \
  } while (0)

  EMIT(NamedGlobalsCount);
  EMIT(AnonymousGlobalsCount);
  EMIT(AliasesCount);
  EMIT(NamedMetadataCount);
  EMIT(MaxNamedMetadataSize);

  EMIT(MaxArrayElements);
  EMIT(MaxStructElements);

  emitIndentation(Output, Indent);
  Output << "AllFunctions:\n";
  AllFunctions.dump(Output, Indent + 1, HasOld ? &Old->AllFunctions : nullptr);

  emitIndentation(Output, Indent);
  Output << "TaggedFunctions:\n";
  for (auto &&[NewEntry, OldEntry] :
       zipmap_range(TaggedFunctions, Old->TaggedFunctions)) {
    const FunctionTags::Tag *Tag = nullptr;
    static const FunctionClass Empty;
    const FunctionClass *NewClass = nullptr;
    if (NewEntry == nullptr) {
      NewClass = &Empty;
      Tag = OldEntry->first;
    } else {
      Tag = NewEntry->first;
      NewClass = &NewEntry->second;
    }

    Output << "  " << Tag->name().str() << "\n";
    const FunctionClass *Other = nullptr;
    if (HasOld and OldEntry != nullptr)
      Other = &OldEntry->second;
    NewClass->dump(Output, Indent + 2, Other);
  }

  EMIT(NamedStructsCount);
  EMIT(AnonymousStructsCount);
  EMIT(DebugCompileUnitsCount);
  EMIT(DebugScopesCount);
  EMIT(DebugSubprogramsCount);
  EMIT(DebugGloblaVariablesCount);
  EMIT(DebugTypesCount);

#undef EMIT
}

ModuleStatistics ModuleStatistics::analyze(const llvm::Module &M) {
  using namespace FunctionTags;

  ModuleStatistics Result;

  // Count global variables
  for (const GlobalVariable &Global : M.globals()) {
    if (Global.hasName())
      ++Result.NamedGlobalsCount;
    else
      ++Result.AnonymousGlobalsCount;
  }

  // Count aliases
  Result.AliasesCount = M.alias_size();

  // Count named metadata
  Result.NamedMetadataCount = M.named_metadata_size();

  // Measure size of named metadata
  for (const NamedMDNode &MD : M.named_metadata()) {
    Result.MaxNamedMetadataSize = std::max(Result.MaxNamedMetadataSize,
                                           MD.getNumOperands());
  }

  // Measure functions
  for (const Function &F : M) {
    Result.AllFunctions.process(F);

    for (const Tag *FunctionTag : TagsSet::from(&F))
      Result.TaggedFunctions[FunctionTag].process(F);
  }

  // Measure types
  {
    std::set<llvm::Type *> Types;

    // TODO: there would be much more stuff to look for

    for (const GlobalVariable &GV : M.globals())
      collectTypes(GV.getValueType(), Types);

    for (const Function &F : M)
      collectTypes(F.getType(), Types);

    for (Type *T : Types) {

      if (auto *Struct = dyn_cast<StructType>(T)) {
        if (Struct->hasName())
          ++Result.NamedStructsCount;
        else
          ++Result.AnonymousStructsCount;

        Result.MaxStructElements = std::max(Result.MaxStructElements,
                                            Struct->getNumElements());
      } else if (auto *Array = dyn_cast<ArrayType>(T)) {
        unsigned Elements = Array->getNumElements();
        Result.MaxArrayElements = std::max(Result.MaxArrayElements, Elements);
      }
    }
  }

  // Measure debug info
  {
    DebugInfoFinder DIFinder;
    DIFinder.processModule(M);
    Result.DebugCompileUnitsCount = DIFinder.compile_unit_count();
    Result.DebugScopesCount = DIFinder.scope_count();
    Result.DebugSubprogramsCount = DIFinder.subprogram_count();
    Result.DebugGloblaVariablesCount = DIFinder.global_variable_count();
    Result.DebugTypesCount = DIFinder.type_count();
  }

  return Result;
}
