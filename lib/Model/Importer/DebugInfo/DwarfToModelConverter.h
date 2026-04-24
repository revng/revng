#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/DebugInfo/DWARF/DWARFContext.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"

class DwarfImporter;

class DwarfToModelConverter : public BinaryImporterHelper {
private:
  DwarfImporter &Importer;
  TupleTree<model::Binary> &Model;
  size_t Index;
  size_t AltIndex;
  size_t TypesWithIdentityCount;
  llvm::DWARFContext &Context;
  std::map<size_t, const model::TypeDefinition *> Placeholders;
  std::set<const model::TypeDefinition *> InvalidPrimitives;
  std::set<const llvm::DWARFDie *> InProgressDies;

public:
  DwarfToModelConverter(DwarfImporter &Importer,
                        llvm::DWARFContext &Context,
                        size_t Index,
                        size_t AltIndex,
                        uint64_t PreferredBaseAddress);

public:
  void run();

private:
  model::ABI::Values
  getABI(llvm::dwarf::CallingConvention CC = llvm::dwarf::DW_CC_normal) const;

  const model::UpcastableType &record(const llvm::DWARFDie &Die,
                                      model::UpcastableType &&Type);

  const model::UpcastableType &recordPlaceholder(const llvm::DWARFDie &Die,
                                                 model::UpcastableType &&Type);

  enum class TypeSearchResult {
    Absent,
    PlaceholderType,
    RegularType
  };

  std::pair<TypeSearchResult, model::UpcastableType>
  findType(const llvm::DWARFDie &Die);

private:
  static bool isType(llvm::dwarf::Tag Tag);

  static bool hasModelIdentity(llvm::dwarf::Tag Tag);

  void createInvalidPrimitivePlaceholder(const llvm::DWARFDie &Die);

  void createType(const llvm::DWARFDie &Die);

  void handleTypeDeclaration(const llvm::DWARFDie &Die);

  void materializeTypesWithIdentity();

  std::string getName(const llvm::DWARFDie &Die) const;

  RecursiveCoroutine<model::UpcastableType> makeType(const llvm::DWARFDie &Die);

  RecursiveCoroutine<model::UpcastableType>
  makeTypeOrVoid(const llvm::DWARFDie &Die);

  RecursiveCoroutine<void> resolveTypeWithIdentity(const llvm::DWARFDie &Die,
                                                   model::UpcastableType &Type);

  RecursiveCoroutine<model::UpcastableType>
  resolveType(const llvm::DWARFDie &Die, bool ResolveIfHasIdentity);

  void resolveAllTypes();

  model::UpcastableType
  getSubprogramPrototype(const llvm::DWARFDie &InitialDie);

  void createFunctions();

  void purgeUnresolvedPlaceholders();
};
