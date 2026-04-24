//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"

#include "revng/Model/Binary.h"
#include "revng/Model/BinaryIdentifier.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/LDDTree.h"

#include "FindMissingTypes.h"
template<IsObjectFile T>
std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const T &ObjectFile,
                                           llvm::StringRef CanonicalPath) {
  LDDTree::SymbolSet MissingSymbols;
  for (auto &Function : Binary->ImportedDynamicFunctions())
    if (Function.Prototype().isEmpty() and Function.Name().size() > 0)
      MissingSymbols.insert(Function.Name());

  return LDDTree::fromPath(CanonicalPath, ObjectFile, MissingSymbols);
}

using namespace llvm::object;

template std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const COFFObjectFile &ObjectFile,
                                           llvm::StringRef CanonicalPath);

template std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const ELF32LEObjectFile &ObjectFile,
                                           llvm::StringRef CanonicalPath);

template std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const ELF64LEObjectFile &ObjectFile,
                                           llvm::StringRef CanonicalPath);

template std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const ELF32BEObjectFile &ObjectFile,
                                           llvm::StringRef CanonicalPath);

template std::optional<LDDTree>
BinaryImporterHelper::identifyDependencies(const ELF64BEObjectFile &ObjectFile,
                                           llvm::StringRef CanonicalPath);
