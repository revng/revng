#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"

namespace llvm {

class InsertValueInst;
class Instruction;
class ExtractValueInst;
class Use;
class Value;

} // end namespace llvm

extern llvm::SmallVector<llvm::Value *, 2>
getInsertValueLeafOperands(llvm::InsertValueInst *);

extern llvm::SmallVector<const llvm::Value *, 2>
getInsertValueLeafOperands(const llvm::InsertValueInst *);

extern llvm::SmallVector<llvm::SmallPtrSet<llvm::ExtractValueInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *);

extern llvm::SmallVector<llvm::SmallPtrSet<const llvm::ExtractValueInst *, 2>,
                         2>
getExtractedValuesFromInstruction(const llvm::Instruction *);

namespace dla {

class LayoutTypeSystem;

bool removeInstanceBackedgesFromInheritanceLoops(LayoutTypeSystem &TS);

} // end namespace dla

uint64_t
getLoadStoreSizeFromPtrOpUse(const llvm::Module &M, const llvm::Use *U);
