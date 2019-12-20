#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

namespace llvm {

class InsertValueInst;
class CallInst;
class Use;
class Value;

} // end namespace llvm

namespace dla {

class LayoutTypeSystem;

} // end namespace dla

extern llvm::SmallVector<llvm::Value *, 2>
getInsertValueLeafOperands(llvm::InsertValueInst *);

extern llvm::SmallVector<const llvm::Value *, 2>
getInsertValueLeafOperands(const llvm::InsertValueInst *);

extern llvm::SmallVector<llvm::Value *, 2>
getExtractedValuesFromCall(llvm::CallInst *);

extern llvm::SmallVector<const llvm::Value *, 2>
getExtractedValuesFromCall(const llvm::CallInst *);

uint64_t getLoadStoreSizeFromPtrOpUse(const dla::LayoutTypeSystem &TS,
                                      const llvm::Use *U);
