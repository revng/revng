#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {

class Use;
class Module;

} // end namespace llvm

namespace dla {

class LayoutTypeSystem;

bool removeInstanceBackedgesFromInheritanceLoops(LayoutTypeSystem &TS);

} // end namespace dla

uint64_t
getLoadStoreSizeFromPtrOpUse(const llvm::Module &M, const llvm::Use *U);
