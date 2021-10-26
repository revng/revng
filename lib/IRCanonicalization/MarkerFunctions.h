#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {

class Type;
class Function;

} // end namespace llvm

extern std::string makeMarkerName(const llvm::Type *Ty);
extern std::string makeModelGEPName(const llvm::Type *Ty);
extern bool isMarker(const llvm::Function &F);
extern bool isModelGEP(const llvm::Function &F);
