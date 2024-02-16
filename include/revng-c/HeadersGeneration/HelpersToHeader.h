#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace llvm {

class raw_ostream;
class Module;

} // end namespace llvm

/// Generate a C header containing the declaration of each non-isolated
/// function in a given LLVM IR module, i.e. QEMU helpers and revng helpers,
/// whose prototype is not in the model. For helpers that return a struct, a
/// new struct type will be defined and serialized on-the-fly.
bool dumpHelpersToHeader(const llvm::Module &M, llvm::raw_ostream &Out);
