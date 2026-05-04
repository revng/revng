#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace llvm {
class Module;
}

namespace ptml {

class ModelCBuilder;

bool printHelpersHeader(ptml::ModelCBuilder &B, const llvm::Module &Module);

} // namespace ptml
