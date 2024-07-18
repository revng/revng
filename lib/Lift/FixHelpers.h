#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

class VariableManager;

namespace llvm {
class Module;
}

void fixHelpers(VariableManager &Variables, llvm::Module &HelpersModule);
