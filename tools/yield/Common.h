#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Pipes/FunctionStringMap.h"
#include "revng/TupleTree/TupleTree.h"

namespace llvm {
class LLVMContext;
class MemoryBuffer;
class Module;
} // namespace llvm

namespace model {
class Binary;
}

struct ObjectLifetimeController {
  std::unique_ptr<llvm::LLVMContext> Context = {};
  std::unique_ptr<llvm::Module> IR = {};
  std::unique_ptr<TupleTree<model::Binary>> Model = {};
  std::unique_ptr<llvm::MemoryBuffer> Binary = {};
};

using ReturnValueType = std::tuple<const llvm::Module &,
                                   const TupleTree<model::Binary> &,
                                   RawBinaryView,
                                   revng::pipes::FunctionStringMap,
                                   ObjectLifetimeController>;
ReturnValueType parseCommandLineOptions(int Argc, char *Argv[]);
