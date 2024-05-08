#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Support/MetaAddress.h"
#include "revng/ValueMaterializer/ValueMaterializer.h"

namespace llvm {
class DataLayout;
}

class JumpTargetManager;

class StaticDataMemoryOracle final : public MemoryOracle {
private:
  JumpTargetManager &JTM;
  const MetaAddress::Features &Features;
  bool IsLittleEndian = false;

public:
  StaticDataMemoryOracle(const llvm::DataLayout &DL,
                         JumpTargetManager &JTM,
                         const MetaAddress::Features &Features);
  ~StaticDataMemoryOracle() final = default;

  MaterializedValue load(uint64_t LoadAddress, unsigned LoadSize) final;
};

class ValueMaterializerPass
  : public llvm::PassInfoMixin<ValueMaterializerPass> {
private:
  StaticDataMemoryOracle &MO;
  static constexpr const char *MarkerName = "revng_avi";

public:
  ValueMaterializerPass(StaticDataMemoryOracle &MO) : MO(MO) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &);

  static llvm::Function *createMarker(llvm::Module *M) {
    using namespace llvm;
    LLVMContext &C = M->getContext();
    auto *Type = FunctionType::get(FunctionType::getVoidTy(C), {}, true);
    FunctionCallee Callee = M->getOrInsertFunction(MarkerName, Type);
    auto *Marker = cast<Function>(Callee.getCallee());
    Marker->setOnlyAccessesInaccessibleMemory();
    return Marker;
  }
};
