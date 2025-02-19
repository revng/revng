#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

public:
  ValueMaterializerPass(StaticDataMemoryOracle &MO) : MO(MO) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &);

  static llvm::Function *createMarker(llvm::Module &M);
};
