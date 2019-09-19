#ifndef ADVANCEDVALUEINFOPASS_H
#define ADVANCEDVALUEINFOPASS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local includes
#include "JumpTargetManager.h"

namespace TrackedInstructionType {

enum Values { Invalid, PCStore, MemoryStore, MemoryLoad };

inline const char *getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case PCStore:
    return "PCStore";
  case MemoryStore:
    return "MemoryStore";
  case MemoryLoad:
    return "MemoryLoad";
  default:
    revng_abort();
  }
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid")
    return Invalid;
  else if (Name == "PCStore")
    return PCStore;
  else if (Name == "MemoryStore")
    return MemoryStore;
  else if (Name == "MemoryLoad")
    return MemoryLoad;
  else
    revng_abort();
}

} // namespace TrackedInstructionType

class StaticDataMemoryOracle {
private:
  const llvm::DataLayout &DL;
  JumpTargetManager &JTM;
  BinaryFile::Endianess E;

public:
  StaticDataMemoryOracle(const llvm::DataLayout &DL, JumpTargetManager &JTM) :
    DL(DL), JTM(JTM) {
    // Read the value using the endianess of the destination architecture,
    // since, if there's a mismatch, in the stack we will also have a byteswap
    // instruction
    using Endianess = BinaryFile::Endianess;
    E = (DL.isLittleEndian() ? Endianess::LittleEndian : Endianess::BigEndian);
  }

  const llvm::DataLayout &getDataLayout() const { return DL; }

  MaterializedValue load(llvm::Constant *Address) {
    return JTM.readFromPointer(Address, E);
  }
};

class AdvancedValueInfoPass
  : public llvm::PassInfoMixin<AdvancedValueInfoPass> {

public:
  AdvancedValueInfoPass(JumpTargetManager *JTM) : JTM(JTM) {}

  llvm::PreservedAnalyses
  run(llvm::Function &F, llvm::FunctionAnalysisManager &);

private:
  JumpTargetManager *JTM;
};

llvm::PreservedAnalyses
AdvancedValueInfoPass::run(llvm::Function &F,
                           llvm::FunctionAnalysisManager &FAM) {
  using namespace llvm;

  auto &LVI = FAM.getResult<LazyValueAnalysis>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  StaticDataMemoryOracle MO(F.getParent()->getDataLayout(), *JTM);
  ReversePostOrderTraversal<Function *> RPOT(&F);
  std::vector<BasicBlock *> RPOTVector;
  std::copy(RPOT.begin(), RPOT.end(), std::back_inserter(RPOTVector));
  AdvancedValueInfo<StaticDataMemoryOracle> AVI(LVI, DT, MO, RPOTVector);
  Value *PCReg = JTM->pcReg();

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (Value *V : I.operands()) {
        revng_assert(isa<PHINode>(V) or V != &I);
      }
    }
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *T = dyn_cast_or_null<MDTuple>(I.getMetadata("revng.avi."
                                                            "mark"))) {
        revng_assert(T->getNumOperands() == 1);

        auto TIT = TrackedInstructionType::Invalid;
        Value *V = nullptr;
        if (auto *Load = dyn_cast<LoadInst>(&I)) {
          if (not isa<GlobalVariable>(skipCasts(Load->getPointerOperand()))) {
            V = Load->getPointerOperand();
            TIT = TrackedInstructionType::MemoryLoad;
          }
        } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
          Value *Address = Store->getPointerOperand();
          Value *ValueOperand = Store->getValueOperand();

          if (auto *CSV = dyn_cast<GlobalVariable>(skipCasts(Address))) {
            if (CSV == PCReg) {
              V = ValueOperand;
              TIT = TrackedInstructionType::PCStore;
            }
          } else if (Address->getType() == PCReg->getType()) {
            V = ValueOperand;
            TIT = TrackedInstructionType::MemoryStore;
          }
        } else {
          revng_abort("Unexpected instruction marked with revng.avi.mark");
        }

        if (V != nullptr) {
          MaterializedValues Values = AVI.explore(I.getParent(), V);

          // If we're tracking the values of a store to pc all the (non-symbol
          // relative) possible values have to be valid program counters
          if (TIT == TrackedInstructionType::PCStore) {
            for (const MaterializedValue &V : Values) {
              if (not V.hasSymbol() and not JTM->isPC(V.value())) {
                Values.clear();
                break;
              }
            }
          }

          QuickMetadata QMD(getContext(&F));
          std::vector<Metadata *> ValuesMD;
          ValuesMD.reserve(Values.size());
          for (const MaterializedValue &V : Values) {
            if (not V.hasSymbol()) {
              ValuesMD.push_back(QMD.get(V.value()));
            }
          }

          auto TITMD = QMD.get(TrackedInstructionType::getName(TIT));
          I.setMetadata("revng.avi", QMD.tuple({ TITMD, QMD.tuple(ValuesMD) }));
        }
      }
    }
  }

  return PreservedAnalyses::none();
}

#endif // ADVANCEDVALUEINFOPASS_H
