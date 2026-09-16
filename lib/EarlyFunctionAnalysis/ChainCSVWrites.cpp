/// Make the second write of a value to a CSV read it back from the first.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"

#include "revng/EarlyFunctionAnalysis/ChainCSVWrites.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log("chain-csv-writes");

namespace {

/// What the CSVs hold at a point of a basic block.
///
/// A value reaches two CSVs because one instruction wrote them both, so a pair
/// never crosses a basic block and one of these covers a single block.
class CSVContents {
private:
  /// A CSV and the value it holds.
  struct Entry {
    GlobalVariable *CSV = nullptr;
    Value *Value = nullptr;
  };

  /// In the order the block wrote them.
  SmallVector<Entry, 8> Entries;

public:
  /// \return the CSV holding \p V, or `nullptr` if none of them does.
  GlobalVariable *holderOf(const Value *V) const {
    for (const Entry &E : Entries)
      if (E.Value == V)
        return E.CSV;
    return nullptr;
  }

  /// Take note of \p CSV holding \p V.
  ///
  /// The CSV that got a value first is the one to read it back from, so this
  /// records nothing when another CSV is holding \p V already.
  void record(GlobalVariable *CSV, Value *V) {
    if (holderOf(V) == nullptr)
      Entries.push_back({ .CSV = CSV, .Value = V });
  }

  /// Take note of whatever \p CSV used to hold being gone.
  void forget(GlobalVariable *CSV) {
    llvm::erase_if(Entries, [CSV](const Entry &E) { return E.CSV == CSV; });
  }

  /// Take note of every CSV possibly no longer holding what it did.
  void forgetAll() { Entries.clear(); }
};

} // namespace

/// \return the CSV \p Store writes to, whole or in part, or `nullptr`.
static GlobalVariable *writtenCSV(StoreInst &Store) {
  return dyn_cast<GlobalVariable>(skipCasts(Store.getPointerOperand()));
}

/// \return whether \p Store leaves \p CSV holding exactly the value it stores.
static bool writesWhole(StoreInst &Store, GlobalVariable *CSV) {
  // A store through a cast can write something narrower than the CSV holds, and
  // reading the CSV back would then hand over more than was written.
  return CSV->getValueType() == Store.getValueOperand()->getType();
}

llvm::PreservedAnalyses
ChainCSVWritesPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  revng::IRBuilder Builder(F.getContext());
  bool Changed = false;

  for (BasicBlock &BB : F) {
    CSVContents Held;

    for (Instruction &I : BB) {
      auto *Store = dyn_cast<StoreInst>(&I);
      GlobalVariable *CSV = Store == nullptr ? nullptr : writtenCSV(*Store);

      if (CSV == nullptr or not Store->isSimple()) {
        // Where this lands is not known, so nothing recorded survives it.
        if (I.mayWriteToMemory())
          Held.forgetAll();
        continue;
      }

      Value *Stored = Store->getValueOperand();
      bool Whole = writesWhole(*Store, CSV);

      if (GlobalVariable *Source = Held.holderOf(Stored);
          Source != nullptr and Source != CSV and Whole) {
        Builder.SetInsertPoint(Store, Store->getDebugLoc());
        LoadInst *Load = Builder.createLoad(Source);
        Store->setOperand(0, Load);
        Loads.emplace_back(Load);
        Changed = true;

        revng_log(Log,
                  "In " << F.getName().str() << ", made the write to "
                        << CSV->getName().str() << " read "
                        << Source->getName().str() << " back");
      }

      Held.forget(CSV);

      if (Whole)
        Held.record(CSV, Stored);
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
