/// \file cache.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local includes
#include "Cache.h"

using llvm::BasicBlock;
using llvm::BinaryOperator;
using llvm::BlockAddress;
using llvm::CallInst;
using llvm::Constant;
using llvm::dyn_cast;
using llvm::Function;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::isa;
using llvm::LoadInst;
using llvm::Module;
using llvm::Optional;
using llvm::SelectInst;
using llvm::StoreInst;
using llvm::Use;
using llvm::User;

static Logger<> SaPreprocess("sa-preprocess");
Logger<> SaLog("sa");

namespace StackAnalysis {

/// \brief Check it two loads are equivalent (load from same CSV, no stores in
///        between)
static bool areEquivalent(const LoadInst *A, const LoadInst *B) {
  if (A == B)
    return true;

  const llvm::Value *Address = A->getPointerOperand();
  if (B->getPointerOperand() != Address)
    return false;

  if (not isa<GlobalVariable>(Address))
    return false;

  const BasicBlock *BB = A->getParent();
  if (B->getParent() != BB)
    return false;

  for (const Instruction &I : *BB) {
    if (&I == A) {
      break;
    } else if (&I == B) {
      std::swap(A, B);
      break;
    }
  }

  auto EndIt = B->getIterator();
  for (auto It = A->getIterator(); It != EndIt; It++)
    if (auto *Store = dyn_cast<StoreInst>(&*It))
      if (Store->getPointerOperand() == Address)
        return false;

  return true;
}

static bool mayAlias(const llvm::Value *A, const llvm::Value *B) {
  return not((isa<GlobalVariable>(A) or isa<GlobalVariable>(B)) and A != B);
}

static bool noWritesTo(const Instruction *Start,
                       const Instruction *End,
                       const llvm::Value *Address) {
  if (Start->getParent() != End->getParent())
    return false;

  for (const Instruction &I :
       llvm::make_range(Start->getIterator(), End->getIterator()))
    if (auto *Store = dyn_cast<StoreInst>(&I))
      if (mayAlias(Store->getPointerOperand(), Address))
        return false;
  return true;
}

void Cache::identifyPartialStores(const Function *F) {
  //
  // Partial store
  //
  // a = rax & 0xffff0000
  // b = 0xaaaa
  // rax = a | b

  // Look for partial stores in registers
  for (const GlobalVariable &CSV : F->getParent()->globals()) {
    for (const Use &U : CSV.uses()) {

      const auto *UserI = dyn_cast<Instruction>(U.getUser());
      if (UserI == nullptr || UserI->getParent()->getParent() != F)
        continue;

      if (const auto *Store = dyn_cast<StoreInst>(U.getUser())) {
        if (U.getOperandNo() != StoreInst::getPointerOperandIndex())
          continue;

        // We have a store
        const llvm::Value *ToStoreValue = Store->getValueOperand();
        const auto *ToStore = dyn_cast<BinaryOperator>(ToStoreValue);
        if (ToStore == nullptr || ToStore->getOpcode() != Instruction::Or)
          continue;

        // We're storing the "or" of two values, one of the two has to be the
        // same as the destination register, with optional partial suppression
        const LoadInst *LoadFromSame = nullptr;
        for (unsigned OperandIndex = 0;
             OperandIndex < ToStore->getNumOperands();
             OperandIndex++) {
          std::set<const llvm::Value *> Visited;
          bool PartialClobber = false;
          const llvm::Value *Operand = ToStore->getOperand(OperandIndex);
          while (true) {
            if (Visited.count(Operand) != 0)
              break;
            Visited.insert(Operand);

            if (const auto *TheLoad = dyn_cast<LoadInst>(Operand)) {

              // We reached a load, is it from the same CSV where we were
              // storing? Also ensure no one wrote to that CSV when we rewrite
              // the (partially clobbered) old value.
              if (TheLoad->getPointerOperand() == &CSV and PartialClobber
                  and noWritesTo(TheLoad, Store, &CSV)) {
                revng_assert(LoadFromSame == nullptr
                             or areEquivalent(LoadFromSame, TheLoad));
                LoadFromSame = TheLoad;
              }

              // In any case stop
              break;
            } else if (const auto *BinOp = dyn_cast<BinaryOperator>(Operand)) {

              // We only allow Ands with constants
              // TODO: allow shifts?
              if (BinOp->getOpcode() != Instruction::And)
                break;

              const llvm::Value *FreeOp = BinOp->getOperand(0);
              const llvm::Value *OtherOp = BinOp->getOperand(1);
              if (BinOp->isCommutative() && isa<Constant>(FreeOp))
                std::swap(FreeOp, OtherOp);

              // We have an And with a Constant, let's proceed towards the free
              // operand, in all other cases skip
              if (isa<Constant>(OtherOp)) {
                Operand = FreeOp;
                PartialClobber = true;
              } else {
                break;
              }

            } else {
              break;
            }
          }
        }

        if (LoadFromSame != nullptr)
          IdentityLoads.insert(LoadFromSame);
      }
    }
  }
}

void Cache::identifyIdentityLoads(const Function *F) {
  //
  // Identity load
  //
  // a = rax
  // rax = a
  // rax = a

  // Look for identity loads
  for (const BasicBlock &BB : *F) {
    for (const Instruction &I : BB) {
      if (const auto *Store = dyn_cast<StoreInst>(&I)) {

        const llvm::Value *StoredValue = Store->getValueOperand();
        unsigned StoreSize = StoredValue->getType()->getIntegerBitWidth();
        const llvm::Value *Address = Store->getPointerOperand();
        const llvm::Value *NextOperand = StoredValue;

        while (NextOperand != nullptr) {
          const llvm::Value *Operand = NextOperand;
          NextOperand = nullptr;

          if (auto *Load = dyn_cast<LoadInst>(Operand)) {
            if (Load->getPointerOperand() == Address
                and noWritesTo(Load, Store, Address))
              IdentityStores.insert(Store);
          } else if (auto *ZExt = dyn_cast<llvm::ZExtInst>(Operand)) {
            NextOperand = ZExt->getOperand(0);
          } else if (auto *Trunc = dyn_cast<llvm::TruncInst>(Operand)) {
            if (Trunc->getType()->getIntegerBitWidth() >= StoreSize)
              NextOperand = Trunc->getOperand(0);
          }
        }

      } else if (const auto *Load = dyn_cast<LoadInst>(&I)) {
        bool IsIdentityStore = true;
        bool AtLeastOneStore = false;
        std::set<const Use *> Visited;
        std::queue<const Use *> WorkList;

        const llvm::Value *Address = Load->getPointerOperand();

        for (const Use &TheUse : Load->uses())
          WorkList.push(&TheUse);

        while (not WorkList.empty()) {
          const Use *I = WorkList.back();
          const User *TheUser = I->getUser();
          WorkList.pop();

          // Don't visit twice the same instruction
          if (Visited.count(I) != 0) {
            IsIdentityStore = false;
            break;
          }
          Visited.insert(I);

          // We whitelist only stores to the original value and select
          // instructions
          bool Proceed = false;
          if (const auto *TheStore = dyn_cast<StoreInst>(TheUser)) {
            Proceed = (I->getOperandNo() == 0
                       and TheStore->getPointerOperand() == Address
                       and noWritesTo(Load, TheStore, Address));
            AtLeastOneStore = true;
          } else if (isa<SelectInst>(TheUser)) {
            Proceed = I->getOperandNo() != 0;
          }

          if (not Proceed) {
            IsIdentityStore = false;
            break;
          }

          for (const Use &TheUse : TheUser->uses())
            WorkList.push(&TheUse);
        }

        if (IsIdentityStore && AtLeastOneStore)
          IdentityLoads.insert(Load);
      }
    }
  }
}

// TODO: we might want to record link register as slots and call them
//       ReturnAddressSlot
void Cache::identifyLinkRegisters(const Module *M) {
  //
  // For each function call identify where the return address is being stored
  //
  Function *FunctionCallFunction = M->getFunction("function_call");

  if (not FunctionCallFunction->user_empty()) {
    std::map<GlobalVariable *, unsigned> LinkRegisterStats;
    std::map<BasicBlock *, GlobalVariable *> LinkRegistersMap;
    for (User *U : FunctionCallFunction->users()) {
      if (auto *Call = dyn_cast<CallInst>(U)) {
        revng_assert(isCallTo(Call, "function_call"));
        auto *LinkRegister = dyn_cast<GlobalVariable>(Call->getArgOperand(3));
        LinkRegisterStats[LinkRegister]++;

        // The callee might be unknown
        if (auto *BBA = dyn_cast<BlockAddress>(Call->getArgOperand(0))) {
          BasicBlock *Callee = BBA->getBasicBlock();
          revng_assert(LinkRegistersMap.count(Callee) == 0
                       || LinkRegistersMap.at(Callee) == LinkRegister);
          LinkRegistersMap[Callee] = LinkRegister;
        }
      }
    }

    // Identify a default storage for the return address (the most common one)
    if (LinkRegisterStats.size() == 1) {
      DefaultLinkRegister = LinkRegisterStats.begin()->first;
    } else {
      std::pair<GlobalVariable *, unsigned> Max = { nullptr, 0 };
      for (auto &P : LinkRegisterStats) {
        if (P.second > Max.second)
          Max = P;
      }
      revng_assert(Max.first != nullptr && Max.second != 0);
      DefaultLinkRegister = Max.first;
    }
  }
}

Cache::Cache(const Function *F) : DefaultLinkRegister(nullptr) {
  identifyPartialStores(F);
  identifyIdentityLoads(F);
  identifyLinkRegisters(F->getParent());

  // Dump the results
  if (SaPreprocess.isEnabled()) {
    SaPreprocess << "IdentityStores:\n";
    for (const StoreInst *I : IdentityStores)
      SaPreprocess << I << "\n";
    SaPreprocess << DoLog;

    SaPreprocess << "IdentityLoads:\n";
    for (const LoadInst *I : IdentityLoads)
      SaPreprocess << I << "\n";
    SaPreprocess << DoLog;

    SaPreprocess << "DefaultLinkRegister: " << DefaultLinkRegister << DoLog;
  }
}

Optional<const IntraproceduralFunctionSummary *>
Cache::get(BasicBlock *Function) const {
  auto It = Results.find(Function);
  if (It != Results.end())
    return { &It->second };

  return Optional<const IntraproceduralFunctionSummary *>();
}

bool Cache::update(BasicBlock *Function,
                   const IntraproceduralFunctionSummary &Result) {

  if (SaLog.isEnabled()) {
    SaLog << "Cache.update(" << getName(Function) << ") with value\n";
    Result.dump(getModule(Function), SaLog);
    SaLog << DoLog;
  }

  auto It = Results.find(Function);
  if (It == Results.end()) {
    Results.emplace(std::make_pair(Function, Result.copy()));
    return false;
  } else {
    auto &Summary = It->second;

    Intraprocedural::Element &Old = Summary.FinalState;
    const Intraprocedural::Element &New = Result.FinalState;

    // We should never put in the cache something more precise than what we had
    // before or the analysis might not terminate.  In any case, we will perform
    // the analysis only once most of the times. The main exception are
    // recursive function calls which will temporarily inject in the cache a
    // temporary top entry, which will be overwritten later on.
    revng_assert(New.lowerThanOrEqual(Old));

    It->second = Result.copy();

    return not Old.lowerThanOrEqual(New);
  }
}

} // namespace StackAnalysis
