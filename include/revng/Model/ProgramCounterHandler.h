#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include "revng/Model/Architecture.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"

namespace revng {
class IRBuilder;
} // namespace revng

inline llvm::IntegerType *getCSVType(llvm::GlobalVariable *CSV) {
  using namespace llvm;
  return cast<IntegerType>(CSV->getValueType());
}

namespace NextJumpTarget {

enum Values {
  Unique,
  Multiple,
  Helper
};

}; // namespace NextJumpTarget

namespace PCAffectingCSV {

enum Values {
  PC,
  IsThumb
};

}; // namespace PCAffectingCSV

namespace revng::detail {

using namespace llvm;

using CSVFactory = std::function<
  GlobalVariable *(PCAffectingCSV::Values CSVID)>;

}; // namespace revng::detail

using CSVFactory = revng::detail::CSVFactory;

class ProgramCounterHandler {
protected:
  static constexpr const char *AddressSpaceName = "pc_address_space";
  static constexpr const char *EpochName = "pc_epoch";
  static constexpr const char *TypeName = "pc_type";

protected:
  unsigned Alignment;
  llvm::GlobalVariable *AddressCSV = nullptr;
  llvm::GlobalVariable *EpochCSV = nullptr;
  llvm::GlobalVariable *AddressSpaceCSV = nullptr;
  llvm::GlobalVariable *TypeCSV = nullptr;

  std::set<llvm::GlobalVariable *> CSVsAffectingPC;

public:
  using DispatcherTarget = std::pair<MetaAddress, llvm::BasicBlock *>;
  using DispatcherTargets = std::vector<DispatcherTarget>;

protected:
  ProgramCounterHandler(unsigned Alignment) : Alignment(Alignment) {}

public:
  virtual ~ProgramCounterHandler() {}

public:
  static std::unique_ptr<ProgramCounterHandler>
  create(model::Architecture::Values Architecture,
         llvm::Module *M,
         const CSVFactory &Factory);

  static std::unique_ptr<ProgramCounterHandler>
  fromModule(model::Architecture::Values Architecture, llvm::Module *M);

public:
  std::array<llvm::GlobalVariable *, 4> pcCSVs() const {
    return { EpochCSV, AddressSpaceCSV, TypeCSV, AddressCSV };
  }

  /// Hook for the emission of a store to a CSV
  ///
  /// \param Builder Builder to employ in order to inject new instructions.
  ///        Those are positioned after \p Store
  /// \param Store the StoreInst targeting a CSV
  ///
  /// \return true if new instructions have been emitted.
  bool handleStore(revng::IRBuilder &Builder, llvm::StoreInst *Store) const {
    if (affectsPC(Store))
      return handleStoreInternal(Builder, Store);
    return false;
  }

  void initializePC(revng::IRBuilder &Builder, MetaAddress NewPC) const {
    setPC(Builder, NewPC);
    initializePCInternal(Builder, NewPC);
  }

  void setPC(revng::IRBuilder &Builder, MetaAddress NewPC) const {
    revng_assert(NewPC.isValid() and NewPC.isCode());
    store(Builder, AddressCSV, NewPC.address());
    store(Builder, EpochCSV, NewPC.epoch());
    store(Builder, AddressSpaceCSV, NewPC.addressSpace());
    store(Builder, TypeCSV, NewPC.type());
  }

  void expandNewPC(llvm::CallBase *Call) const {
    revng_assert(isCallTo(Call, "newpc"));
    MetaAddress Address = addressFromNewPC(Call);
    revng::NonDebugInfoCheckingIRBuilder Builder(Call);
    setPC(Builder, Address);
  }

  void setCurrentPCPlainMetaAddress(revng::IRBuilder &Builder) const;
  void setLastPCPlainMetaAddress(revng::IRBuilder &Builder,
                                 const MetaAddress &Address) const;
  void setPlainMetaAddress(revng::IRBuilder &Builder,
                           llvm::StringRef GlobalName,
                           const MetaAddress &Address) const;

protected:
  virtual void initializePCInternal(revng::IRBuilder &Builder,
                                    MetaAddress NewPC) const = 0;

  virtual bool handleStoreInternal(revng::IRBuilder &Builder,
                                   llvm::StoreInst *Store) const = 0;

public:
  bool affectsPC(llvm::GlobalVariable *GV) const {
    return CSVsAffectingPC.contains(GV);
  }

  bool affectsPC(llvm::StoreInst *Store) const {
    using namespace llvm;
    if (auto *CSV = dyn_cast<GlobalVariable>(Store->getPointerOperand()))
      return affectsPC(CSV);
    else
      return false;
  }

  bool isPCAffectingHelper(llvm::Instruction *I) const;

  /// \return an empty optional if the PC has not changed on at least one path,
  ///         an invalid MetaAddress in case there isn't a single next PC, or,
  ///         finally, a valid MetaAddress representing the only possible next
  ///         PC
  std::pair<NextJumpTarget::Values, MetaAddress>
  getUniqueJumpTarget(llvm::BasicBlock *BB);

  void deserializePC(revng::IRBuilder &Builder) const {
    using namespace llvm;

    // Load and re-store each CSV affecting the PC and then feed them to
    // handleStore
    for (GlobalVariable *CSVAffectingPC : CSVsAffectingPC) {
      auto *FakeLoad = createLoad(Builder, CSVAffectingPC);
      auto *FakeStore = Builder.CreateStore(FakeLoad, CSVAffectingPC);
      bool HasInjectedCode = handleStore(Builder, FakeStore);
      eraseFromParent(FakeStore);

      if (not HasInjectedCode) {
        // The store did not produce any effect, the load is useless too
        revng_assert(FakeLoad->use_begin() == FakeLoad->use_end());
        eraseFromParent(FakeLoad);
      }
    }
  }

  virtual llvm::Value *loadJumpablePC(revng::IRBuilder &Builder) const = 0;

  virtual std::array<llvm::Value *, 4>
  dissectJumpablePC(revng::IRBuilder &Builder,
                    llvm::Value *ToDissect,
                    model::Architecture::Values Arch) const = 0;

  virtual void
  deserializePCFromSignalContext(revng::IRBuilder &Builder,
                                 llvm::Value *PCAddress,
                                 llvm::Value *SavedRegisters) const = 0;

  llvm::Instruction *composeIntegerPC(revng::IRBuilder &B) const {
    return MetaAddress::composeIntegerPC(B,
                                         align(B, createLoad(B, AddressCSV)),
                                         createLoad(B, EpochCSV),
                                         createLoad(B, AddressSpaceCSV),
                                         createLoad(B, TypeCSV));
  }

  bool isPCSizedType(llvm::Type *T) const {
    return T == AddressCSV->getValueType();
  }

public:
  struct DispatcherInfo {
    llvm::SmallVector<llvm::BasicBlock *, 4> NewBlocks;
    llvm::SwitchInst *Switch;
  };

  /// \param Targets the targets to materialize for the dispatcher. Will be
  ///        sorted.
  DispatcherInfo
  buildDispatcher(DispatcherTargets &Targets,
                  revng::IRBuilder &Builder,
                  llvm::BasicBlock *Default,
                  std::optional<BlockType::Values> SetBlockType) const;

  DispatcherInfo
  buildDispatcher(DispatcherTargets &Targets,
                  llvm::BasicBlock *CreateIn,
                  llvm::BasicBlock *Default,
                  std::optional<BlockType::Values> SetBlockType) const {
    revng::NonDebugInfoCheckingIRBuilder Builder(CreateIn);
    return buildDispatcher(Targets, Builder, Default, SetBlockType);
  }

  /// \note \p Root must not already contain a case for \p NewTarget
  void addCaseToDispatcher(llvm::SwitchInst *Root,
                           const DispatcherTarget &NewTarget,
                           std::optional<BlockType::Values> SetBlockType) const;

  void destroyDispatcher(llvm::SwitchInst *Root) const;

  void buildHotPath(revng::IRBuilder &Builder,
                    const DispatcherTarget &CandidateTarget,
                    llvm::BasicBlock *Default) const;

protected:
  void createMissingVariables(llvm::Module *M) {
    if (EpochCSV == nullptr)
      EpochCSV = createEpoch(M);
    if (AddressSpaceCSV == nullptr)
      AddressSpaceCSV = createAddressSpace(M);
    if (TypeCSV == nullptr)
      TypeCSV = createType(M);
  }

  llvm::Value *align(revng::IRBuilder &Builder, llvm::Value *V) const {
    revng_assert(Alignment != 0);

    if (Alignment == 1)
      return V;

    using namespace llvm;
    revng_assert(isPowerOf2_64(Alignment));
    auto *Type = cast<IntegerType>(V->getType());
    Value *Mask = ConstantInt::get(Type, ~(Alignment - 1));
    return Builder.CreateAnd(V, Mask);
  }

protected:
  void setMissingVariables(llvm::Module *M, llvm::StringRef AddressName) {
    AddressCSV = M->getGlobalVariable(AddressName, true);
    EpochCSV = M->getGlobalVariable(EpochName, true);
    AddressSpaceCSV = M->getGlobalVariable(AddressSpaceName, true);
    TypeCSV = M->getGlobalVariable(TypeName, true);

    revng_assert(AddressCSV != nullptr and EpochCSV != nullptr
                 and AddressSpaceCSV != nullptr and TypeCSV != nullptr);
  }

private:
  static llvm::GlobalVariable *createEpoch(llvm::Module *M) {
    return createVariable(M, EpochName, sizeof(MetaAddress::Epoch));
  }

  static llvm::GlobalVariable *createAddressSpace(llvm::Module *M) {
    return createVariable(M,
                          AddressSpaceName,
                          sizeof(MetaAddress::AddressSpace));
  }

  static llvm::GlobalVariable *createType(llvm::Module *M) {
    return createVariable(M, TypeName, sizeof(MetaAddress::Type));
  }

  static llvm::GlobalVariable *
  createVariable(llvm::Module *M, llvm::StringRef Name, size_t Size) {
    using namespace llvm;
    auto *T = Type::getIntNTy(M->getContext(), Size * 8);
    return new GlobalVariable(*M,
                              T,
                              false,
                              GlobalValue::ExternalLinkage,
                              ConstantInt::get(T, 0),
                              Name);
  }

protected:
  static llvm::StoreInst *
  store(revng::IRBuilder &Builder, llvm::GlobalVariable *GV, uint64_t Value) {
    using namespace llvm;
    auto *Type = cast<IntegerType>(GV->getValueType());
    return Builder.CreateStore(ConstantInt::get(Type, Value), GV);
  }
};
