#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/IRHelpers.h"

#include "PTCInterface.h"

inline llvm::IntegerType *getCSVType(llvm::GlobalVariable *CSV) {
  using namespace llvm;
  return cast<IntegerType>(CSV->getType()->getPointerElementType());
}

namespace NextJumpTarget {

enum Values { Unique, Multiple, Helper };

};

using CSVFactory = std::function<llvm::GlobalVariable *(intptr_t Offset,
                                                        llvm::StringRef Name)>;

class ProgramCounterHandler {
protected:
  static constexpr const char *AddressName = "pc";
  static constexpr const char *AddressSpaceName = "pc_address_space";
  static constexpr const char *EpochName = "pc_epoch";
  static constexpr const char *TypeName = "pc_type";

protected:
  llvm::GlobalVariable *AddressCSV;
  llvm::GlobalVariable *EpochCSV;
  llvm::GlobalVariable *AddressSpaceCSV;
  llvm::GlobalVariable *TypeCSV;

  std::set<llvm::Value *> CSVsAffectingPC;

public:
  using DispatcherTarget = std::pair<MetaAddress, llvm::BasicBlock *>;
  using DispatcherTargets = std::vector<DispatcherTarget>;

protected:
  ProgramCounterHandler() :
    AddressCSV(nullptr),
    EpochCSV(nullptr),
    AddressSpaceCSV(nullptr),
    TypeCSV(nullptr) {}

public:
  virtual ~ProgramCounterHandler() {}

public:
  static std::unique_ptr<ProgramCounterHandler>
  create(llvm::Triple::ArchType Architecture,
         llvm::Module *M,
         PTCInterface *PTC,
         const CSVFactory &Factory);

public:
  std::array<llvm::GlobalVariable *, 4> pcCSVs() const {
    return { AddressCSV, EpochCSV, AddressSpaceCSV, TypeCSV };
  }

  /// \brief Hook for the emission of a store to a CSV
  ///
  /// \param Builder IRBuilder to employ in order to inject new instructions. It
  ///        is positioned after \p Store
  /// \param Store the StoreInst targeting a CSV
  ///
  /// \return true if new instructions have been emitted.
  bool handleStore(llvm::IRBuilder<> &Builder, llvm::StoreInst *Store) const {
    if (affectsPC(Store))
      return handleStoreInternal(Builder, Store);
    return false;
  }

  void initializePC(llvm::IRBuilder<> &Builder, MetaAddress NewPC) const {
    setPC(Builder, NewPC);
    initializePCInternal(Builder, NewPC);
  }

  void setPC(llvm::IRBuilder<> &Builder, MetaAddress NewPC) const {
    revng_assert(NewPC.isValid() and NewPC.isCode());
    store(Builder, AddressCSV, NewPC.address());
    store(Builder, EpochCSV, NewPC.epoch());
    store(Builder, AddressSpaceCSV, NewPC.addressSpace());
    store(Builder, TypeCSV, NewPC.type());
  }

  void expandNewPC(llvm::CallInst *Call) const {
    revng_assert(isCallTo(Call, "newpc"));
    auto MA = MetaAddress::fromConstant(Call->getArgOperand(0));
    llvm::IRBuilder<> Builder(Call);
    setPC(Builder, MA);
  }

protected:
  virtual void
  initializePCInternal(llvm::IRBuilder<> &Builder, MetaAddress NewPC) const = 0;

  virtual bool handleStoreInternal(llvm::IRBuilder<> &Builder,
                                   llvm::StoreInst *Store) const = 0;

public:
  bool affectsPC(llvm::GlobalVariable *GV) const {
    return CSVsAffectingPC.count(GV) != 0;
  }

  bool affectsPC(llvm::StoreInst *Store) const {
    using namespace llvm;
    if (auto *CSV = dyn_cast<GlobalVariable>(Store->getPointerOperand()))
      return affectsPC(CSV);
    else
      return false;
  }

  /// \return an empty Optional if the PC has not changed on at least one path,
  ///         an invalid MetaAddress in case there isn't a single next PC, or,
  ///         finally, a valid MetaAddress representing the only possible next
  ///         PC
  std::pair<NextJumpTarget::Values, MetaAddress>
  getUniqueJumpTarget(llvm::BasicBlock *BB);

  void deserializePC(llvm::IRBuilder<> &Builder) const {
    using namespace llvm;

    // Load and re-store each CSV affecting the PC and then feed them to
    // handleStore
    for (Value *CSVAffectingPC : CSVsAffectingPC) {
      auto *FakeLoad = Builder.CreateLoad(CSVAffectingPC);
      auto *FakeStore = Builder.CreateStore(FakeLoad, CSVAffectingPC);
      bool HasInjectedCode = handleStore(Builder, FakeStore);
      FakeStore->eraseFromParent();

      if (not HasInjectedCode) {
        // The store did not produce any effect, the load is useless too
        revng_assert(FakeLoad->use_begin() == FakeLoad->use_end());
        FakeLoad->eraseFromParent();
      }
    }
  }

  virtual llvm::Value *loadJumpablePC(llvm::IRBuilder<> &Builder) const = 0;

  virtual void
  deserializePCFromSignalContext(llvm::IRBuilder<> &Builder,
                                 llvm::Value *PCAddress,
                                 llvm::Value *SavedRegisters) const = 0;

  llvm::Instruction *composeIntegerPC(llvm::IRBuilder<> &B) const {
    return MetaAddress::composeIntegerPC(B,
                                         B.CreateLoad(AddressCSV),
                                         B.CreateLoad(EpochCSV),
                                         B.CreateLoad(AddressSpaceCSV),
                                         B.CreateLoad(TypeCSV));
  }

  bool isPCSizedType(llvm::Type *T) const {
    return T == AddressCSV->getType()->getPointerElementType();
  }

public:
  /// \param Targets the targets to materialize for the dispatcher. Will be
  ///        sorted.
  llvm::SwitchInst *
  buildDispatcher(DispatcherTargets &Targets,
                  llvm::IRBuilder<> &Builder,
                  llvm::BasicBlock *Default,
                  llvm::Optional<BlockType::Values> SetBlockType) const;

  llvm::SwitchInst *
  buildDispatcher(DispatcherTargets &Targets,
                  llvm::BasicBlock *CreateIn,
                  llvm::BasicBlock *Default,
                  llvm::Optional<BlockType::Values> SetBlockType) const {
    llvm::IRBuilder<> Builder(CreateIn);
    return buildDispatcher(Targets, Builder, Default, SetBlockType);
  }

  void
  addCaseToDispatcher(llvm::SwitchInst *Root,
                      const DispatcherTarget &NewTarget,
                      llvm::Optional<BlockType::Values> SetBlockType) const;

  void destroyDispatcher(llvm::SwitchInst *Root) const;

  void buildHotPath(llvm::IRBuilder<> &Builder,
                    const DispatcherTarget &CandidateTarget,
                    llvm::BasicBlock *Default) const;

protected:
  void createMissingVariables(llvm::Module *M) {
    if (AddressCSV == nullptr)
      AddressCSV = createAddress(M);
    if (EpochCSV == nullptr)
      EpochCSV = createEpoch(M);
    if (AddressSpaceCSV == nullptr)
      AddressSpaceCSV = createAddressSpace(M);
    if (TypeCSV == nullptr)
      TypeCSV = createType(M);
  }

private:
  bool isPCAffectingHelper(llvm::Instruction *I) const;

  static llvm::GlobalVariable *createAddress(llvm::Module *M) {
    return createVariable(M, AddressName, sizeof(MetaAddress::Address));
  }

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
  store(llvm::IRBuilder<> Builder, llvm::GlobalVariable *GV, uint64_t Value) {
    using namespace llvm;
    auto *Type = cast<IntegerType>(GV->getType()->getPointerElementType());
    return Builder.CreateStore(ConstantInt::get(Type, Value), GV);
  }
};

class PCOnlyProgramCounterHandler : public ProgramCounterHandler {
public:
  PCOnlyProgramCounterHandler(llvm::Module *M,
                              PTCInterface *PTC,
                              const CSVFactory &Factory) {
    AddressCSV = Factory(PTC->pc, "pc");
    CSVsAffectingPC.insert(AddressCSV);

    createMissingVariables(M);
  }

public:
  bool handleStoreInternal(llvm::IRBuilder<> &Builder,
                           llvm::StoreInst *Store) const final {
    revng_assert(Store->getPointerOperand() == AddressCSV);
    return false;
  }

  llvm::Value *loadJumpablePC(llvm::IRBuilder<> &Builder) const final {
    return Builder.CreateLoad(AddressCSV);
  }

  void deserializePCFromSignalContext(llvm::IRBuilder<> &Builder,
                                      llvm::Value *PCAddress,
                                      llvm::Value *SavedRegisters) const final {
    Builder.CreateStore(PCAddress, AddressCSV);
  }

protected:
  void initializePCInternal(llvm::IRBuilder<> &Builder,
                            MetaAddress NewPC) const final {}
};

class ARMProgramCounterHandler : public ProgramCounterHandler {
private:
  llvm::GlobalVariable *IsThumb;

public:
  ARMProgramCounterHandler(llvm::Module *M,
                           PTCInterface *PTC,
                           const CSVFactory &Factory) {
    AddressCSV = Factory(PTC->pc, AddressName);
    IsThumb = Factory(PTC->is_thumb, "is_thumb");
    CSVsAffectingPC.insert(AddressCSV);
    CSVsAffectingPC.insert(IsThumb);

    createMissingVariables(M);
  }

private:
  bool handleStoreInternal(llvm::IRBuilder<> &B,
                           llvm::StoreInst *Store) const final {
    using namespace llvm;
    revng_assert(affectsPC(Store));

    Value *Pointer = Store->getPointerOperand();
    Value *StoredValue = Store->getValueOperand();

    if (Pointer == IsThumb) {
      // Update Type
      using CI = ConstantInt;
      using namespace MetaAddressType;

      auto *TypeType = getCSVType(TypeCSV);
      auto *ArmCode = CI::get(TypeType, Code_arm);
      auto *ThumbCode = CI::get(TypeType, Code_arm_thumb);
      // We don't use select here, SCEV can't handle it
      // NewType = ARM + IsThumb * (Thumb - ARM)
      auto *NewType = B.CreateAdd(ArmCode,
                                  B.CreateMul(B.CreateTrunc(StoredValue,
                                                            TypeType),
                                              B.CreateSub(ThumbCode, ArmCode)));

      B.CreateStore(NewType, TypeCSV);

      return true;
    }

    return false;
  }

  llvm::Value *loadJumpablePC(llvm::IRBuilder<> &Builder) const final {
    auto *Address = Builder.CreateLoad(AddressCSV);
    auto *AddressType = Address->getType();
    return Builder.CreateOr(Address,
                            Builder.CreateZExt(Builder.CreateLoad(IsThumb),
                                               AddressType));
  }

  void deserializePCFromSignalContext(llvm::IRBuilder<> &B,
                                      llvm::Value *PCAddress,
                                      llvm::Value *SavedRegisters) const final {
    using namespace llvm;

    constexpr uint32_t CPSRIndex = 19;
    constexpr unsigned IsThumbBitIndex = 5;

    Type *IsThumbType = IsThumb->getType()->getPointerElementType();

    // Load the CPSR field
    Value *CPSRAddress = B.CreateGEP(SavedRegisters, B.getInt32(CPSRIndex));
    Value *CPSR = B.CreateLoad(CPSRAddress);

    // Select the T bit
    Value *TBit = B.CreateAnd(B.CreateLShr(CPSR, IsThumbBitIndex), 1);

    // Zero-extend and store in IsThumb CSV
    auto *IsThumbStore = B.CreateStore(B.CreateZExt(TBit, IsThumbType),
                                       IsThumb);

    // Let handleStore do his thing
    handleStore(B, IsThumbStore);

    // Update the PC address too
    B.CreateStore(PCAddress, AddressCSV);
  }

protected:
  void initializePCInternal(llvm::IRBuilder<> &Builder,
                            MetaAddress NewPC) const final {
    using namespace MetaAddressType;
    store(Builder, IsThumb, NewPC.type() == Code_arm_thumb ? 1 : 0);
  }
};
