/// \file ProgramCounterHandler.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ModRef.h"

#include "revng/Support/ProgramCounterHandler.h"

using namespace llvm;
using PCH = ProgramCounterHandler;

class PCOnlyProgramCounterHandler : public ProgramCounterHandler {
public:
  PCOnlyProgramCounterHandler(unsigned Alignment) :
    ProgramCounterHandler(Alignment) {}

public:
  static std::unique_ptr<ProgramCounterHandler>
  create(Module *M, const CSVFactory &Factory, unsigned Alignment) {
    auto Result = std::make_unique<PCOnlyProgramCounterHandler>(Alignment);

    // Create and register the pc CSV
    Result->AddressCSV = Factory(PCAffectingCSV::PC, AddressName);
    Result->CSVsAffectingPC.insert(Result->AddressCSV);

    // Create the other variables (non-CSV)
    Result->createMissingVariables(M);

    return Result;
  }

  static std::unique_ptr<ProgramCounterHandler> fromModule(Module *M,
                                                           unsigned Alignment) {
    auto Result = std::make_unique<PCOnlyProgramCounterHandler>(Alignment);

    // Initialize the standard variables
    Result->setMissingVariables(M);

    // Register pc as a CSV affecting the program counter
    Result->CSVsAffectingPC.insert(Result->AddressCSV);

    return Result;
  }

public:
  bool handleStoreInternal(IRBuilder<> &Builder, StoreInst *Store) const final {
    revng_assert(Store->getPointerOperand() == AddressCSV);
    return false;
  }

  Value *loadJumpablePC(IRBuilder<> &Builder) const final {
    return createLoad(Builder, AddressCSV);
  }

  std::array<Value *, 4> dissectJumpablePC(IRBuilder<> &Builder,
                                           Value *ToDissect,
                                           Triple::ArchType Arch) const final {
    IntegerType *Ty = getCSVType(TypeCSV);

    Value *Address = align(Builder, ToDissect);
    Value *Epoch = ConstantInt::get(Ty, 0);
    Value *AddressSpace = ConstantInt::get(Ty, 0);
    Value *Type = ConstantInt::get(Ty,
                                   MetaAddressType::defaultCodeFromArch(Arch));
    return { Address, Epoch, AddressSpace, Type };
  }

  void deserializePCFromSignalContext(IRBuilder<> &Builder,
                                      Value *PCAddress,
                                      Value *SavedRegisters) const final {
    Builder.CreateStore(PCAddress, AddressCSV);
  }

protected:
  void initializePCInternal(IRBuilder<> &Builder,
                            MetaAddress NewPC) const final {}
};

class ARMProgramCounterHandler : public ProgramCounterHandler {
private:
  static constexpr const char *IsThumbName = "is_thumb";

private:
  GlobalVariable *IsThumb;

public:
  ARMProgramCounterHandler() : ProgramCounterHandler(2), IsThumb(nullptr) {}

public:
  static std::unique_ptr<ProgramCounterHandler>
  create(Module *M, const CSVFactory &Factory) {
    auto Result = std::make_unique<ARMProgramCounterHandler>();

    // Create and register the pc and is_thumb CSV
    Result->AddressCSV = Factory(PCAffectingCSV::PC, AddressName);
    Result->CSVsAffectingPC.insert(Result->AddressCSV);

    Result->IsThumb = Factory(PCAffectingCSV::IsThumb, IsThumbName);
    Result->CSVsAffectingPC.insert(Result->IsThumb);

    Result->createMissingVariables(M);

    return Result;
  }

  static std::unique_ptr<ProgramCounterHandler> fromModule(Module *M) {
    auto Result = std::make_unique<ARMProgramCounterHandler>();

    // Initialize the standard variablesx
    Result->setMissingVariables(M);

    // Get is_thumb
    Result->IsThumb = M->getGlobalVariable(IsThumbName, true);
    revng_assert(Result->IsThumb != nullptr);

    // Register pc and is_thumb as a CSV affecting the program counter
    Result->CSVsAffectingPC.insert(Result->IsThumb);
    Result->CSVsAffectingPC.insert(Result->AddressCSV);

    return Result;
  }

private:
  bool handleStoreInternal(IRBuilder<> &B, StoreInst *Store) const final {
    using namespace llvm;
    revng_assert(affectsPC(Store));

    Value *Pointer = Store->getPointerOperand();

    if (Pointer == IsThumb) {
      // Compute Type and update it.
      Value *ThumbValue = Store->getValueOperand();
      B.CreateStore(computeMetaAddressType(B, ThumbValue), TypeCSV);

      return true;
    }

    return false;
  }

  Value *loadJumpablePC(IRBuilder<> &Builder) const final {
    auto *Address = createLoad(Builder, AddressCSV);
    auto *AddressType = Address->getType();
    return Builder.CreateOr(Address,
                            Builder.CreateZExt(createLoad(Builder, IsThumb),
                                               AddressType));
  }

  std::array<Value *, 4> dissectJumpablePC(IRBuilder<> &Builder,
                                           Value *ToDissect,
                                           Triple::ArchType Arch) const final {
    constexpr uint32_t ThumbMask = 0x1;
    constexpr uint32_t AddressMask = 0xFFFFFFFE;
    IntegerType *Ty = getCSVType(TypeCSV);

    Value *IsThumb = Builder.CreateAnd(ToDissect, ThumbMask);
    // We should align(Address) but we're doing exactly the same here so no need
    Value *Address = Builder.CreateAnd(ToDissect, AddressMask);
    Value *Epoch = ConstantInt::get(Ty, 0);
    Value *AddressSpace = ConstantInt::get(Ty, 0);
    Value *Type = computeMetaAddressType(Builder, IsThumb);
    return { Address, Epoch, AddressSpace, Type };
  }

  void deserializePCFromSignalContext(IRBuilder<> &B,
                                      Value *PCAddress,
                                      Value *SavedRegisters) const final {
    using namespace llvm;

    constexpr uint32_t CPSRIndex = 19;
    constexpr unsigned IsThumbBitIndex = 5;

    Type *IsThumbType = IsThumb->getValueType();

    // Load the CPSR field
    IntegerType *RegisterType = IntegerType::get(AddressCSV->getContext(), 32);
    Value *CPSRAddress = B.CreateGEP(RegisterType,
                                     SavedRegisters,
                                     B.getInt32(CPSRIndex));
    Value *CPSR = B.CreateLoad(RegisterType, CPSRAddress);

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

  Value *computeMetaAddressType(IRBuilder<> &B, Value *IsThumb) const {
    using CI = ConstantInt;
    using namespace MetaAddressType;

    auto *TypeType = getCSVType(TypeCSV);
    auto *ArmCode = CI::get(TypeType, Code_arm);
    auto *ThumbCode = CI::get(TypeType, Code_arm_thumb);
    // We don't use select here, SCEV can't handle it
    // NewType = ARM + IsThumb * (Thumb - ARM)
    auto *NewType = B.CreateAdd(ArmCode,
                                B.CreateMul(B.CreateTrunc(IsThumb, TypeType),
                                            B.CreateSub(ThumbCode, ArmCode)));
    return NewType;
  }

protected:
  void initializePCInternal(IRBuilder<> &Builder,
                            MetaAddress NewPC) const final {
    using namespace MetaAddressType;
    store(Builder, IsThumb, NewPC.type() == Code_arm_thumb ? 1 : 0);
  }
};

static void eraseIfNoUse(const WeakVH &V) {
  if (V != nullptr)
    if (Instruction *I = dyn_cast_or_null<Instruction>(&*V))
      if (I->use_begin() == I->use_end())
        eraseFromParent(I);
}

static SwitchInst *getNextSwitch(SwitchInst::CaseHandle Case) {
  return cast<SwitchInst>(Case.getCaseSuccessor()->getTerminator());
}

static SwitchInst *getNextSwitch(SwitchInst::CaseIt It) {
  return getNextSwitch(*It);
}

static ConstantInt *caseConstant(SwitchInst *Switch, uint64_t Value) {
  auto *ConditionType = cast<IntegerType>(Switch->getCondition()->getType());
  return ConstantInt::get(ConditionType, Value);
}

static void addCase(SwitchInst *Switch, uint64_t Value, BasicBlock *BB) {
#if defined(NDEBUG) && defined(EXPENSIVE_ASSERTIONS)
  auto *C = caseConstant(AddressSwitch, Value);
  auto CaseIt = AddressSwitch->findCaseValue(C);
  revng_assert(CaseIt == AddressSwitch->case_default());
#endif
  Switch->addCase(caseConstant(Switch, Value), BB);
}

class PartialMetaAddress {
private:
  std::optional<uint64_t> Address;
  std::optional<uint64_t> Epoch;
  std::optional<uint64_t> AddressSpace;
  std::optional<uint64_t> Type;

public:
  bool isEmpty() const { return not(Address or Epoch or AddressSpace or Type); }

  void set(const MetaAddress &MA) {
    setAddress(MA.address());
    setEpoch(MA.epoch());
    setAddressSpace(MA.addressSpace());
    setType(MA.type());
  }

  void setAddress(uint64_t V) {
    if (not Address)
      Address = V;
  }

  void setEpoch(uint64_t V) {
    if (not Epoch)
      Epoch = V;
  }

  void setAddressSpace(uint64_t V) {
    if (not AddressSpace)
      AddressSpace = V;
  }

  void setType(uint64_t V) {
    if (not Type)
      Type = V;
  }

  bool hasAddress() { return Address.has_value(); }
  bool hasEpoch() { return Epoch.has_value(); }
  bool hasAddressSpace() { return AddressSpace.has_value(); }
  bool hasType() { return Type.has_value(); }

  MetaAddress toMetaAddress() const {
    if (Type and Address and Epoch and AddressSpace) {
      auto TheType = static_cast<MetaAddressType::Values>(*Type);
      if (MetaAddressType::isValid(TheType)) {
        return MetaAddress(*Address, TheType, *Epoch, *AddressSpace);
      }
    }

    return MetaAddress::invalid();
  }
};

class State {
private:
  PartialMetaAddress PMA;
  SmallSet<BasicBlock *, 4> Visited;

public:
  bool visit(BasicBlock *BB) {
    // Check if we already visited this block
    if (Visited.contains(BB)) {
      return true;
    } else {
      // Register as visited
      Visited.insert(BB);
      return false;
    }
  }

  PartialMetaAddress &agreement() { return PMA; }
};

class StackEntry {
private:
  State S;
  pred_iterator Next;
  pred_iterator End;

public:
  StackEntry(pred_iterator Begin, pred_iterator End, const State &S) :
    S(S), Next(Begin), End(End) {}

  bool isDone() const { return Next == End; }

  std::pair<State *, BasicBlock *> next() {
    revng_assert(not isDone());
    BasicBlock *NextBB = *Next;
    ++Next;
    return { &S, NextBB };
  }
};

bool PCH::isPCAffectingHelper(Instruction *I) const {
  CallInst *HelperCall = getCallToHelper(I);
  if (HelperCall == nullptr)
    return false;

  auto MaybeUsedCSVs = getCSVUsedByHelperCallIfAvailable(HelperCall);

  // If CSAA didn't consider this helper, be conservative
  if (not MaybeUsedCSVs)
    return true;

  for (GlobalVariable *CSV : MaybeUsedCSVs->Written)
    if (affectsPC(CSV))
      return true;

  return false;
}

static llvm::Value *buildPlainMetaAddressImpl(llvm::IRBuilder<> &Builder,
                                              Value *Epoch,
                                              Value *AddressSpace,
                                              Value *Type,
                                              Value *Address) {
  using namespace llvm;

  BasicBlock *BB = Builder.GetInsertBlock();
  Module *M = BB->getParent()->getParent();

  Function *MetaAddressConstuctor = M->getFunction("build_PlainMetaAddress");
  MetaAddressConstuctor->setMemoryEffects(MemoryEffects::none());
  MetaAddressConstuctor->addFnAttr(Attribute::WillReturn);
  MetaAddressConstuctor->addFnAttr(Attribute::NoUnwind);
  auto *FT = MetaAddressConstuctor->getFunctionType();
  return Builder
    .CreateCall(MetaAddressConstuctor,
                { Builder.CreateZExt(Epoch, FT->getParamType(0)),
                  Builder.CreateZExt(AddressSpace, FT->getParamType(1)),
                  Builder.CreateZExt(Type, FT->getParamType(2)),
                  Builder.CreateZExt(Address, FT->getParamType(3)) });
}

Value *PCH::buildCurrentPCPlainMetaAddress(IRBuilder<> &Builder) const {
  return buildPlainMetaAddressImpl(Builder,
                                   createLoad(Builder, EpochCSV),
                                   createLoad(Builder, AddressSpaceCSV),
                                   createLoad(Builder, TypeCSV),
                                   createLoad(Builder, AddressCSV));
}

llvm::Value *
ProgramCounterHandler::buildPlainMetaAddress(llvm::IRBuilder<> &Builder,
                                             const MetaAddress &Address) const {
  auto CI = [](llvm::GlobalVariable *Example, uint64_t Value) -> ConstantInt * {
    auto *Type = Example->getValueType();
    return ConstantInt::get(cast<IntegerType>(Type), Value);
  };
  return buildPlainMetaAddressImpl(Builder,
                                   CI(EpochCSV, Address.Epoch),
                                   CI(AddressSpaceCSV, Address.AddressSpace),
                                   CI(TypeCSV, Address.Type),
                                   CI(AddressCSV, Address.Address));
}

std::pair<NextJumpTarget::Values, MetaAddress>
PCH::getUniqueJumpTarget(BasicBlock *BB) {
  std::vector<StackEntry> Stack;

  enum ProcessResult {
    Proceed,
    DontProceed,
    BailOut
  };

  std::optional<MetaAddress> AgreedMA;

  bool ChangedByHelper = false;

  auto Process = [&AgreedMA,
                  this,
                  &ChangedByHelper](State &S, BasicBlock *BB) -> ProcessResult {
    // Do not follow backedges
    if (S.visit(BB))
      return DontProceed;

    PartialMetaAddress &PMA = S.agreement();

    // Iterate backward on all instructions
    for (Instruction &I : make_range(BB->rbegin(), BB->rend())) {
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        // We found a store
        Value *Pointer = Store->getPointerOperand();
        Value *V = Store->getValueOperand();
        bool AffectsPC = (Pointer == AddressCSV || Pointer == EpochCSV
                          || Pointer == AddressSpaceCSV || Pointer == TypeCSV);

        if (not AffectsPC)
          continue;

        if (auto *StoredValue = dyn_cast<ConstantInt>(skipCasts(V))) {
          // The store affects the PC and it's constant
          uint64_t Value = getLimitedValue(StoredValue);
          if (Pointer == AddressCSV) {
            PMA.setAddress(Value);
          } else if (Pointer == EpochCSV) {
            PMA.setEpoch(Value);
          } else if (Pointer == AddressSpaceCSV) {
            PMA.setAddressSpace(Value);
          } else if (Pointer == TypeCSV) {
            PMA.setType(Value);
          }

        } else if ((Pointer == AddressCSV and not PMA.hasAddress())
                   or (Pointer == EpochCSV and not PMA.hasEpoch())
                   or (Pointer == AddressSpaceCSV and not PMA.hasAddressSpace())
                   or (Pointer == TypeCSV and not PMA.hasType())) {
          AgreedMA = MetaAddress::invalid();
          return BailOut;
        }

      } else if (CallInst *NewPCCall = getCallTo(&I, "newpc")) {
        //
        // We reached a call to newpc
        //

        if (PMA.isEmpty()) {
          // We have found a path on which the PC doesn't change
          AgreedMA = MetaAddress::invalid();
          return BailOut;
        }

        // Obtain the current PC and fill in all the missing fields
        PMA.set(addressFromNewPC(NewPCCall));

        // Compute the final MetaAddress on this path and ensure it's the same
        // as previous ones
        auto MA = PMA.toMetaAddress();
        if (AgreedMA and MA != *AgreedMA) {
          AgreedMA = MetaAddress::invalid();
          return BailOut;
        } else {
          AgreedMA = MA;
          return DontProceed;
        }

      } else if (PMA.isEmpty() and isPCAffectingHelper(&I)) {
        // Non-constant store to PC CSV when no other value of the PC has been
        // written yet, bail out
        AgreedMA = MetaAddress::invalid();
        ChangedByHelper = true;
        return BailOut;
      }
    }

    return Proceed;
  };

  State Initial;

  BasicBlock *CurrentBB = BB;
  State *CurrentState = &Initial;

  while (true) {
    ProcessResult Result = Process(*CurrentState, CurrentBB);
    switch (Result) {
    case Proceed:
      Stack.emplace_back(pred_begin(CurrentBB),
                         pred_end(CurrentBB),
                         *CurrentState);
      break;

    case BailOut:
      Stack.clear();
      break;

    case DontProceed:
      break;
    }

    while (Stack.size() > 0 and Stack.back().isDone())
      Stack.pop_back();

    if (Stack.size() == 0)
      break;

    std::tie(CurrentState, CurrentBB) = Stack.back().next();
  }

  if (ChangedByHelper) {
    return { NextJumpTarget::Helper, MetaAddress::invalid() };
  } else if (AgreedMA and AgreedMA->isValid()) {
    return { NextJumpTarget::Unique, *AgreedMA };
  } else {
    return { NextJumpTarget::Multiple, MetaAddress::invalid() };
  }
}

class SwitchManager {
private:
  LLVMContext &Context;
  Function *F;
  BasicBlock *Default;

  Value *CurrentEpoch;
  Value *CurrentAddressSpace;
  Value *CurrentType;
  Value *CurrentAddress;

  std::optional<BlockType::Values> SetBlockType;

  SmallVectorImpl<BasicBlock *> *NewBlocksRegistry = nullptr;

public:
  SwitchManager(BasicBlock *Default,
                Value *CurrentEpoch,
                Value *CurrentAddressSpace,
                Value *CurrentType,
                Value *CurrentAddress,
                std::optional<BlockType::Values> SetBlockType,
                SmallVectorImpl<BasicBlock *> *NewBlocksRegistry = nullptr) :
    Context(getContext(Default)),
    F(Default->getParent()),
    Default(Default),
    CurrentEpoch(CurrentEpoch),
    CurrentAddressSpace(CurrentAddressSpace),
    CurrentType(CurrentType),
    CurrentAddress(CurrentAddress),
    SetBlockType(SetBlockType),
    NewBlocksRegistry(NewBlocksRegistry) {}

  SwitchManager(SwitchInst *Root,
                GlobalVariable *EpochCSV,
                GlobalVariable *AddressSpaceCSV,
                GlobalVariable *TypeCSV,
                GlobalVariable *AddressCSV,
                std::optional<BlockType::Values> SetBlockType) :
    Context(getContext(Root)),
    F(Root->getParent()->getParent()),
    Default(Root->getDefaultDest()),
    SetBlockType(SetBlockType) {

    bool Empty = Root->case_begin() == Root->case_end();
    if (Empty) {
      IRBuilder<> Builder(Root);
      CurrentEpoch = createLoad(Builder, EpochCSV);
      CurrentAddressSpace = createLoad(Builder, AddressSpaceCSV);
      CurrentType = createLoad(Builder, TypeCSV);
      CurrentAddress = createLoad(Builder, AddressCSV);
    } else {
      // Get the switches of the the first MA. This is just in order to get a
      // reference to their conditions
      SwitchInst *EpochSwitch = Root;
      SwitchInst *AddressSpaceSwitch = getNextSwitch(EpochSwitch->case_begin());
      SwitchInst *TypeSwitch = getNextSwitch(AddressSpaceSwitch->case_begin());
      SwitchInst *AddressSwitch = getNextSwitch(TypeSwitch->case_begin());

      // Get the conditions
      CurrentEpoch = EpochSwitch->getCondition();
      CurrentAddressSpace = AddressSpaceSwitch->getCondition();
      CurrentType = TypeSwitch->getCondition();
      CurrentAddress = AddressSwitch->getCondition();
    }
  }

public:
  void destroy(SwitchInst *Root) {
    std::vector<BasicBlock *> AddressSpaceSwitchesBBs;
    std::vector<BasicBlock *> TypeSwitchesBBs;
    std::vector<BasicBlock *> AddressSwitchesBBs;

    // Collect all the switches basic blocks in post-order
    for (const auto &EpochCase : Root->cases()) {
      AddressSpaceSwitchesBBs.push_back(EpochCase.getCaseSuccessor());
      for (const auto &AddressSpaceCase : getNextSwitch(EpochCase)->cases()) {
        TypeSwitchesBBs.push_back(AddressSpaceCase.getCaseSuccessor());
        for (const auto &TypeCase : getNextSwitch(AddressSpaceCase)->cases()) {
          AddressSwitchesBBs.push_back(TypeCase.getCaseSuccessor());
        }
      }
    }

    WeakVH EpochVH(CurrentEpoch);
    WeakVH AddressSpaceVH(CurrentAddressSpace);
    WeakVH TypeVH(CurrentType);
    WeakVH AddressVH(CurrentAddress);

    // Drop the epoch switch
    eraseFromParent(Root);

    // Drop all the switches on address space
    for (BasicBlock *BB : AddressSpaceSwitchesBBs)
      eraseFromParent(BB);

    // Drop all the switches on type
    for (BasicBlock *BB : TypeSwitchesBBs)
      eraseFromParent(BB);

    // Drop all the switches on address
    for (BasicBlock *BB : AddressSwitchesBBs)
      eraseFromParent(BB);

    eraseIfNoUse(EpochVH);
    eraseIfNoUse(AddressSpaceVH);
    eraseIfNoUse(TypeVH);
    eraseIfNoUse(AddressVH);
  }

  SwitchInst *createSwitch(Value *V, IRBuilder<> &Builder) {
    return Builder.CreateSwitch(V, Default, 0);
  }

  SwitchInst *getOrCreateAddressSpaceSwitch(SwitchInst *EpochSwitch,
                                            const MetaAddress &MA) {
    if (auto *Existing = getSwitchForLabel(EpochSwitch, MA.epoch())) {
      return Existing;
    } else {
      return registerEpochCase(EpochSwitch, MA);
    }
  }

  SwitchInst *getOrCreateTypeSwitch(SwitchInst *AddressSpaceSwitch,
                                    const MetaAddress &MA) {
    if (auto *Existing = getSwitchForLabel(AddressSpaceSwitch,
                                           MA.addressSpace())) {
      return Existing;
    } else {
      return registerAddressSpaceCase(AddressSpaceSwitch, MA);
    }
  }

  SwitchInst *getOrCreateAddressSwitch(SwitchInst *TypeSwitch,
                                       const MetaAddress &MA) {
    if (auto *Existing = getSwitchForLabel(TypeSwitch, MA.type())) {
      return Existing;
    } else {
      return registerTypeCase(TypeSwitch, MA);
    }
  }

  SwitchInst *registerEpochCase(SwitchInst *Switch, const MetaAddress &MA) {
    return registerNewCase(Switch,
                           MA.epoch(),
                           Twine("epoch_") + Twine(MA.epoch()),
                           CurrentAddressSpace);
  }

  SwitchInst *registerAddressSpaceCase(SwitchInst *Switch,
                                       const MetaAddress &MA) {
    return registerNewCase(Switch,
                           MA.addressSpace(),
                           "address_space_" + Twine(MA.addressSpace()),
                           CurrentType);
  }

  SwitchInst *registerTypeCase(SwitchInst *Switch, const MetaAddress &MA) {
    const char *TypeName = MetaAddressType::toString(MA.type());
    return registerNewCase(Switch,
                           MA.type(),
                           "type_" + Twine(TypeName),
                           CurrentAddress);
  }

private:
  SwitchInst *getSwitchForLabel(SwitchInst *Parent, uint64_t CaseValue) {
    auto *CaseConstant = caseConstant(Parent, CaseValue);
    auto CaseIt = Parent->findCaseValue(CaseConstant);
    if (CaseIt != Parent->case_default())
      return getNextSwitch(CaseIt);
    else
      return nullptr;
  }

  /// Helper to create a new case in the parent switch and create a new switch
  SwitchInst *registerNewCase(SwitchInst *Switch,
                              uint64_t NewCaseValue,
                              const Twine &NewSuffix,
                              Value *SwitchOn) {
    using BB = BasicBlock;
    auto *NewSwitchBB = BB::Create(Context,
                                   (Switch->getParent()->getName() + "_"
                                    + NewSuffix),
                                   F);

    if (NewBlocksRegistry != nullptr)
      NewBlocksRegistry->push_back(NewSwitchBB);

    ::addCase(Switch, NewCaseValue, NewSwitchBB);
    IRBuilder<> Builder(NewSwitchBB);
    SwitchInst *Result = createSwitch(SwitchOn, Builder);
    if (SetBlockType)
      setBlockType(Result, *SetBlockType);
    return Result;
  }
};

using std::optional;

void PCH::addCaseToDispatcher(SwitchInst *Root,
                              const DispatcherTarget &NewTarget,
                              optional<BlockType::Values> SetBlockType) const {
  auto &[MA, BB] = NewTarget;

  SwitchManager SM(Root,
                   EpochCSV,
                   AddressSpaceCSV,
                   TypeCSV,
                   AddressCSV,
                   SetBlockType);

  SwitchInst *EpochSwitch = Root;
  SwitchInst *AddressSpaceSwitch = nullptr;
  SwitchInst *TypeSwitch = nullptr;
  SwitchInst *AddressSwitch = nullptr;

  // Get or create, step by step, the switches for MA
  AddressSpaceSwitch = SM.getOrCreateAddressSpaceSwitch(EpochSwitch, MA);
  TypeSwitch = SM.getOrCreateTypeSwitch(AddressSpaceSwitch, MA);
  AddressSwitch = SM.getOrCreateAddressSwitch(TypeSwitch, MA);

  // We are the switch of the addresses, add a case targeting BB, if required
  ::addCase(AddressSwitch, MA.address(), BB);
}

void PCH::destroyDispatcher(SwitchInst *Root) const {
  SwitchManager SM(Root, EpochCSV, AddressSpaceCSV, TypeCSV, AddressCSV, {});
  SM.destroy(Root);
}

PCH::DispatcherInfo
PCH::buildDispatcher(DispatcherTargets &Targets,
                     IRBuilder<> &Builder,
                     BasicBlock *Default,
                     std::optional<BlockType::Values> SetBlockType) const {
  DispatcherInfo Result;

  LLVMContext &Context = getContext(Default);

  // Sort by MetaAddress
  std::sort(Targets.begin(),
            Targets.end(),
            [](const DispatcherTarget &LHS, const DispatcherTarget &RHS) {
              return std::less<MetaAddress>()(LHS.first, RHS.first);
            });

  // First of all, create code to load the components of the MetaAddress
  Value *CurrentEpoch = createLoad(Builder, EpochCSV);
  Value *CurrentAddressSpace = createLoad(Builder, AddressSpaceCSV);
  Value *CurrentType = createLoad(Builder, TypeCSV);
  Value *CurrentAddress = createLoad(Builder, AddressCSV);

  SwitchManager SM(Default,
                   CurrentEpoch,
                   CurrentAddressSpace,
                   CurrentType,
                   CurrentAddress,
                   SetBlockType,
                   &Result.NewBlocks);

  // Create the first switch, for epoch
  SwitchInst *EpochSwitch = SM.createSwitch(CurrentEpoch, Builder);
  SwitchInst *AddressSpaceSwitch = nullptr;
  SwitchInst *TypeSwitch = nullptr;
  SwitchInst *AddressSwitch = nullptr;

  // Initially, we need to create a switch at each level
  bool ForceNewSwitch = true;

  MetaAddress Last = MetaAddress::invalid();
  for (const auto &[MA, BB] : Targets) {
    // Extract raw values for the current MetaAddress
    uint64_t Epoch = MA.epoch();
    uint64_t AddressSpace = MA.addressSpace();
    uint64_t Type = MA.type();
    uint64_t Address = MA.address();

    // If it's the first iteration, or any of the components of the
    // MetaAddress has a different value, emit the required switch and new
    // cases

    if (ForceNewSwitch or Epoch != Last.epoch()) {
      AddressSpaceSwitch = SM.registerEpochCase(EpochSwitch, MA);
      ForceNewSwitch = true;
    }

    if (ForceNewSwitch or AddressSpace != Last.addressSpace()) {
      TypeSwitch = SM.registerAddressSpaceCase(AddressSpaceSwitch, MA);
      ForceNewSwitch = true;
    }

    if (ForceNewSwitch or Type != Last.type()) {
      const char *TypeName = MetaAddressType::toString(MA.type());
      AddressSwitch = SM.registerTypeCase(TypeSwitch, MA);
      ForceNewSwitch = true;
    }

    ::addCase(AddressSwitch, Address, BB);

    Last = MA;
    ForceNewSwitch = false;
  }

  Result.Switch = EpochSwitch;

  return Result;
}

static unsigned getMinimumPCAlignment(Triple::ArchType Architecture) {
  switch (Architecture) {
  case Triple::x86:
  case Triple::x86_64:
    return 1;
  case Triple::arm:
  case Triple::systemz:
    return 2;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::aarch64:
    return 4;
  default:
    revng_abort();
  }
}

std::unique_ptr<ProgramCounterHandler>
PCH::create(Triple::ArchType Architecture,
            Module *M,
            const CSVFactory &Factory) {
  auto Alignment = getMinimumPCAlignment(Architecture);

  switch (Architecture) {
  case Triple::arm:
    return ARMProgramCounterHandler::create(M, Factory);

  case Triple::x86_64:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::aarch64:
  case Triple::systemz:
  case Triple::x86:
    return PCOnlyProgramCounterHandler::create(M, Factory, Alignment);

  default:
    revng_abort("Unsupported architecture");
  }

  revng_abort();
}

std::unique_ptr<ProgramCounterHandler>
PCH::fromModule(Triple::ArchType Architecture, Module *M) {
  auto Alignment = getMinimumPCAlignment(Architecture);

  switch (Architecture) {
  case Triple::arm:
    return ARMProgramCounterHandler::fromModule(M);

  case Triple::x86_64:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::aarch64:
  case Triple::systemz:
  case Triple::x86:
    return PCOnlyProgramCounterHandler::fromModule(M, Alignment);

  default:
    revng_abort("Unsupported architecture");
  }

  revng_abort();
}

void PCH::buildHotPath(IRBuilder<> &B,
                       const DispatcherTarget &CandidateTarget,
                       BasicBlock *Default) const {
  auto &[Address, BB] = CandidateTarget;

  auto CreateCmp = [&B](GlobalVariable *CSV, uint64_t Value) {
    Instruction *Load = createLoad(B, CSV);
    Type *LoadType = Load->getType();
    return B.CreateICmpEQ(Load, ConstantInt::get(LoadType, Value));
  };

  std::array<Value *, 4> ToAnd = { CreateCmp(EpochCSV, Address.epoch()),
                                   CreateCmp(AddressSpaceCSV,
                                             Address.addressSpace()),
                                   CreateCmp(TypeCSV, Address.type()),
                                   CreateCmp(AddressCSV, Address.address()) };
  auto *Condition = B.CreateAnd(ToAnd);

  // Emit branches of targets in dedicated basic blocks. As of now,
  // IsolateFunction struggles with basic blocks that are both direct and
  // indirect at the same time. This allows the pass to execute correctly.
  auto *JumpToBB = BasicBlock::Create(B.getContext(), "", BB->getParent());
  auto *JumpToDefault = BasicBlock::Create(B.getContext(), "", BB->getParent());

  BranchInst::Create(BB, JumpToBB);
  BranchInst::Create(Default, JumpToDefault);

  B.CreateCondBr(Condition, JumpToBB, JumpToDefault);
}
