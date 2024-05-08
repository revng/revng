/// \file ValueMaterializerPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/Support/Debug.h"
#include "revng/ValueMaterializer/ValueMaterializer.h"

#include "JumpTargetManager.h"
#include "ValueMaterializerPass.h"

using namespace llvm;

cl::list<uint64_t> DumpValueMaterializerAt("dump-vm-at", cl::ZeroOrMore);
cl::opt<bool> DumpValueMaterializer("dump-all-vm");

using SDMO = StaticDataMemoryOracle;

SDMO::StaticDataMemoryOracle(const DataLayout &DL,
                             JumpTargetManager &JTM,
                             const MetaAddress::Features &Features) :
  JTM(JTM), Features(Features) {
  // Read the value using the endianness of the destination architecture,
  // since, if there's a mismatch, in the stack we will also have a byteswap
  // instruction
  IsLittleEndian = DL.isLittleEndian();
}

MaterializedValue StaticDataMemoryOracle::load(uint64_t LoadAddress,
                                               unsigned LoadSize) {
  auto Address = MetaAddress::fromGeneric(LoadAddress, Features);
  return JTM.readFromPointer(Address, LoadSize, IsLittleEndian);
}

static void demoteOrToAdd(Function &F) {
  using namespace llvm;

  auto &DL = F.getParent()->getDataLayout();
  std::set<Instruction *> ToReplace;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::Or) {

        Value *LHS = I.getOperand(0);
        Value *RHS = I.getOperand(1);
        const APInt &LHSZeros = computeKnownBits(LHS, DL).Zero;
        const APInt &RHSZeros = computeKnownBits(RHS, DL).Zero;

        if ((~RHSZeros & ~LHSZeros).isNullValue())
          ToReplace.insert(&I);
      }
    }
  }

  for (Instruction *I : ToReplace) {
    I->replaceAllUsesWith(BinaryOperator::Create(Instruction::Add,
                                                 I->getOperand(0),
                                                 I->getOperand(1),
                                                 Twine(),
                                                 I));
    eraseFromParent(I);
  }
}

PreservedAnalyses ValueMaterializerPass::run(Function &F,
                                             FunctionAnalysisManager &FAM) {
  using namespace llvm;

  llvm::EliminateUnreachableBlocks(F, nullptr, false);

  demoteOrToAdd(F);

  // Early exit in case nothing was marked
  Function *Marker = F.getParent()->getFunction(MarkerName);
  if (Marker == nullptr)
    return PreservedAnalyses::all();

  auto &LVI = FAM.getResult<LazyValueAnalysis>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);

  BasicBlock *Entry = &F.getEntryBlock();
  SwitchInst *Terminator = cast<SwitchInst>(Entry->getTerminator());
  BasicBlock *Dispatcher = Terminator->getDefaultDest();

  auto GetConstantArgument = [](CallBase *Call, unsigned Index) {
    return cast<ConstantInt>(Call->getArgOperand(Index))->getLimitedValue();
  };

  for (CallBase *Call : callersIn(Marker, &F)) {
    // Decode arguments
    revng_assert(Call->arg_size() >= 4);
    Value *ToTrack = Call->getArgOperand(0);
    uint64_t MaxPhiLike = GetConstantArgument(Call, 1);
    uint64_t MaxLoad = GetConstantArgument(Call, 2);

    auto Oracle = static_cast<Oracle::Values>(GetConstantArgument(Call, 3));
    revng_assert(Oracle < Oracle::Count);

    auto AddressAsString = extractFromConstantStringPtr(Call->getArgOperand(4));
    auto Address = MetaAddress::fromString(AddressAsString);
    revng_assert(Address.isValid());
    uint64_t CurrentAddress = Address.address();

    MaterializedValues Values;
    DataFlowGraph::Limits Limits(MaxPhiLike, MaxLoad);
    auto Results = ValueMaterializer::getValuesFor(Call,
                                                   ToTrack,
                                                   MO,
                                                   LVI,
                                                   DT,
                                                   Limits,
                                                   Oracle);
    if (Results.values())
      Values = std::move(*Results.values());

    if (DumpValueMaterializer
        or count(DumpValueMaterializerAt, CurrentAddress) > 0) {
      // User asked to dump information about this address
      dbg << "Values produced by ValueMaterializer for " << getName(ToTrack)
          << " at " << Address.toString() << ":\n";
      for (const MaterializedValue &V : Values) {
        dbg << "  ";
        V.dump(dbg);
        dbg << "\n";
      }

      dbg << "Dumping graphs\n";
      Results.dataFlowGraph().dump();
      AdvancedValueInfoMFI::dump(&Results.cfeg(), Results.mfiResult());
    }

    //
    // Create a revng.avi metadata containing the type of instruction and
    // all the possible values we identified
    //
    QuickMetadata QMD(getContext(&F));
    std::vector<Metadata *> ValuesMD;
    ValuesMD.reserve(Values.size());
    for (const MaterializedValue &V : Values) {
      // TODO: we are we ignoring those with symbols
      auto Offset = V.value();
      std::string SymbolName;
      if (V.hasSymbol())
        SymbolName = V.symbolName();

      ValuesMD.push_back(QMD.tuple({ QMD.get(SymbolName), QMD.get(Offset) }));
    }

    Call->setMetadata("revng.avi", QMD.tuple(ValuesMD));
  }

  return PreservedAnalyses::all();
}
