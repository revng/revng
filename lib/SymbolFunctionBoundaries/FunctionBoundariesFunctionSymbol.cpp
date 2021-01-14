/// \file FunctionBoundariesFunctionSymbol.cpp
/// \brief ModulePass which executes a function boundaries analysis using the
///        information provided by the debug function symbols.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <sstream>

#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/SymbolFunctionBoundaries/FunctionBoundariesFunctionSymbol.h"

using namespace llvm;

static Logger<> FBFS("function-symbols");

char FunctionBoundariesFunctionSymbol::ID = 0;

namespace {

using RegisterFBFS = RegisterPass<FunctionBoundariesFunctionSymbol>;
RegisterFBFS X("function-boundaries-function-symbol",
               "Execute function boundaries analysis using function symbols",
               false,
               false);

} // namespace

bool FunctionBoundariesFunctionSymbol::runOnModule(Module &M) {
  Function &RootFunction = *M.getFunction("root");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  // Container for the deserialized function labels.
  std::map<uint64_t, std::pair<StringRef, uint64_t>> LabelMap;
  std::map<uint64_t, MDTuple *> FunctionMDMap;

  QuickMetadata QMD(getContext(&M));

  // Obtain the pointer to the named metadata containing the labels.
  const char *MDName = "revng.input.symbol-labels";
  const char *RHFEName = "revng.hint.func.entry";
  const char *RHFMOName = "revng.hint.func.member.of";
  NamedMDNode *FunctionLabelsMD = M.getNamedMetadata(MDName);
  revng_assert(FunctionLabelsMD != nullptr);

  // Iterate over all the serialized labels and add them into the map.
  for (MDNode *FunctionLabelMD : FunctionLabelsMD->operands()) {
    auto *FunctionLabelTuple = cast<MDTuple>(FunctionLabelMD);
    auto Name = QMD.extract<StringRef>(FunctionLabelTuple, 0);
    auto TypeString = QMD.extract<StringRef>(FunctionLabelTuple, 1);
    SymbolType::Values Type = SymbolType::fromName(TypeString);
    auto StartAddress = QMD.extract<uint64_t>(FunctionLabelTuple, 2);
    auto Size = QMD.extract<uint64_t>(FunctionLabelTuple, 3);
    uint64_t EndAddress = StartAddress + Size;

    if (Type == SymbolType::Code) {
      LabelMap[StartAddress] = std::make_pair(Name, EndAddress);

      // Debug info on retrieved labels
      revng_log(FBFS, Name);
      revng_log(FBFS, StartAddress);
      revng_log(FBFS, EndAddress);
    }
  }

  // Create a new metadata for each function which has a corresponding symbol
  for (BasicBlock &BB : RootFunction) {
    if (GCBI.isTranslated(&BB)) {
      uint64_t BBPC = getBasicBlockPC(&BB).asPCOrZero();
      if (LabelMap.count(BBPC) == 1) {
        StringRef FunctionName = BB.getName();
        auto TypeMD = QMD.get("Regular");
        std::vector<Metadata *> ClobberedMDs;
        std::vector<Metadata *> SlotMDs;
        MDTuple *FunctionMD = QMD.tuple({ QMD.get(FunctionName),
                                          QMD.get(BBPC),
                                          TypeMD,
                                          QMD.tuple(ClobberedMDs),
                                          QMD.tuple(SlotMDs) });
        FunctionMDMap[BBPC] = FunctionMD;
      }
    }
  }

  // Attach the function.entry and function.member of metadata to the basic
  // blocks.
  for (BasicBlock &BB : RootFunction) {

    // Attach the metadata only to the translated blocks
    if (GCBI.isTranslated(&BB)) {
      uint64_t BBPC = getBasicBlockPC(&BB).asPCOrZero();
      for (auto &P : LabelMap) {
        uint64_t StartAddress = P.first;
        uint64_t EndAddress = P.second.second;
        if (BBPC == StartAddress) {
          MDTuple *FunctionMD = FunctionMDMap.at(BBPC);
          BB.getTerminator()->setMetadata(RHFEName, FunctionMD);
        }

        if (BBPC >= StartAddress and BBPC <= EndAddress) {
          MDTuple *FunctionMD = FunctionMDMap.at(StartAddress);
          std::vector<Metadata *> MetadataArray;
          auto *Pair = QMD.tuple({ FunctionMD, QMD.get("Invalid") });
          MetadataArray.push_back(Pair);
          BB.getTerminator()->setMetadata(RHFMOName, QMD.tuple(MetadataArray));
        }
      }
    }
  }

  // Intialize the dominator tree over the `root` function.
  DominatorTree DT(RootFunction);

  for (BasicBlock &BB : RootFunction) {
    if (GCBI.isTranslated(&BB)) {

      // Go again over the basic blocks in the root function and if they do not
      // have an associated `revng.func.member.of` metadata associated try to
      // explore the dominator tree going over the parents until we reach a
      // block which has been already tagged. This solution is more elegant than
      // going over the predecessors.
      Instruction *Terminator = BB.getTerminator();
      MDNode *Node = Terminator->getMetadata(RHFMOName);
      if (Node == nullptr) {
        BasicBlock *CurrentNode = &BB;

        // Iterate over the parent nodes stopping when we reach non translated
        // blocks (e.g., the dispatcher).
        while (true) {
          DomTreeNodeBase<BasicBlock> *DomNode = DT.getNode(CurrentNode);
          DomTreeNodeBase<BasicBlock> *ImmediateDom = DomNode->getIDom();
          BasicBlock *ImmediateDomBB = ImmediateDom->getBlock();
          Instruction *DomTerminator = ImmediateDomBB->getTerminator();
          MDNode *ParentMD = DomTerminator->getMetadata(RHFMOName);
          if (ParentMD != nullptr) {
            Terminator->setMetadata(RHFMOName, ParentMD);
            break;
          } else if (GCBI.isTranslated(ImmediateDomBB)) {
            CurrentNode = ImmediateDomBB;
            continue;
          } else {
            break;
          }
        }
      }
    }
  }

  return true;
}
