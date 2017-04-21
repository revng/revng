/// \file functionboundariesdetection.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

// Boost includes
#include <boost/icl/interval_set.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/icl/right_open_interval.hpp>

// LLVM includes
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// Local includes
#include "debug.h"
#include "datastructures.h"
#include "functionboundariesdetection.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"

using namespace llvm;

using std::map;
using std::vector;

class FunctionBoundariesDetectionImpl;

using FBDP = FunctionBoundariesDetectionPass;
using FBD = FunctionBoundariesDetectionImpl;
using interval_set = boost::icl::interval_set<uint64_t>;
using interval = boost::icl::interval<uint64_t>;

char FBDP::ID = 0;
static RegisterPass<FBDP> X("fbdp",
                            "Function Boundaries Detection Pass",
                            true,
                            true);

class FunctionBoundariesDetectionImpl {
public:
  FunctionBoundariesDetectionImpl(Function &F,
                                  JumpTargetManager *JTM) : F(F), JTM(JTM) { }

  map<BasicBlock *, vector<BasicBlock *>> run();

private:
  enum RelationType {
    UnknownRelation = 0,
    Head = 1,
    Fallthrough = 2,
    Jump = 4,
    Return = 8
  };

  enum CFEPReason {
    UnknownReason = 0,
    Callee = 1,
    GlobalData = 2,
    InCode = 4,
    SkippingJump = 8
  };

  class CFEPRelation {
  public:
    CFEPRelation(BasicBlock *CFEP) : CFEP(CFEP), Distance(0), Type(0) { }

    void setType(RelationType T) { Type |= T; }
    bool hasType(RelationType T) const { return Type & T; }

    void setDistance(uint32_t New) { Distance = std::max(Distance, New); }
    bool isSkippingJump() const { return Distance > 0 && hasType(Jump); }
    bool isNonSkippingJump() const { return Distance == 0 && hasType(Jump); }

    BasicBlock *cfep() const { return CFEP; }

    std::string describe() const;

  private:
    BasicBlock *CFEP;
    uint32_t Distance;
    uint32_t Type;
  };

  class CFEP {
  public:
    CFEP() : Reasons(0) { }

    void setReason(CFEPReason Reason) { Reasons |= Reason; }
    bool hasReason(CFEPReason Reason) { return Reasons & Reason; }

  private:
    uint32_t Reasons;
  };

private:
  void initPostDispatcherIt();
  void collectFunctionCalls();
  void collectReturnInstructions();
  void registerBasicBlockAddressRanges();
  interval_set findCoverage(BasicBlock *BB);

  // CFEP related methods
  void collectInitialCFEPSet();
  void cfepProcessPhase1();
  void cfepProcessPhase2();

  /// Associate to each basic block a metadata with the list of functions it
  /// belongs to
  void createMetadata();

  void serialize();

  void setRelation(BasicBlock *CFEP, BasicBlock *Affected, RelationType T) {
    assert(CFEP != nullptr);
    CFEPRelation &Relation = getRelation(CFEP, Affected);
    Relation.setType(T);
  }

  void setDistance(BasicBlock *CFEP, BasicBlock *Affected, uint64_t Distance) {
    assert(CFEP != nullptr);
    const uint64_t Max = std::numeric_limits<uint32_t>::max();
    Distance = std::min(Distance, Max);
    CFEPRelation &Relation = getRelation(CFEP, Affected);
    Relation.setDistance(Distance);
  }

  bool isCFEP(BasicBlock *BB) const { return CFEPs.count(BB); }
  void registerCFEP(BasicBlock *BB, CFEPReason Reason) {
    assert(BB != nullptr);
    CFEPs[BB].setReason(Reason);
    setRelation(BB, BB, Head);
  }

  void filterCFEPs();

  CFEPRelation &getRelation(BasicBlock *CFEP, BasicBlock *Affected) {
    SmallVector<CFEPRelation, 2> &BBRelations = Relations[Affected];
    auto It = std::find_if(BBRelations.begin(),
                           BBRelations.end(),
                           [CFEP] (CFEPRelation &R) {
                             return R.cfep() == CFEP;
                           });
    if (It != BBRelations.end()) {
      return *It;
    } else {
      BBRelations.emplace_back(CFEP);
      return BBRelations.back();
    }
  }

  std::vector<BasicBlock *> cfeps() const {
    std::vector<BasicBlock *> Result;
    Result.reserve(CFEPs.size());
    for (auto &P : CFEPs)
      Result.push_back(P.first);

    return Result;
  }

private:
  Function &F;
  JumpTargetManager *JTM;

  std::map<TerminatorInst *, BasicBlock *> FunctionCalls;
  std::map<BasicBlock *, std::vector<BasicBlock *>> CallPredecessors;
  std::set<uint64_t> ReturnPCs;
  std::set<TerminatorInst *> Returns;
  ilist_iterator<BasicBlock> PostDispatcherIt;
  std::map<BasicBlock *, interval_set> Coverage;

  // CFEP related data
  std::map<BasicBlock *, CFEP> CFEPs;
  std::map<BasicBlock *, SmallVector<CFEPRelation, 2>> Relations;
  OnceQueue<BasicBlock *> CFEPWorkList;
  interval_set Callees;
  std::map<BasicBlock *, std::vector<BasicBlock *>> Functions;
};


void FBD::initPostDispatcherIt() {
  // Skip dispatcher and friends
  auto It = F.begin();
  for (; It != F.end(); It++) {
    if (!It->empty()) {
      if (auto *Call = dyn_cast<CallInst>(&*It->begin())) {
        Function *Callee = Call->getCalledFunction();
        if (Callee != nullptr && Callee->getName() == "newpc")
          break;
      }
    }
  }
  PostDispatcherIt = It;
}

static inline BasicBlock *getBlock(Value *V) {
  return cast<BlockAddress>(V)->getBasicBlock();
}

void FBD::collectFunctionCalls() {
  Module *M = F.getParent();
  Function *FC = M->getFunction("function_call");
  for (User *U : FC->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      BasicBlock *ReturnBB = getBlock(Call->getOperand(1));
      uint32_t ReturnPC = getLimitedValue(Call->getOperand(2));
      auto *Terminator = cast<TerminatorInst>(getNext(Call));
      assert(Terminator != nullptr);
      FunctionCalls[Terminator] = ReturnBB;
      CallPredecessors[ReturnBB].push_back(Call->getParent());
      ReturnPCs.insert(ReturnPC);
    }
  }

  // Mark all the callee basic blocks as such
  for (auto P : FunctionCalls)
    for (BasicBlock *S : P.first->successors())
      if (JTM->isTranslatedBB(S))
        JTM->registerJT(S, JumpTargetManager::Callee);
}

void FBD::collectReturnInstructions() {
  // Detect return instructions

  // TODO: there is a remote possibility that we're mishandling some case here,
  //       in the future we should perform a stack analysis to prove that a
  //       register has not been touched since the function entry.
  for (BasicBlock &BB : make_range(PostDispatcherIt, F.end())) {
    auto *Terminator = BB.getTerminator();

    if (FunctionCalls.count(Terminator) != 0)
      continue;

    bool JumpsToDispatcher = false;
    bool IsReturn = true;

    for (BasicBlock *Successor : Terminator->successors()) {
      assert(!Successor->empty());

      // A return instruction must jump to JTM->anyPC, while all the other
      // successors (if any) must be registered returns addresses
      if (Successor == JTM->anyPC()) {
        JumpsToDispatcher = true;
      } else if (ReturnPCs.count(JTM->getPC(&*Successor->begin()).first) == 0) {
        IsReturn = false;
        break;
      }

    }

    IsReturn &= JumpsToDispatcher;

    if (IsReturn) {
      // TODO: assert that the destnation is the content of a register, or a
      //       load from a memory location at a constant offset from the content
      //       of a register
      Returns.insert(Terminator);
    }

  }
}

/// \brief Register for each basic block the range of addresses it covers
void FBD::registerBasicBlockAddressRanges() {
  // Register the range of addresses covered by each basic block
  for (User *U : F.getParent()->getFunction("newpc")->users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (Call == nullptr)
      continue;

    BasicBlock *BB = Call->getParent();
    uint64_t Address = getLimitedValue(Call->getOperand(0));
    uint64_t Size = getLimitedValue(Call->getOperand(1));
    assert(Address > 0 && Size > 0);

    Coverage[BB] += interval::right_open(Address, Address + Size);
  }
}

interval_set FBD::findCoverage(BasicBlock *BB) {
  auto It = Coverage.find(BB);
  if (It != Coverage.end()) {
    return It->second;
  }

  OnceQueue<BasicBlock *> WorkList;
  WorkList.insert(BB);

  while (!WorkList.empty()) {
    BB = WorkList.pop();

    It = Coverage.find(BB);
    if (It != Coverage.end()) {
      return It->second;
    }

    for (BasicBlock *Predecessor : predecessors(BB))
      if (JTM->isTranslatedBB(Predecessor))
        WorkList.insert(Predecessor);
  }

  llvm_unreachable("Couldn't find basic block");
}

void FBD::collectInitialCFEPSet() {
  // TODO: handle entry points
  // registerCFEP(JTM->getBlockAt(EntryPoint), Callee);

  // Collect initial set of CFEPs
  for (auto &P : *JTM) {
    if (contains(JTM->readRange(), P.first))
      continue;

    const JumpTargetManager::JumpTarget &JT = P.second;

    BasicBlock *CFEPHead = JT.head();
    bool Insert = false;

    DBG("functions", dbg << JT.describe() << "\n");

    if (JT.hasReason(JumpTargetManager::Callee)) {
      registerCFEP(CFEPHead, Callee);

      assert(Coverage.find(CFEPHead) != Coverage.end());
      Callees += Coverage[CFEPHead];

      Insert = true;
    }

    if (JT.hasReason(JumpTargetManager::UnusedGlobalData)) {
      registerCFEP(CFEPHead, GlobalData);
      Insert = true;
    }

    if (JT.hasReason(JumpTargetManager::SETNotToPC)
        && !JT.hasReason(JumpTargetManager::SETToPC)) {
      registerCFEP(CFEPHead, InCode);
      Insert = true;
    }

    if (Insert)
      CFEPWorkList.insert(CFEPHead);
  }
}

void FBD::cfepProcessPhase1() {
  // For each CFEP record which basic block it can reach and how. Then also
  // detect skipping jumps.
  while (!CFEPWorkList.empty()) {
    BasicBlock *CFEP = CFEPWorkList.pop();

    // Find all the basic block it can reach
    OnceQueue<BasicBlock *> WorkList;
    WorkList.insert(CFEP);

    while (!WorkList.empty()) {
      BasicBlock *RelatedBB = WorkList.pop();

      auto FCIt = FunctionCalls.find(RelatedBB->getTerminator());
      if (FCIt != FunctionCalls.end()) {
        // This basic block ends with a function call, proceed with the return
        // address, unless it's a call to a noreturn function.
        if (JTM->noReturn().isNoreturnBasicBlock(RelatedBB)) {
          DBG("nra", dbg << "Stopping at " << getName(RelatedBB) << " since it's a noreturn call\n");
        } else {
          BasicBlock *ReturnBB = FCIt->second;
          setRelation(CFEP, ReturnBB, Return);
          WorkList.insert(ReturnBB);
        }

      } else if (Returns.count(RelatedBB->getTerminator()) == 0) {
        // It's not a return, it's not a function call, it must be a branch part
        // of the ordinary control flow of the function.
        for (BasicBlock *S : successors(RelatedBB)) {
          if (!JTM->isTranslatedBB(S))
            continue;

          // TODO: track fallthrough
          setRelation(CFEP, S, Jump);
          WorkList.insert(S);
        }

      }
    }

    // Compute distance of jumps

    // For each basic block look at his Jump successors
    for (BasicBlock *BB : WorkList.visited()) {
      TerminatorInst *T = BB->getTerminator();
      if (FunctionCalls.count(T) != 0 || Returns.count(T) != 0)
        continue;

      interval_set StartAddressRange = findCoverage(BB);
      uint64_t StartAddress = StartAddressRange.begin()->lower();
      assert(StartAddress != 0);

      for (BasicBlock *S : successors(BB)) {
        if (!JTM->isTranslatedBB(S) || Coverage.count(S) == 0)
          continue;

        assert(Coverage.find(S) != Coverage.end());
        interval_set &DestinationAddressRange = Coverage[S];

        // TODO: why this?
        if (DestinationAddressRange.size() == 0)
          continue;

        uint64_t DestinationAddress = DestinationAddressRange.begin()->lower();

        interval_set JumpInterval;
        if (StartAddress <= DestinationAddress)
          JumpInterval += interval::closed(StartAddress, DestinationAddress);
        else
          JumpInterval += interval::closed(DestinationAddress, StartAddress);

        JumpInterval -= StartAddressRange;
        JumpInterval -= DestinationAddressRange;
        JumpInterval &= Callees;
        uint64_t Distance = JumpInterval.size();

        if (Distance > 0) {
          setDistance(CFEP, S, Distance);
          registerCFEP(S, SkippingJump);
          CFEPWorkList.insert(S);
        }

      }

    }
  }
}

void FBD::filterCFEPs() {
  std::map<BasicBlock *, CFEP>::iterator It = CFEPs.begin();
  while (It != CFEPs.end()) {
    BasicBlock *CFEPHead = It->first;
    assert(CFEPHead != nullptr);
    CFEP &C = It->second;
    assert(!C.hasReason(UnknownReason));

    // Keep a CFEP only if its address is taken, it's a callee or all the
    // paths leading there are skipping jumps
    bool Keep = C.hasReason(Callee);
    bool AddressTaken = C.hasReason(GlobalData) || C.hasReason(InCode);

    if (!Keep && AddressTaken) {
      Keep = true;
      // Check no relation of Jump type and 0-distance exist
      for (CFEPRelation &Relation : Relations[CFEPHead])
        Keep = Keep
          && !Relation.isNonSkippingJump()
          && !Relation.hasType(Return);
    }

    if (!Keep && !AddressTaken) {
      auto &CFEPRelations = Relations[CFEPHead];
      Keep = Relations.size() > 1;
      if (Keep)
        for (CFEPRelation &Relation : CFEPRelations)
          Keep = Keep && (Relation.hasType(Head)
                          || Relation.isSkippingJump());
    }

    if (Keep) {
      DBG("functions", {
          dbg << std::hex << "0x" << getBasicBlockPC(CFEPHead)
              << " is a FEP: "
              << " Callee? " << C.hasReason(Callee)
              << " GlobalData? " << C.hasReason(GlobalData)
              << " InCode? " << C.hasReason(InCode)
              << " SkippingJump? " << C.hasReason(SkippingJump)
              << "\n";
        });
      It++;
    } else {
      DBG("functions", {
          dbg << std::hex << "0x" << getBasicBlockPC(CFEPHead)
              << " is a not a FEP:";
          for (CFEPRelation &Relation : Relations[CFEPHead])
            dbg << " {" << Relation.describe() << "}";
          dbg << "\n";
        });
      It = CFEPs.erase(It);
    }
  }

  Relations.clear();
}

void FBD::cfepProcessPhase2() {
  // Find all the basic block it can reach
  for (BasicBlock *CFEP : cfeps()) {
    OnceQueue<BasicBlock *> WorkList;
    WorkList.insert(CFEP);

    while (!WorkList.empty()) {
      BasicBlock *RelatedBB = WorkList.pop();
      assert(JTM->isTranslatedBB(RelatedBB));

      auto FCIt = FunctionCalls.find(RelatedBB->getTerminator());
      if (FCIt != FunctionCalls.end()) {
        BasicBlock *ReturnBB = FCIt->second;
        if (!isCFEP(ReturnBB))
          WorkList.insert(ReturnBB);
      } else if (Returns.count(RelatedBB->getTerminator()) == 0) {
        for (BasicBlock *S : successors(RelatedBB)) {
          if (!JTM->isTranslatedBB(S))
            continue;

          // TODO: doesn't handle the div in div case
          if (!isCFEP(S))
            WorkList.insert(S);
        }

      }
    }

    for (BasicBlock *Member : WorkList.visited())
      Functions[CFEP].push_back(Member);
  }
}

void FBD::createMetadata() {
  LLVMContext &Context = getContext(&F);

  // We first compute the list of all the functions each basic block belongs to,
  // so we don't have to create a huge number of metadata which are never
  // deleted (?)
  std::map<BasicBlock *, std::vector<Metadata *>> ReversedFunctions;

  for (auto &P : Functions) {
    BasicBlock *Header = P.first;
    auto *Name = MDString::get(Context, getName(Header));
    MDTuple *FunctionMD = MDNode::get(Context, { Name });

    Instruction *Terminator = Header->getTerminator();
    assert(Terminator != nullptr);
    Terminator->setMetadata("func.entry", FunctionMD);

    for (BasicBlock *Member : P.second)
      ReversedFunctions[Member].push_back(FunctionMD);

  }

  // Associate the terminator of each basic block with the previously created
  // metadata node
  for (auto &P : ReversedFunctions) {
    BasicBlock *BB = P.first;
    if (!BB->empty() ) {
      Instruction *Terminator = BB->getTerminator();
      assert(Terminator != nullptr);
      auto *FuncMDs =  MDTuple::get(Context, ArrayRef<Metadata *>(P.second));
      Terminator->setMetadata("func.member.of", FuncMDs);
    }
  }

}

map<BasicBlock *, vector<BasicBlock *>> FBD::run() {
  assert(JTM != nullptr);

  initPostDispatcherIt();

  collectFunctionCalls();

  collectReturnInstructions();

  // TODO: move this code in JTM
  JTM->setCFGForm(JumpTargetManager::NoFunctionCallsCFG);
  JTM->noReturn().computeKillerSet(CallPredecessors, Returns);
  JTM->setCFGForm(JumpTargetManager::SemanticPreservingCFG);

  registerBasicBlockAddressRanges();

  collectInitialCFEPSet();

  cfepProcessPhase1();

  filterCFEPs();

  cfepProcessPhase2();

  createMetadata();

  return std::move(Functions);
}

std::string FBD::CFEPRelation::describe() const {
  std::stringstream SS;
  SS << getName(CFEP)
     << " Distance: " << Distance;

  if (hasType(UnknownRelation))
    SS << " UnknownRelation";
  if (hasType(Head))
    SS << " Head";
  if (hasType(Fallthrough))
    SS << " Fallthrough";
  if (hasType(Jump))
    SS << " Jump";
  if (hasType(Return))
    SS << " Return";

  return SS.str();
}

bool FBDP::runOnFunction(Function &F) {
  FBD Impl(F, JTM);
  Functions = Impl.run();
  serialize();
  return false;
}

void FBDP::serialize() const {
  if (SerializePath.size() == 0)
    return;

  // Emit results
  std::ofstream Output(SerializePath);
  Output << "index,start,end\n";
  for (auto &P : Functions) {
    for (BasicBlock *BB : P.second) {
      for (Instruction &I : *BB) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          Function *Callee = Call->getCalledFunction();
          if (Callee != nullptr && Callee->getName() == "newpc") {
            uint64_t StartPC = getLimitedValue(Call->getArgOperandUse(0));
            uint64_t Size = getLimitedValue(Call->getArgOperandUse(1));
            uint64_t EndPC = StartPC + Size;
            Output << std::dec << getName(P.first) << ","
                   << "0x" << std::hex << StartPC << ","
                   << "0x" << std::hex << EndPC << "\n";
          }
        }
      }
    }
  }
}
