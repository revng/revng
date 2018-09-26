/// \file element.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/Support/Debug.h"

// Local includes
#include "Element.h"

using llvm::Module;

Logger<> SaDiffLog("sa-diff");

RunningStatistics AddressSpaceSizeStats("AddressSpaceSizeStats");

Logger<> SaVerboseLog("sa-verbose");

static size_t combineHash(size_t A, size_t B) {
  return (A << 1 | A >> 31) ^ B;
}

namespace StackAnalysis {

size_t ASID::hash() const {
  return std::hash<uint32_t>()(ID);
}

bool ASSlot::lowerThanOrEqual(const ASSlot &Other) const {
  return cmp<false, true>(Other, nullptr) == 0;
}

template<bool Diff, bool EarlyExit>
unsigned ASSlot::cmp(const ASSlot &Other, const Module *M) const {
  revng_assert(!this->isInvalid() and !Other.isInvalid());

  LoggerIndent<> Y(SaDiffLog);
  bool Result = not(AS.lowerThanOrEqual(Other.AS) && Offset == Other.Offset);

  if (SaDiffLog.isEnabled() && Result && Diff) {
    Other.dump(M, SaDiffLog);
    SaDiffLog << " does not contain ";
    dump(M, SaDiffLog);
    SaDiffLog << DoLog;
  }

  return Result;
}

size_t ASSlot::hash() const {
  return combineHash(std::hash<ASID>()(AS), std::hash<int32_t>()(Offset));
}

namespace Intraprocedural {

bool Value::lowerThanOrEqual(const Value &Other) const {
  return cmp<false, true>(Other, nullptr) == 0;
}

template<bool Diff, bool EarlyExit>
unsigned Value::cmp(const Value &Other, const Module *M) const {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  if (hasDirectContent() && Other.hasDirectContent()) {
    // Force equality
    // TODO: is this correct? shouldn't we assert DirectContent ==
    //       Other.DirectContent?
    ROA((DirectContent.cmp<Diff, EarlyExit>(Other.DirectContent, M)),
        { revng_log(SaDiffLog, "DirectContent vs DirectContent"); });
  }

  // hasDirectContent() && !Other.hasDirectContent() is fine

  // Other has direct content and we don't, it's more specific than us
  ROA(!hasDirectContent() && Other.hasDirectContent(),
      { revng_log(SaDiffLog, "RHS has direct content, LHS doesn't"); });

  // Losing the name is fine, acquiring it is not
  ROA(!hasTag() && Other.hasTag(),
      { revng_log(SaDiffLog, "RHS has tag, LHS doesn't"); });

  ROA(hasTag() && Other.hasTag() && TheTag.greaterThan(Other.TheTag),
      { revng_log(SaDiffLog, "Tag"); });

  return Result;
}

size_t Value::hash() const {
  size_t Result = 0;

  Result = combineHash(Result, hasDirectContent());
  if (hasDirectContent())
    Result = combineHash(Result, std::hash<ASSlot>()(*directContent()));
  else
    Result = combineHash(Result, Result);

  Result = combineHash(Result, hasTag());
  if (hasTag())
    Result = combineHash(Result, std::hash<ASSlot>()(*tag()));
  else
    Result = combineHash(Result, Result);

  return Result;
}

bool AddressSpace::lowerThanOrEqual(const AddressSpace &Other) const {
  return cmp<false, true>(Other, nullptr) == 0;
}

template<bool Diff, bool EarlyExit>
unsigned AddressSpace::cmp(const AddressSpace &Other, const Module *M) const {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  for (auto &P : ASOContent) {
    auto It = Other.ASOContent.find(P.first);

    // Check if Other has it
    if (It != Other.ASOContent.end()) {
      // Check the actual value
      ROA((P.second.cmp<Diff, EarlyExit>(It->second, M)), {
        slot(P.first).dump(M, SaDiffLog);
        SaDiffLog << DoLog;
      });
    }
  }

  for (auto &P : Other.ASOContent) {
    auto It = ASOContent.find(P.first);
    // TODO: assert this matters in the PruneLog
    ROA(It == ASOContent.end() && P.second.hasDirectContent(), {
      slot(P.first).dump(M, SaDiffLog);
      SaDiffLog << " is absent in the LHS and has direct content on the";
      revng_log(SaDiffLog, " RHS");
    });
  }

  return Result;
}

size_t AddressSpace::hash() const {
  size_t Result = 0;

  for (auto &P : ASOContent) {
    Result = combineHash(Result, P.first);
    Result = combineHash(Result, std::hash<Value>()(P.second));
  }

  return Result;
}

std::set<ASSlot> Element::collectSlots(int32_t CSVCount) const {
  ASID CPU = ASID::cpuID();
  std::set<ASSlot> SlotsPool;

  if (State.size() > CPU.id())
    for (auto &P : State[CPU.id()].ASOContent)
      if (P.first < CSVCount)
        SlotsPool.insert(ASSlot::create(CPU, P.first));

  return SlotsPool;
}

template unsigned
Element::cmp<true, false>(const Element &Other, const Module *M) const;

bool Element::lowerThanOrEqual(const Element &Other) const {
  return cmp<false, true>(Other, nullptr) == 0;
}

template<bool Diff, bool EarlyExit>
unsigned Element::cmp(const Element &Other, const Module *M) const {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  if (Other.State.size() == 0)
    return 0;

  if (State.size() == 0)
    return 1;

  revng_assert(State.size() == Other.State.size());

  size_t TotalASCount = State.size();
  for (unsigned I = 0; I < TotalASCount; I++) {
    ROA((State[I].cmp<Diff, EarlyExit>(Other.State[I], M)), {
      ASID(I).dump(SaDiffLog);
      SaDiffLog << DoLog;
    });
  }

  // TODO: we're ignoring FrameSizeAtCallSite and ABI

  return Result;
}

size_t Element::hash() const {
  size_t Result = 0;
  for (const AddressSpace &AS : State)
    Result = combineHash(Result, std::hash<AddressSpace>()(AS));
  return Result;
}

Element &Element::combine(const Element &Other) {
  if (isBottom()) {
    *this = Other.copy();
    return *this;
  }

  revng_assert(State.size() == Other.State.size());
  for (unsigned I = 0; I < State.size(); I++)
    mergeASState(State[I], Other.State[I]);

  return *this;
}

void Element::cleanup() {
  for (AddressSpace &AS : State) {
    for (auto It = AS.ASOContent.begin(); It != AS.ASOContent.end(); /**/) {
      if (const ASSlot *TheTag = It->second.tag()) {
        if (*TheTag == ASSlot::create(AS.ID, It->first)) {
          It = AS.ASOContent.erase(It);
          continue;
        }
      }
      It++;
    }
  }
}

void Element::apply(const Element &Other) {
  revng_assert(State.size() == Other.State.size());

  ASID CPU = ASID::cpuID();
  const AddressSpace &OtherCPU = Other.State[CPU.id()];
  for (auto &P : OtherCPU.ASOContent)
    store(Value::fromSlot(CPU, P.first), P.second);
}

std::set<ASSlot> Element::computeCalleeSavedSlots() const {
  std::set<ASSlot> Result;

  unsigned I = 0;
  for (const AddressSpace &ASS : State) {
    for (auto &P : ASS.ASOContent) {
      // Do we have direct content with a name?
      if (const ASSlot *T = P.second.tag()) {
        // Is the name the same as the current slot?
        ASSlot Slot = ASSlot::create(ASID(I), P.first);
        if (*T == Slot)
          Result.insert(Slot);
      }
    }

    I++;
  }

  return Result;
}

void Element::mergeASState(AddressSpace &ThisState,
                           const AddressSpace &OtherState) {
  // The following implementation can be easily replaced by any other
  // implementation using a data structure allowing to iterate over a sorted
  // pair of <ASO, *> pairs. In particular, instead of a std::map we could use
  // a sorted std::vector of pairs.

  // Iterate in parallel
  auto ThisIt = ThisState.ASOContent.begin();
  auto ThisEndIt = ThisState.ASOContent.end();
  auto OtherIt = OtherState.ASOContent.begin();
  auto OtherEndIt = OtherState.ASOContent.end();
  std::vector<std::pair<int32_t, Value>> NewEntries;

  bool ThisDone = ThisIt == ThisEndIt;
  bool OtherDone = OtherIt == OtherEndIt;
  while (!ThisDone || !OtherDone) {
    Value *ThisContent = nullptr;
    const Value *OtherContent = nullptr;
    Value TmpContent = Value::empty();

    if (ThisDone || (!OtherDone && ThisIt->first > OtherIt->first)) {
      // Only Other has the current offset: create a new default entry for
      // delayed appending in this and merge it with OtherContent
      auto ASO = ASSlot::create(ThisState.id(), OtherIt->first);
      NewEntries.emplace_back(OtherIt->first, ThisState.load(ASO));

      ThisContent = &NewEntries.back().second;
      OtherContent = &OtherIt->second;

      OtherIt++;
    } else if (OtherDone || (!ThisDone && OtherIt->first > ThisIt->first)) {
      // Only this has the current offset: create a default OtherContent and
      // merge with ThisContent
      auto ASO = ASSlot::create(OtherState.id(), ThisIt->first);
      TmpContent = OtherState.load(ASO);

      ThisContent = &ThisIt->second;
      OtherContent = &TmpContent;

      ThisIt++;
    } else {
      // Both have the current offset: update ThisContent with OtherContent
      revng_assert(ThisIt != ThisEndIt && OtherIt != OtherEndIt);
      revng_assert(ThisIt->first == OtherIt->first);

      ThisContent = &ThisIt->second;
      OtherContent = &OtherIt->second;

      ThisIt++;
      OtherIt++;
    }

    // Perform the merge
    ThisContent->combine(*OtherContent);

    ThisDone = ThisIt == ThisEndIt;
    OtherDone = OtherIt == OtherEndIt;
  }

  for (std::pair<int32_t, Value> &P : NewEntries)
    ThisState.ASOContent[P.first] = P.second;

  // Cleanup phase
  for (auto It = ThisState.ASOContent.begin(); It != ThisState.ASOContent.end();
       /**/) {

    if (!It->second.hasDirectContent() && !It->second.hasTag())
      It = ThisState.eraseASO(It);
    else
      It++;
  }
}

} // namespace Intraprocedural

} // namespace StackAnalysis
