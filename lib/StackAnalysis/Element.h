#ifndef ELEMENT_H
#define ELEMENT_H

// Standard includes
#include <set>

// Local libraries includes
#include "revng/ADT/LazySmallBitVector.h"
#include "revng/Support/Statistics.h"

// Local includes
#include "ASSlot.h"
#include "BasicBlockInstructionPair.h"

/// \brief Average number of slots tracked by an address space
extern RunningStatistics AddressSpaceSizeStats;

extern Logger<> SaVerboseLog;

namespace StackAnalysis {

namespace Intraprocedural {

/// \brief A Value represents the value associated by the analysis to an SSA
///        value/slot
///
/// A Value tracks two things: the actual content of an SSA value in a certain
/// moment (according to the expressive power of our analysis) and/or a "tag",
/// i.e., the fact that this value contains the value that the ASSlot associated
/// to the tag contained at function entry. This useful, e.g., to detect
/// callee-saved registers or if an indirect jump is targeting the value saved
/// in the link register.
class Value {
private:
  ASSlot DirectContent;
  ASSlot TheTag;

public:
  Value() : DirectContent(ASSlot::invalid()), TheTag(ASSlot::invalid()) {}

  static Value empty() { return Value(); }

  static Value fromSlot(ASSlot Slot) {
    Value Result;
    Result.DirectContent = Slot;
    Result.TheTag = ASSlot::invalid();
    return Result;
  }

  static Value fromSlot(ASID ID, int32_t Offset) {
    return fromSlot(ASSlot::create(ID, Offset));
  }

  static Value fromTag(ASSlot TheTag) {
    Value Result;
    Result.DirectContent = ASSlot::invalid();
    Result.TheTag = TheTag;
    return Result;
  }

public:
  bool hasDirectContent() const { return !DirectContent.isInvalid(); }

  bool hasTag() const { return not TheTag.isInvalid(); }

  bool isEmpty() const { return not(hasDirectContent() || hasTag()); }

  bool operator==(const Value &Other) const {
    return DirectContent == Other.DirectContent && TheTag == Other.TheTag;
  }

  bool operator!=(const Value &Other) const { return !(*this == Other); }

  size_t hash() const;

  /// \brief Perform a comparison according to the analysis' lattice
  bool lowerThanOrEqual(const Value &Other) const;

  template<bool Diff, bool EarlyExit>
  unsigned cmp(const Value &Other, const llvm::Module *M) const;

  bool greaterThan(const Value &Other) const {
    return !this->lowerThanOrEqual(Other);
  }

  Value &combine(const Value &Other) {
    // If direct content is different go to top (invalid)
    if (DirectContent != Other.DirectContent)
      DirectContent = ASSlot::invalid();

    if (Other.TheTag.isInvalid() || TheTag != Other.TheTag)
      TheTag = ASSlot::invalid();

    return *this;
  }

  const ASSlot *directContent() const {
    if (DirectContent.isInvalid())
      return nullptr;
    else
      return &DirectContent;
  }

  const ASSlot *tag() const {
    if (TheTag.isInvalid())
      return nullptr;
    else
      return &TheTag;
  }

  // TODO: Handle size of the offset
  bool add(int32_t Addend) {
    if (!hasDirectContent())
      return false;

    DirectContent.add(Addend);
    return true;
  }

  bool mask(uint64_t Operand) {
    if (!hasDirectContent())
      return false;

    DirectContent.mask(Operand);
    return true;
  }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    if (hasTag())
      TheTag.dump(M, Output);

    if (hasDirectContent())
      DirectContent.dump(M, Output);
    else
      Output << " T";
  }
};

/// \brief Class representing the content of an address space
///
/// An address space is composed by a set of <Offset, Value> pairs recording
/// what are the possible values of the slot at the given offset.
class AddressSpace {
  friend class Element;

public:
  using Container = std::map<int32_t, Value>;

private:
  /// Address space identifier
  ASID ID;
  /// Map associating an offset within the address space with a Value
  Container ASOContent;

public:
  AddressSpace(ASID ID) : ID(ID) {}

  AddressSpace(const AddressSpace &) = default;
  AddressSpace &operator=(const AddressSpace &) = default;
  AddressSpace(AddressSpace &&) = default;
  AddressSpace &operator=(AddressSpace &&) = default;

  ~AddressSpace() { AddressSpaceSizeStats.push(ASOContent.size()); }

  using ASOContentIt = Container::iterator;
  ASOContentIt eraseASO(ASOContentIt It) {
    revng_assert(!It->second.hasDirectContent());
    return ASOContent.erase(It);
  }

  bool operator==(const AddressSpace &Other) const {
    return ASOContent == Other.ASOContent;
  }

  bool operator!=(const AddressSpace &Other) const { return !(*this == Other); }

  /// \brief Perform a comparison according to the analysis' lattice
  bool lowerThanOrEqual(const AddressSpace &Other) const;

  template<bool Diff, bool EarlyExit>
  unsigned cmp(const AddressSpace &Other, const llvm::Module *M) const;

  size_t hash() const;

  bool greaterThan(const AddressSpace &Other) const {
    return not this->lowerThanOrEqual(Other);
  }

  bool contains(int32_t Offset) const { return ASOContent.count(Offset) != 0; }

  void set(int32_t Offset, Value V) { ASOContent[Offset] = V; }

  ASID id() const { return ID; }
  ASSlot slot(int32_t Offset) const { return ASSlot::create(ID, Offset); }

  Container::const_iterator begin() const { return ASOContent.begin(); }
  Container::const_iterator end() const { return ASOContent.end(); }

  /// \brief Handle loading from a specific slot
  Value load(ASSlot Address) const {
    revng_assert(Address.addressSpace() == ID);

    // If we can load from it, return the result right away, otherwise return a
    // value tagged with the requested address
    if (const Value *LoadedASSlot = get(Address.offset())) {
      return *LoadedASSlot;
    } else {
      // We're loading from a specific location in TargetAS, but we have no
      // recorded information about that location
      return Value::fromTag(Address);
    }
  }

  /// \brief Return the number of slots available in this state
  size_t size() const { return ASOContent.size(); }

  bool verify(ASID StateID) const { return StateID == ID; }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    ID.dump(Output);
    Output << ":";

    for (auto &P : ASOContent) {
      Output << "\n    ";
      ASSlot::dumpOffset(M, ID, P.first, Output);
      Output << ": ";
      P.second.dump(M, Output);
    }
  }

private:
  const Value *get(int32_t Offset) const {
    auto It = ASOContent.find(Offset);
    if (It == ASOContent.end())
      return nullptr;
    else
      return &It->second;
  }
};

/// \brief Represents an element of the lattice of the stack analysis
///
/// This class basically keeps the state of all the address spaces being
/// considered in the current analysis.
class Element {
public:
  using Container = llvm::SmallVector<AddressSpace, 2>;

private:
  // The following vector is indexed with ASID
  Container State;
  std::map<CallSite, llvm::Optional<int32_t>> FrameSizeAtCallSite;

private:
  Element() {}

public:
  /// \brief Create a bottom element, which tracks nothing
  static Element bottom() { return Element(); }

  /// \brief Create a regular element, which tracks the CPU and stack state
  static Element initial() {
    Element Result;

    unsigned Count = ASID::stackID().id() + 1;
    Result.State.reserve(Count);
    for (unsigned I = 0; I < Count; I++)
      Result.State.emplace_back(ASID(I));

    return Result;
  }

  Element(const Element &Other) = delete;
  Element &operator=(const Element &Other) = delete;

  Element(Element &&Other) = default;
  Element &operator=(Element &&Other) = default;

  /// \note Copy constructor has been deleted, so that we don't accidentally
  ///       call it. Use this method instead.
  Element copy() const {
    Element Result;
    Result.State = State;
    Result.FrameSizeAtCallSite = FrameSizeAtCallSite;
    return Result;
  }

  bool operator==(const Element &Other) const {
    // TODO: we're ignoring FrameSizeAtCallSite
    return State == Other.State;
  }

  bool operator!=(const Element &Other) const { return !(*this == Other); }

  /// \brief Perform a comparison according to the analysis' lattice
  bool lowerThanOrEqual(const Element &Other) const;

  bool greaterThan(const Element &RHS) const {
    return !this->lowerThanOrEqual(RHS);
  }

  bool equal(const Element &RHS) const {
    return this->lowerThanOrEqual(RHS) && RHS.lowerThanOrEqual(*this);
  }

  size_t hash() const;

  /// \brief Performs a comparison with \p Other
  ///
  /// \tparam Diff should the differences be printed to dbg?
  /// \tparam EarlyExit should the comparison stop at the first difference?
  template<bool Diff, bool EarlyExit>
  unsigned cmp(const Element &Other, const llvm::Module *M) const;

  bool isBottom() const { return State.size() == 0; }

  /// \brief Combine this lattice element with \p Other
  Element &combine(const Element &Other);

  /// \brief Remove all the slots that say that they contain their initial value
  void cleanup();

  bool addressSpaceContainsTag(ASID AddressSpace, const ASSlot *TheTag) const {
    for (auto &P : State[AddressSpace.id()].ASOContent)
      if (P.second.hasTag() && *P.second.tag() == *TheTag)
        return true;

    return false;
  }

  /// \brief Apply to this context the given store log
  void apply(const Element &StoreLog);

  std::set<int32_t> stackArguments(int32_t CallerStackSize) const {
    std::set<int32_t> Result;
    if (State.size() > 0)
      for (auto &P : State[ASID::stackID().id()].ASOContent)
        if (P.first >= 0)
          Result.insert(P.first - CallerStackSize);

    return Result;
  }

  /// \brief Update the element after a store of \p StoredValue has been
  ///        performed to \p Address
  void store(Value Address, Value StoredValue) {
    if (SaVerboseLog.isEnabled()) {
      // TODO: get module
      SaVerboseLog << "Storing ";
      StoredValue.dump(nullptr, SaVerboseLog);
      SaVerboseLog << " to ";
      Address.dump(nullptr, SaVerboseLog);
      SaVerboseLog << DoLog;
    }

    // Does target have a direct component?
    if (const ASSlot *AddressASO = Address.directContent()) {
      ASID TargetASID = AddressASO->addressSpace();
      State[TargetASID.id()].set(AddressASO->offset(), StoredValue);
    }
  }

  /// \brief Return the content of \p TargetAddress according to this Element
  Value load(const Value &TargetAddress) const {
    // Does target have a direct component?
    if (const ASSlot *ASO = TargetAddress.directContent())
      return State[ASO->addressSpace().id()].load(*ASO);

    return Value::empty();
  }

  /// \brief begin iterator for the states handled by this lattice element
  Container::const_iterator begin() const { return State.begin(); }
  Container::const_iterator end() const { return State.end(); }

  /// \brief Verify that this Element is coherent
  bool verify() const {
    unsigned ID = 0;
    for (const AddressSpace &ASS : State)
      if (not ASS.verify(ASID(ID++)))
        return false;

    return true;
  }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    for (const AddressSpace &ASS : State) {
      ASS.dump(M, Output);
      Output << "\n";
    }
  }

  /// \brief Collect all the slots about which we have information
  std::set<ASSlot> collectSlots(int32_t CSVCount) const;

  /// \brief Identify the explicitly callee saved slots
  std::set<ASSlot> computeCalleeSavedSlots() const;

private:
  /// \brief Implement the combine for AddressSpace
  void mergeASState(AddressSpace &ThisState, const AddressSpace &OtherState);
};

} // namespace Intraprocedural

} // namespace StackAnalysis

namespace std {

template<>
struct hash<StackAnalysis::Intraprocedural::Element> {
  size_t operator()(const StackAnalysis::Intraprocedural::Element &K) const {
    return K.hash();
  }
};

template<>
struct hash<StackAnalysis::Intraprocedural::AddressSpace> {
  size_t
  operator()(const StackAnalysis::Intraprocedural::AddressSpace &K) const {
    return K.hash();
  }
};

template<>
struct hash<StackAnalysis::Intraprocedural::Value> {
  size_t operator()(const StackAnalysis::Intraprocedural::Value &K) const {
    return K.hash();
  }
};

} // namespace std

#endif // ELEMENT_H
