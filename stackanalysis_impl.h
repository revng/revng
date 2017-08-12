#ifndef _STACKANALYSIS_IMPL_H
#define _STACKANALYSIS_IMPL_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>

// LLVM includes
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// Boost includes
#include <boost/iterator/transform_iterator.hpp>
#include <boost/variant.hpp>

// Local includes
#include "debug.h"
#include "lazysmallbitvector.h"

// We don't put using namespace llvm since it would create conflicts with the
// local definition of Value
using llvm::BasicBlock;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::Module;
using llvm::Optional;

namespace StackAnalysis {

struct LoadStoreLog;
class ASIndexer;

/// \brief Identifier of an address space
class ASID {
public:

  ASID(uint32_t ID) : ID(ID) { }

  // TODO: I don't like the term virtual
  enum {
    InvalidAddressSpaceID = std::numeric_limits<uint32_t>::max(),
    DeadStackID = InvalidAddressSpaceID - 1, ///< Value representing a stack
                                             ///  that no longer exists (the
                                             ///  function returned)
    FirstVirtualID = DeadStackID,
    MaximumStackID = FirstVirtualID - 1, ///< The highest possible index of a
                                         ///  stack address space
    GlobalAddressSpaceID = 0, ///< The address space representing global
                              ///  variables, code and absolute addresses
    CPUAddressSpaceID, ///< The address space representing the CPU state,
                       ///  registers in particular
    RestOfTheStackID, ///< The address space representing all the stack address
                      ///  spaces above the last one we're explicitly tracking
    LastStackID ///< The topmost stack we're tracking (SP0)
  };

  static ASID invalidID() {
    return ASID(InvalidAddressSpaceID);
  }

  static ASID globalID() {
    return ASID(GlobalAddressSpaceID);
  }

  static ASID cpuID() {
    return ASID(CPUAddressSpaceID);
  }

  static ASID lastStackID() {
    return ASID(LastStackID);
  }

  static ASID deadStackID() {
    return ASID(DeadStackID);
  }

  static ASID restOfTheStackID() {
    return ASID(RestOfTheStackID);
  }

  static ASID maximumStackID() {
    return ASID(MaximumStackID);
  }

  uint32_t id() const { return ID; }

  bool operator<(const ASID &Other) const { return ID < Other.ID; }
  bool operator==(const ASID &Other) const { return ID == Other.ID; }
  bool operator!=(const ASID &Other) const {
    return !(*this == Other);
  }

  bool lowerThanOrEqual(const ASID &Other, ASID Last) const {
    assert(ID != InvalidAddressSpaceID && ID != DeadStackID);
    return ID == Other.ID || (Other.ID == RestOfTheStackID && ID > Last.id());
  }

  bool greaterThan(const ASID &Other, ASID Last) const {
    return !(this->lowerThanOrEqual(Other, Last));
  }

  bool verify(ASID Last) const {
    return isVirtual() || ID <= Last.ID;
  }

  void dump() const {
    dump(dbg);
  }

  void dump(std::ostream &Output) const {
    switch(ID) {
    case InvalidAddressSpaceID:
      Output << "INV";
      break;
    case GlobalAddressSpaceID:
      Output << "GLB";
      break;
    case CPUAddressSpaceID:
      Output << "CPU";
      break;
    case RestOfTheStackID:
      Output << "RST";
      break;
    case DeadStackID:
      Output << "DED";
      break;
    default:
      Output << "SP" << ID - LastStackID;
      break;
    }
  }

  bool isStack() const {
    assert(ID != InvalidAddressSpaceID && ID != DeadStackID);
    return ID >= LastStackID;
  }

  bool isDeadStack() const { return ID == DeadStackID; }

  bool isVirtual() const { return ID >= FirstVirtualID; }

  // TODO: switch from uint32_t to ASID
  ASID getCallerStack(uint32_t Last) const {
    assert(isStack());
    return shiftAddressSpaces(+1).cap(Last);
  }

  ASID shiftAddressSpaces(int N) const {
    assert(N != 0);
    if (!isDeadStack() && isStack()) {
      if (ID + N < LastStackID)
        return ASID::deadStackID();
      else
        return ASID(ID + N);
    } else {
      return *this;
    }
  }

  // TODO: switch from uint32_t to ASID
  ASID cap(uint32_t Last) const {
    if (ID > Last)
      return ASID::restOfTheStackID();
    else
      return *this;
  }

private:
  uint32_t ID;
};

/// \brief Instance of an invalid address space ID to use in default arguments
static const ASID InvalidAS = ASID::invalidID();

/// \brief Class representing a set of address space IDs
class ASSet {
public:
  using Container = LazySmallBitVector;
  template<typename F, typename I>
  using transform_iterator = boost::transform_iterator<F, I>;
  template<typename A>
  using function = std::function<A>;
  using const_iterator = transform_iterator<function<ASID(unsigned)>,
                                            typename Container::const_iterator>;

public:
  /// \brief Factory method to build an ASSet composed by a single address space
  static ASSet singleElement(ASID ID) {
    ASSet Result;
    Result.add(ID);
    return Result;
  }

  /// \brief Augment this ASSet with address space \p ID
  void add(ASID ID) { Arguments.set(wrap(ID)); }

  bool operator[](ASID ID) const { return Arguments[wrap(ID)]; }

  /// \brief Return whether ID is contained in this ASSet (given \p Last)
  bool contains(ASID ID, ASID Last) const {
    return (*this)[ID] || (hasRestOfTheStack() && ID.id() > Last.id());
  }

  bool lowerThanOrEqual(const ASSet &Other, ASID Last) const {
    LazySmallBitVector NonCommon = Other.Arguments;
    NonCommon &= Arguments;
    NonCommon ^= Arguments;

    unsigned RequiredBits = NonCommon.requiredBits();
    if (RequiredBits == 0)
      return true;

    unsigned FirstBit = RequiredBits - 1;
    return Other.hasRestOfTheStack() && unwrap(FirstBit).id() > Last.id();
  }

  bool greaterThan(const ASSet &Other, ASID Last) const {
    return !this->lowerThanOrEqual(Other, Last);
  }

  bool operator==(const ASSet &Other) const {
    return Arguments == Other.Arguments;
  }

  bool operator!=(const ASSet &Other) const {
    return !(*this == Other);
  }

  ASSet &combine(const ASSet &Other, ASID Last=InvalidAS) {
    size_t LastIndex = wrap(Last);
    bool HasRestOfTheStack = hasRestOfTheStack()
      || Other.hasRestOfTheStack()
      || (!Last.isVirtual()
          && (Arguments.findNext(LastIndex + 1) != 0
              || Other.Arguments.findNext(LastIndex + 1) != 0));

    // Combine the bit vectors
    Arguments |= Other.Arguments;

    // Drop all those above Last
    if (!Last.isVirtual())
      Arguments.zero(wrap(Last) + 1);

    // Enable RestOfTheStack if necessary
    if (HasRestOfTheStack)
      add(ASID::restOfTheStackID());

    return *this;
  }

  ASSet &drop(ASID ID) {
    auto ToDrop = ASSet::singleElement(ID);
    return drop(ToDrop);
  }

  ASSet &drop(const ASSet &Other) {
    // (this & Other) ^ this
    LazySmallBitVector Tmp = Arguments;
    Tmp &= Other.Arguments;
    Arguments ^= Tmp;
    return *this;
  }

  bool empty() const { return Arguments.isZero(); }

  const_iterator begin() const {
    return const_iterator(Arguments.begin(), unwrap);
  }

  const_iterator end() const {
    return const_iterator(Arguments.end(), unwrap);
  }

  void shiftAddressSpaces(int N) {
    shiftAddressSpaces(N, Arguments);
  }

  static void shiftAddressSpaces(int N, LazySmallBitVector &Arguments) {
    assert(N != 0);

    LazySmallBitVector Saved = Arguments;
    static LazySmallBitVector Mask = getMask();
    Saved &= Mask;

    if (N < 0) {
      N = -N;
      // This 1-based
      int FirstSetStackIndex = Arguments.findNext(NonStackCount);
      bool HasDED = FirstSetStackIndex != 0
        && FirstSetStackIndex - NonStackCount <= N;
      // Suppose NonStackCount == 3
      // 00010 FirstSetStackIndex == 4, (4 - 3 <= 1) == true
      // 00001 FirstSetStackIndex == 5, (5 - 3 <= 1) == false

      // Shift right, blank, restore
      Arguments >>= N;
      Arguments.zero(0, NonStackCount);
      Arguments |= Saved;
      if (HasDED)
        Arguments.set(wrap(ASID::deadStackID()));
    } else {
      // Blank, shift left, restore
      Arguments.zero(0, NonStackCount);
      Arguments <<= N;
      Arguments |= Saved;
    }
  }

  void capAddressSpaces(uint32_t Last) {
    bool HasRestOfTheStack = hasRestOfTheStack()
      || (Arguments.requiredBits() - 1) > Last;

    Arguments.zero(wrap(ASID(Last)) + 1);

    if (HasRestOfTheStack)
      add(ASID::restOfTheStackID());
  }

  bool verify(ASID Last) const {
    for (ASID ID : *this) {
      if (ID == ASID::cpuID())
        return false;

      if (!ID.verify(Last))
        return false;
    }

    return true;
  }

  void dump() const {
    dump(dbg);
  }

  void dump(std::ostream &Output) const {
    Output << "{";
    for (ASID ID : *this) {
      Output << " ";
      ID.dump(Output);
    }
    Output << " }";
  }

private:
  bool hasRestOfTheStack() const {
    return (*this)[ASID::restOfTheStackID()];
  }

  static const int VirtualCount = -ASID::FirstVirtualID;
  static const int NonStackCount = VirtualCount + ASID::LastStackID;

  static ASID unwrap(unsigned V) {
    return ASID(V - VirtualCount);
  }

  static unsigned wrap(ASID ID) {
    // We want a slot in the bit vector for each regular address space plus all
    // the virtual address spaces (i.e., with a value greater than
    // ASID::FirstVirtualID)
    return ID.id() + VirtualCount;
  }

  static LazySmallBitVector getMask() {
    LazySmallBitVector Result;
    for (unsigned I = 0; I < NonStackCount; I++)
      Result.set(I);
    return Result;
  }

private:
  LazySmallBitVector Arguments;
};

/// \brief Class representing the address of an address space slot
class ASSlot {
public:
  ASSlot(ASID ID, int32_t Offset) : AS(ID), Offset(Offset) { }

  static ASSlot invalid() {
    return ASSlot(ASID::invalidID(), 0);
  }

  bool lowerThanOrEqual(const ASSet &Other, ASID Last) const {
    return Other.contains(AS, Last);
  }

  bool lowerThanOrEqual(const ASSlot &Other, ASID Last) const {
    return AS.lowerThanOrEqual(Other.AS, Last) && Offset == Other.Offset;
  }

  bool greaterThan(const ASSlot &Other, ASID Last) const {
    return !(this->lowerThanOrEqual(Other, Last));
  }

  bool operator==(const ASSlot &Other) const {
    return std::tie(AS, Offset) == std::tie(Other.AS, Other.Offset);
  }

  bool operator!=(const ASSlot &Other) const {
    return !(*this == Other);
  }

  bool operator<(const ASSlot &Other) const {
    auto ThisTuple = std::make_tuple(AS.id(), Offset);
    auto OtherTuple = std::make_tuple(Other.AS.id(), Other.Offset);
    return ThisTuple < OtherTuple;
  }

  int32_t offset() const { return Offset; }
  ASID addressSpace() const { return AS; }
  bool isInvalid() const { return AS == ASID::invalidID(); }

  /// \brief Add a constant to the offset associated with this slot
  void add(int32_t Addend) { Offset += Addend; }

  /// \brief Mask the offset associated to this slot with a value
  void mask(uint64_t Operand) { Offset = Offset & Operand; }

  ASSet flatten() const {
    if (isInvalid())
      return ASSet();
    else
      return ASSet::singleElement(addressSpace());
  }

  void shiftAddressSpaces(int N) {
    AS = AS.shiftAddressSpaces(N);
  }

  void capAddressSpaces(uint32_t Last) {
    AS = AS.cap(Last);
  }

  bool verify(ASID Last) const {
    return AS.verify(Last);
  }

  static void dumpOffset(const Module *M, ASID AS, int32_t Offset) {
    dumpOffset(M, AS, Offset, dbg);
  }

  static void dumpOffset(const Module *M,
                         ASID AS,
                         int32_t Offset,
                         std::ostream &Output) {
    if (M != nullptr && AS == ASID::cpuID()) {
      int32_t I = 0;
      for (const GlobalVariable &GV : M->globals()) {
        if (Offset == I) {
          Output << GV.getName().str();
          return;
        }
        I++;
      }
    }

    if (Offset < 0) {
      Offset = -Offset;
      Output << "-";
    }
    Output << "0x" << std::hex << Offset;
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    AS.dump(Output);
    if (Offset >= 0)
      Output << "+";
    dumpOffset(M, AS, Offset, Output);
  }

private:
  ASID AS;
  int32_t Offset;
};

/// \brief Class representing the name associated to a TaggedValue
///
/// A name represents the "initial value" a certain address space slot had at
/// the entry of the function. The name is therefore associated with an ASSlot
/// but also with an index, similar to the index of the stack address space,
/// which indicates if it's the "initial value" met in this function, in the
/// caller function, in the caller's caller function and so on.
class Tag {
public:
  Tag(ASSlot Name) : Index(0), Name(Name) { }

  static Tag invalid() {
    return Tag(ASSlot::invalid());
  }

  bool operator==(const Tag &Other) const {
    return Name == Other.Name && Index == Other.Index;
  }

  bool operator!=(const Tag &Other) const {
    return !(*this == Other);
  }

  bool lowerThanOrEqual(const Tag &Other, ASID Last) const {
    return Name.lowerThanOrEqual(Other.Name, Last) && Index == Other.Index;
  }

  bool greaterThan(const Tag &Other, ASID Last) const {
    return !(this->lowerThanOrEqual(Other, Last));
  }

  bool isCalleeInit() const {
    assert(Index >= 0);
    return Index == 0;
  }

  void moveUp() {
    Index++;
  }

  bool isInvalid() const { return Name.isInvalid(); }

  const ASSlot name() const { return Name; }

  bool verify(ASID Last) const {
    if (Index < 0)
      return false;

    if (!Name.isInvalid())
      return Name.verify(Last);

    return true;
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    Output << "[";
    if (!Name.isInvalid()) {
      Output << "init" << Index << "(";
      Name.dump(M, Output);
      Output << ")";
    }
    Output << "]";
  }

  void shiftAddressSpaces(int N) {
    if (!Name.isInvalid())
      Name.shiftAddressSpaces(N);
    Index += N;

    assert(Index >= 0);
  }

  void capAddressSpaces(uint32_t Last) {
    if (!Name.isInvalid())
      Name.capAddressSpaces(Last);
  }

private:
  int32_t Index;
  ASSlot Name;
};

/// \brief A TaggedValue is either an ASSlot or an ASSet associated to a name
class TaggedValue {
private:
  TaggedValue(Tag TheTag) : TheTag(TheTag), Content(ASSlot::invalid()) { }

public:
  TaggedValue(ASSlot Slot) : TheTag(Tag::invalid()), Content(Slot) { }
  TaggedValue(Tag TheTag, ASSet ASF) : TheTag(TheTag), Content(ASF) { }

public:
  static TaggedValue invalid() {
    return TaggedValue(Tag::invalid());
  }

  bool hasTag() const { return !TheTag.isInvalid(); }

  bool isInvalid() const {
    // Invalid means we have no name and the Content is an invalid ASO
    if (hasTag())
      return false;

    if (const auto *ASO = boost::get<ASSlot>(&Content))
      return ASO->isInvalid();

    return false;
  }

  bool isASSlot() const {
    return boost::get<ASSlot>(&Content) != nullptr;
  }

  ASSlot *getASSlot() {
    return boost::get<ASSlot>(&Content);
  }

  ASSet *getASFunction() {
    return boost::get<ASSet>(&Content);
  }

  const ASSlot *getASSlot() const {
    return boost::get<ASSlot>(&Content);
  }

  const ASSet *getASFunction() const {
    return boost::get<ASSet>(&Content);
  }

  bool lowerThanOrEqual(const ASSet &Other, ASID Last) const {
    if (const ASSlot *Slot = getASSlot())
      return Slot->lowerThanOrEqual(Other, Last);
    else
      return getASFunction()->lowerThanOrEqual(Other, Last);
  }

  bool lowerThanOrEqual(const TaggedValue &Other, ASID Last) const {
    // Losing the name is fine, acquiring it is not
    if (!hasTag() && Other.hasTag())
      return false;

    if (hasTag() && Other.hasTag())
      if (TheTag.greaterThan(Other.TheTag, Last))
        return false;

    // Similar argument for having an ASO
    bool ThisIsASO = isASSlot();
    bool OtherIsASO = Other.isASSlot();
    if (!ThisIsASO && OtherIsASO) {
      return false;
    } else if (ThisIsASO && OtherIsASO) {
      return getASSlot()->lowerThanOrEqual(*Other.getASSlot(), Last);
    } else if (ThisIsASO && !OtherIsASO) {
      ASID AS = getASSlot()->addressSpace();
      return Other.getASFunction()->contains(AS, Last);
    } else {
      assert(!ThisIsASO && !OtherIsASO);
      return getASFunction()->lowerThanOrEqual(*Other.getASFunction(), Last);
    }
  }

  bool greaterThan(const TaggedValue &Other, ASID Last) const {
    return !this->lowerThanOrEqual(Other, Last);
  }

  bool operator==(const TaggedValue &Other) const {
    return std::tie(TheTag, Content) == std::tie(Other.TheTag, Other.Content);
  }

  bool operator!=(const TaggedValue &Other) const {
    return !(*this == Other);
  }

  ASSet flatten() const {
    if (const ASSlot *ASO = getASSlot()) {
      return ASO->flatten();
    } else if (const ASSet *ASF = getASFunction()) {
      return *ASF;
    } else {
      assert(false);
    }
  }

  bool add(int32_t Addend) {
    if (ASSlot *ASO = getASSlot()) {
      ASO->add(Addend);
      return true;
    }

    return false;
  }

  bool mask(uint64_t Operand) {
    if (ASSlot *ASO = getASSlot()) {
      ASO->mask(Operand);
      return true;
    }

    return false;
  }

  void shiftAddressSpaces(int N) {
    if (!TheTag.isInvalid())
      TheTag.shiftAddressSpaces(N);

    if (ASSlot *ASO = getASSlot()) {
      ASO->shiftAddressSpaces(N);
    } else if (ASSet *ASF = getASFunction()) {
      ASF->shiftAddressSpaces(N);
    } else {
      assert(false);
    }
  }

  void capAddressSpaces(uint32_t Last) {
    TheTag.capAddressSpaces(Last);

    if (ASSlot *ASO = getASSlot()) {
      ASO->capAddressSpaces(Last);
    } else if (ASSet *ASF = getASFunction()) {
      ASF->capAddressSpaces(Last);
    } else {
      assert(false);
    }
  }

  bool verify(ASID Last) const {
    if (!TheTag.isInvalid())
      if (!TheTag.verify(Last))
        return false;

    if (const ASSlot *ASO = getASSlot()) {
      return ASO->verify(Last);
    } else if (const ASSet *ASF = getASFunction()) {
      return ASF->verify(Last);
    } else {
      assert(false);
    }
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    TheTag.dump(M, Output);
    Output << ": ";

    if (const ASSlot *ASO = getASSlot()) {
      ASO->dump(M, Output);
    } else if (const ASSet *ASF = getASFunction()) {
      ASF->dump(Output);
    } else {
      assert(false);
    }
  }

  Tag *tag() {
    if (hasTag())
      return &TheTag;
    else
      return nullptr;
  }

  const Tag *tag() const {
    if (hasTag())
      return &TheTag;
    else
      return nullptr;
  }

  void setTag(Tag TheTag) {
    this->TheTag = TheTag;
  }

private:
  Tag TheTag;
  boost::variant<ASSlot, ASSet> Content;
};

/// \brief A Value represents the value associated by the analysis to an SSA
///        value/slot
class Value {
public:
  Value() : IndirectContent(), DirectContent(TaggedValue::invalid()) { }

  Value(ASSlot Slot) : IndirectContent(), DirectContent(Slot) { }

  Value(ASSet ASF) :
    IndirectContent(ASF),
    DirectContent(TaggedValue::invalid()) { }

  Value(ASSlot Tag, ASSet IndirectContent) :
    IndirectContent(IndirectContent),
    DirectContent(Tag, ASSet()) { }

  bool hasDirectContent() const { return !DirectContent.isInvalid(); }

  bool operator==(const Value &Other) const {
    return IndirectContent == Other.IndirectContent
      && DirectContent == Other.DirectContent;
  }

  bool operator!=(const Value &Other) const {
    return !(*this == Other);
  }

  bool lowerThanOrEqual(const ASSet &Other, ASID Last) const {
    if (IndirectContent.greaterThan(Other, Last))
      return false;

    if (hasDirectContent()) {
      return DirectContent.lowerThanOrEqual(Other, Last);
    }

    return true;
  }

  bool lowerThanOrEqual(const Value &Other, ASID Last) const {
    // Prerequisite: my indirect content should be smaller than or equal Other's
    if (IndirectContent.greaterThan(Other.IndirectContent, Last))
      return false;

    if (hasDirectContent() && Other.hasDirectContent()) {
      // Force equality
      // TODO: is this correct? shouldn't we assert DirectContent ==
      //       Other.DirectContent?
      return DirectContent.lowerThanOrEqual(Other.DirectContent, Last);
    } else if (hasDirectContent()) {
      // Other has no direct content but we do: check that Other's indirect
      // content contains the AS associated with us
      return DirectContent.flatten().lowerThanOrEqual(Other.IndirectContent,
                                                      Last);
    } else if (Other.hasDirectContent()) {
      // Other has direct content and we don't, it's more specific than us
      return false;
    }

    return true;
  }

  bool greaterThan(const Value &Other, ASID Last) const {
    return !this->lowerThanOrEqual(Other, Last);
  }

  bool isBottom() const {
    return !hasDirectContent() && IndirectContent.empty();
  }

  Value &combine(const Value &Other, ASID Last=InvalidAS) {
    // First merge the ASFs
    IndirectContent.combine(Other.IndirectContent, Last);

    // If the DirectContent is identical we're done, otherwise drop it and
    // flatten out everything into the ASF
    if (DirectContent != Other.DirectContent) {
      IndirectContent.combine(DirectContent.flatten(), Last);
      IndirectContent.combine(Other.DirectContent.flatten(), Last);

      // If the two direct conts are different but have the same name, preserve
      // the name and set the direct content to an ASSet with the same value as
      // the indirect content
      const Tag *ThisTag = DirectContent.tag();
      const Tag *OtherTag = Other.DirectContent.tag();
      if (ThisTag != nullptr
          && OtherTag != nullptr
          && *DirectContent.tag() == *Other.DirectContent.tag()) {
        // TODO: should we use only the combine of the two direct contents
        //       instead of taking into account the original indirect contents
        //       too?
        DirectContent = TaggedValue(*DirectContent.tag(), IndirectContent);
      } else {
        DirectContent = TaggedValue::invalid();
      }
    }

    return *this;
  }

  void store(const ASSet StoredASs) {
    assert(!StoredASs.empty());

    if (IndirectContent.empty()) {
      // We're "infecting" this Value with an indirect store for the first time,
      // so first clone into IndirectContent the DirectContent flattened
      IndirectContent = DirectContent.flatten();
    }

    IndirectContent.combine(StoredASs);
  }

  const ASSlot *getASSlot() const {
    if (!IndirectContent.empty())
      return nullptr;

    return DirectContent.getASSlot();
  }

  TaggedValue *directContent() {
    if (DirectContent.isInvalid())
      return nullptr;
    else
      return &DirectContent;
  }

  const TaggedValue *directContent() const {
    if (DirectContent.isInvalid())
      return nullptr;
    else
      return &DirectContent;
  }

  const ASSet &indirectContent() const { return IndirectContent; }

  // TODO: Handle size of the offset
  bool add(int32_t Addend) {
    if (hasDirectContent())
      return DirectContent.add(Addend);
    return false;
  }

  bool mask(uint64_t Operand) {
    if (hasDirectContent())
      return DirectContent.mask(Operand);
    return false;
  }

  ASSet flatten() const {
    ASSet Result;
    Result.combine(IndirectContent);
    Result.combine(DirectContent.flatten());
    return Result;
  }

  void shiftAddressSpaces(int N) {
    IndirectContent.shiftAddressSpaces(N);
    if (!DirectContent.isInvalid())
      DirectContent.shiftAddressSpaces(N);
  }

  void freeze(ASSlot T) {
    if (TaggedValue *Direct = directContent()) {
      if (!Direct->hasTag())
        Direct->setTag(Tag(T));
    } else {
      DirectContent = TaggedValue(T, flatten());
    }
  }

  bool verify(ASID Last) const {
    if (!IndirectContent.verify(Last))
      return false;

    if (!DirectContent.isInvalid())
      return DirectContent.verify(Last);

    return true;
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    IndirectContent.dump(Output);
    if (!DirectContent.isInvalid()) {
      Output << " || ";
      DirectContent.dump(M, Output);
    }
  }

  void capAddressSpaces(uint32_t Last) {
    IndirectContent.capAddressSpaces(Last);
    if (!DirectContent.isInvalid())
      DirectContent.capAddressSpaces(Last);
  }

private:
  ASSet IndirectContent;
  TaggedValue DirectContent;
};

/// \brief Class representing the content of an address space
///
/// An address space is composed by:
///
/// * AddressSpaceWideContent: an ASSet representing the content that might be
///   found in all the the slots except those explicitly handled in
///   ASOContent. This is also the default value that a load from a slot will
///   initially assume.
/// * ASOContent: a set of <Offset, Value> pairs recording what are the possible
///   values of the slot at the given offset.
class AddressSpace {
public:
  using Container = std::map<int32_t, Value>;

public:
  AddressSpace(ASID ID) : ID(ID) { }

  AddressSpace(ASID ID, ASSet AddressSpaceWideContent) :
    ID(ID), AddressSpaceWideContent(AddressSpaceWideContent) { }

  bool operator==(const AddressSpace &Other) const {
    if (AddressSpaceWideContent != Other.AddressSpaceWideContent)
      return false;

    if (ASOContent != Other.ASOContent)
      return false;

    return true;
  }

  bool operator!=(const AddressSpace &Other) const {
    return !(*this == Other);
  }

  bool lowerThanOrEqual(const AddressSpace &Other, ASID Last) const {
    if (AddressSpaceWideContent.greaterThan(Other.AddressSpaceWideContent,
                                            Last)) {
      return false;
    }

    for (auto &P : ASOContent) {
      auto It = Other.ASOContent.find(P.first);
      // Check Other has at least all our ASOs
      if (It == Other.ASOContent.end())
        return P.second.lowerThanOrEqual(Other.AddressSpaceWideContent, Last);

      // Check the actual value
      if (P.second.greaterThan(It->second, Last))
        return false;
    }

    for (auto &P : Other.ASOContent) {
      auto It = ASOContent.find(P.first);
      if (It == ASOContent.end()) {
        const Value &OtherV = P.second;
        if (OtherV.hasDirectContent())
          return false;

        if (AddressSpaceWideContent.greaterThan(OtherV.indirectContent(), Last))
          return false;
      }
    }

    return true;
  }

  bool greaterThan(const AddressSpace &Other, ASID Last) const {
    return !this->lowerThanOrEqual(Other, Last);
  }

  bool contains(int32_t Offset) const {
    return ASOContent.count(Offset) != 0;
  }

  void set(int32_t Offset, Value V) { ASOContent[Offset] = V; }

  void discard(int32_t Offset) { ASOContent.erase(Offset); }

  void capAddressSpaces(uint32_t Last) {
    AddressSpaceWideContent.capAddressSpaces(Last);
    for (auto &P : ASOContent)
      P.second.capAddressSpaces(Last);
  }

  void freeze() {
    assert(ID.isStack());
    for (auto &P : ASOContent)
      P.second.freeze(ASSlot(ID, P.first));
  }

  ASID id() const { return ID; }

  Container::const_iterator begin() const { return ASOContent.begin(); }
  Container::const_iterator end() const { return ASOContent.end(); }

  /// \brief Contaminate this address space with the set of address spaces in
  ///        \p StoredAS
  void store(const ASSet StoredASs) {
    AddressSpaceWideContent.combine(StoredASs);

    // Merge each slot with the stored stuff
    for (auto It = ASOContent.begin(); It != ASOContent.end();) {
      It->second.store(StoredASs);

      // Cleanup phase
      if (!It->second.hasDirectContent()
          && It->second.indirectContent() == AddressSpaceWideContent) {
        It = eraseASO(It);
      } else {
        It++;
      }

    }
  }

  /// \brief Handle loading from a specific slot
  Value load(ASSlot Address) const {
    assert(Address.addressSpace() == ID);

    // If we can load from it, return the result right away, otherwise perform a
    // load from the whole address space
    if (const Value *LoadedASSlot = get(Address.offset())) {
      Value Result = *LoadedASSlot;
      if (!LoadedASSlot->hasDirectContent())
        Result.freeze(Address);

      return Result;
    } else {
      // We're loading from a specific location in TargetAS, but we have no
      // recorded information about that location. We do not use
      // AddressSpace::flatten() since know that we're loading from the
      // TargetAS, but not from any of the ASSlot registered, so exclude them
      return Value(Address, AddressSpaceWideContent);
    }
  }

  const ASSet &addressSpaceWideContent() const {
    return AddressSpaceWideContent;
  }

  ASSet flatten() const {
    ASSet Result = AddressSpaceWideContent;

    // If it's a stack, in the flattening ignore all the slots with a positive
    // offset
    // TODO: do we want to keep this?
    if (ID.isStack()) {
      for (auto &P : ASOContent)
        if (P.first > 0)
          Result.combine(P.second.flatten());
    } else {
      for (auto &P : ASOContent)
        Result.combine(P.second.flatten());
    }

    return Result;
  }

  void shiftAddressSpaces(int N) {
    ID = ID.shiftAddressSpaces(N);
    assert(!ID.isVirtual());
    AddressSpaceWideContent.shiftAddressSpaces(N);

    for (auto &P : ASOContent)
      P.second.shiftAddressSpaces(N);
  }

  /// \brief Return the number of slots available in this state
  size_t size() const { return ASOContent.size(); }

  bool verify(ASID StateID, ASID Last) const {
    if (StateID != ID)
      return false;

    if (!AddressSpaceWideContent.verify(Last))
      return false;

    for (auto &P : ASOContent)
      if (!P.second.verify(Last))
        return false;

    return true;
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    ID.dump(Output);
    Output << ": ASWC ";
    AddressSpaceWideContent.dump(Output);

    for (auto &P : ASOContent) {
      Output << "\n    ";
      ASSlot::dumpOffset(M, ID, P.first);
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

  using ASOContentIt = Container::iterator;
  ASOContentIt eraseASO(ASOContentIt It) {
    assert(!It->second.hasDirectContent());
    return ASOContent.erase(It);
  }

private:
  friend class Element;

  ASID ID;
  ASSet AddressSpaceWideContent;
  Container ASOContent;
};

/// \brief Represents an element of the lattice of the stack analysis
///
/// This class basically keeps the state of all the address spaces being
/// considered in the current analysis.
class Element {
public:
  // TODO: use SmallVector instead of std::vector for AddressSpace
  using Container = std::vector<AddressSpace>;

public:
  Element() : Element(ASID::lastStackID()) { }

  Element(ASID Last) {
    unsigned Count = Last.id() + 1;
    State.reserve(Count);
    for (unsigned I = 0; I < Count; I++)
      State.emplace_back(ASID(I));
  }

  bool operator==(const Element &Other) const {
    return State == Other.State;
  }

  bool operator!=(const Element &Other) const {
    return !(*this == Other);
  }

  bool lowerThanOrEqual(const Element &Other) const {
    size_t CommonASCount = std::min(State.size(), Other.State.size());
    size_t LastStackID = CommonASCount - 1;
    size_t TotalASCount = std::max(State.size(), Other.State.size());
    uint32_t Rst = ASID::RestOfTheStackID;
    for (unsigned I = 0; I < TotalASCount; I++) {
      unsigned ThisID = I < State.size() ? I : Rst;
      unsigned OtherID = I < Other.State.size() ? I : Rst;
      if (State.at(ThisID).greaterThan(Other.State.at(OtherID), LastStackID))
        return false;
    }

    return true;
  }

  bool greaterThan(const Element &RHS) const {
    return !this->lowerThanOrEqual(RHS);
  }

  bool equal(const Element &RHS) const {
    return this->lowerThanOrEqual(RHS) && RHS.lowerThanOrEqual(*this);
  }

  /// \brief Return the number of handled address spaces
  size_t size() const { return State.size(); }

  ASID last() const {
    if (State.size() >= ASID::lastStackID().id())
      return ASID(State.size() - 1);
    else
      return ASID::restOfTheStackID();
  }

  Element &combine(const Element &Other) {
    // Merge all the common address spaces
    size_t CommonASCount = std::min(State.size(), Other.State.size());
    size_t LastStackID = CommonASCount - 1;
    size_t TotalASCount = std::max(State.size(), Other.State.size());
    uint32_t Rst = ASID::RestOfTheStackID;
    for (unsigned I = 0; I < TotalASCount; I++) {
      unsigned ThisID = I < State.size() ? I : Rst;
      unsigned OtherID = I < Other.State.size() ? I : Rst;
      mergeASState(State.at(ThisID), Other.State.at(OtherID), LastStackID);
    }

    // Drop all the extra address spaces
    if (State.size() >= Other.State.size())
      State.erase(State.begin() + CommonASCount, State.end());

    assert(State.size() == CommonASCount);

    return *this;
  }

  /// \brief Perform pruning over this lattice element employing the given usage
  ///        log information
  ///
  /// This function sends to top certain parts of the lattice element that have
  /// never been touched. For instance, if this is a context of a function that
  /// just reads a single register, we can safely drop (i.e., send to top) all
  /// the other information stored in the lattice element. This has the benefit
  /// of making a certain entry of the results cache more applicable to
  /// different context, saving useless recomputations.
  void prune(const LoadStoreLog *Log, const ASIndexer *Indexer) {
    pruneImpl(Log, Indexer);

    // Assert that pruning twice doesn't change anything
    Element Copy = *this;
    Copy.pruneImpl(Log, Indexer);
    assert(Copy == *this);
  }

  void copyStackArguments(ASID SourceID, ASID DestinationID, int32_t Size) {
    assert(Size >= 0);

    AddressSpace &Destination = State.at(DestinationID.id());
    for (auto &P : State.at(SourceID.id()).ASOContent) {
      int32_t Offset = P.first;
      if (Offset <= 0) {
        Offset = -Offset;

        // Copy only the arguments below the limit of the stack
        if (Offset <= Size)
          Destination.ASOContent[Size - Offset] = P.second;

      }
    }
  }

  bool contains(const ASSlot Search) const {
    // TODO: should we handle the RestOfTheStack case?
    if (Search.addressSpace().id() >= State.size())
      return false;

    return State.at(Search.addressSpace().id()).contains(Search.offset());
  }

  /// \brief Apply to this context the given store log
  // TODO: use reference
  void apply(Element StoreLog) {
    uint32_t CommonASCount = std::min(State.size(), StoreLog.State.size());
    StoreLog.capAddressSpaces(CommonASCount - 1);
    uint32_t Rst = ASID::RestOfTheStackID;

    uint32_t TotalASCount = std::max(State.size(), StoreLog.State.size());
    for (uint32_t I = 0; I < TotalASCount; I++) {
      ASID SourceID(I < StoreLog.State.size() ? I : Rst);
      ASID DestinationID(I < State.size() ? I : Rst);

      const AddressSpace &StoreLogAS = StoreLog.State.at(SourceID.id());

      const ASSet &ASWC = StoreLogAS.AddressSpaceWideContent;
      if (!ASWC.empty())
        store(Value(ASSet::singleElement(DestinationID)), Value(ASWC));

      for (auto &P : StoreLogAS.ASOContent) {
        Value Address(ASSlot(DestinationID, P.first));
        store(Address, P.second);
      }

    }


  }

  /// \brief Look for all the named values in the lattice and replace them with
  ///        the correspoding value in \p Context
  void replaceTaggedValues(const Element &Context) {
    for (AddressSpace &S : State) {
      for (auto &P : S.ASOContent) {
        Value &Content = P.second;
        if (TaggedValue *DirectContent = Content.directContent()) {
          if (Tag *T = DirectContent->tag()) {
            if (T->isCalleeInit()) {
              // Load the initial value in the context

              Value InitialValue = Context.load(T->name());

              // If the loaded value has a name (i.e., it's in the initial value
              // of the context) ensure it's expressed as relative to the caller
              if (TaggedValue *DirectIV = InitialValue.directContent())
                if (Tag *TagIV = DirectIV->tag())
                  TagIV->moveUp();

              // Add the indirect part from the new value
              ASSet IndirectContent = Content.indirectContent();
              if (!IndirectContent.empty())
                InitialValue.store(IndirectContent);

              // Put back at the given offset the result
              P.second = InitialValue;
            }
          }
        }
      }
    }
  }

  ASID getCallerStack(ASID ID) const {
    return ID.getCallerStack(State.size() - 1);
  }

  ASSet flatten() const {
    ASSet Result;
    for (const AddressSpace &S : State)
      Result.combine(S.flatten());
    return Result;
  }

  void capAddressSpaces(uint32_t Last) {
    for (AddressSpace &S : State)
      S.capAddressSpaces(Last);
  }

  void shiftAddressSpaces(int N) {
    assert(N != 0);

    // Remove or add stack address spaces depending on the value of N
    if (N > 0) {
      // Increase all references to stack address spaces
      for (AddressSpace &S : State)
        S.shiftAddressSpaces(N);

      // Inject N new empty address spaces
      AddressSpace Empty(ASID::lastStackID());
      State.insert(State.begin() + ASID::lastStackID().id(),
                   N,
                   Empty);

      // Shift them, i.e., set the right address space ID
      for (int I = 1; I < N; I++) {
        auto It = State.begin() + (ASID::lastStackID().id() + I);
        It->shiftAddressSpaces(I);
      }

    } else {
      auto First = State.begin() + ASID::lastStackID().id();
      State.erase(First, First + -N);

      // Increase all references to stack address spaces
      for (AddressSpace &S : State)
        S.shiftAddressSpaces(N);
    }

  }

  void discard(ASSlot Slot) {
    State.at(Slot.addressSpace().id()).discard(Slot.offset());
  }

  void store(Value Address, Value StoredValue) {
    DBG("sa-verbose", {
        // TODO: get module
        dbg << "Storing ";
        StoredValue.dump(nullptr);
        dbg << " to ";
        Address.dump(nullptr);
        dbg << "\n";
      });

    if (const ASSlot *AddressASO = Address.getASSlot()) {
      ASID TargetASID = AddressASO->addressSpace();

      // Check we're not writing outside our stack frame
      // TODO: report this, in particular if it's not a fake function

      AddressSpace &TargetAS = State.at(TargetASID.id());
      TargetAS.set(AddressASO->offset(), StoredValue);
    } else {
      ASSet TargetASs = Address.flatten();
      ASSet StoredASs = StoredValue.flatten();
      for (ASID TargetID : TargetASs)
        if (!TargetID.isVirtual())
          State.at(TargetID.id()).store(StoredASs);
    }
  }

  Value load(const Value &TargetAddress) const {
    Value Result;

    // Is it an ASSlot?
    if (const ASSlot *ASO = TargetAddress.getASSlot()) {

      return State.at(ASO->addressSpace().id()).load(*ASO);

    } else {

      // It's not an ASSlot, let's iterate over all the involved address spaces
      ASSet Result;
      for (ASID ID : TargetAddress.indirectContent()) {
        if (ID.isVirtual())
          Result.combine(ASSet::singleElement(ID));
        else
          Result.combine(State.at(ID.id()).flatten());
      }

      return Value(Result);
    }
  }

  /// \brief begin iterator for the states handled by this lattice element
  Container::const_iterator begin() const { return State.begin(); }
  Container::const_iterator end() const { return State.end(); }

  std::set<ASSlot> computeCalleeSavedSlots() const {
    std::set<ASSlot> Result;

    unsigned I = 0;
    for (const AddressSpace &ASS : State) {
      for (auto &P : ASS.ASOContent) {
        // Do we have direct content with a name?
        if (const TaggedValue *V = P.second.directContent()) {
          if (const Tag *T = V->tag()) {
            // Is the name the same as the current slot?
            ASSlot Slot(ASID(I), P.first);
            if (*T == Tag(Slot))
              Result.insert(Slot);
          }
        }
      }

      I++;
    }

    return Result;
  }

  bool verify() const {
    unsigned ID = 0;
    for (const AddressSpace &ASS : State)
      if (!ASS.verify(ASID(ID++), last()))
        return false;

    return true;
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    for (const AddressSpace &ASS : State) {
      ASS.dump(M, Output);
      Output << "\n";
    }
  }

private:
  void pruneImpl(const LoadStoreLog *Log, const ASIndexer *Indexer);

  void pruneAddressSpace(const LoadStoreLog *L,
                         const ASIndexer *Indexer,
                         ASID ToPruneID,
                         const LazySmallBitVector &Against,
                         const LazySmallBitVector &LoadButUntracked);

  /// \brief Return an ASSet composed by all the address spaces tracked by this
  ///        lattice element
  ASSet all() const {
    ASSet Result;
    for (unsigned I = 0; I < size(); I++)
      Result.add(ASID(I));
    Result.add(ASID::invalidID());
    Result.add(ASID::deadStackID());
    return Result;
  }

  void mergeASState(AddressSpace &ThisState,
                    const AddressSpace &OtherState,
                    ASID Last) {
    // Handle AddressSpaceWideContent
    ASSet OtherASWC = OtherState.AddressSpaceWideContent;

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
      Value TmpContent;

      if (ThisDone || (!OtherDone && ThisIt->first > OtherIt->first)) {
        // Only Other has the current offset: create a new default entry for
        // delayed appending in this and merge it with OtherContent
        auto ASO = ASSlot(ThisState.id(), OtherIt->first);
        NewEntries.emplace_back(OtherIt->first, ThisState.load(ASO));

        ThisContent = &NewEntries.back().second;
        OtherContent = &OtherIt->second;

        OtherIt++;
      } else if (OtherDone || (!ThisDone && OtherIt->first > ThisIt->first)) {
        // Only this has the current offset: create a default OtherContent and
        // merge with ThisContent
        auto ASO = ASSlot(OtherState.id(), ThisIt->first);
        TmpContent = OtherState.load(ASO);

        ThisContent = &ThisIt->second;
        OtherContent = &TmpContent;

        ThisIt++;
      } else {
        // Both have the current offset: update ThisContent with OtherContent
        assert(ThisIt != ThisEndIt && OtherIt != OtherEndIt);
        assert(ThisIt->first == OtherIt->first);

        ThisContent = &ThisIt->second;
        OtherContent = &OtherIt->second;

        ThisIt++;
        OtherIt++;
      }

      // Perform the merge
      ThisContent->combine(*OtherContent, Last);

      ThisDone = ThisIt == ThisEndIt;
      OtherDone = OtherIt == OtherEndIt;
    }

    for (std::pair<int32_t, Value> P : NewEntries)
      ThisState.ASOContent[P.first] = P.second;

    // At this point we can safely update the ASWC too
    ThisState.AddressSpaceWideContent.combine(OtherASWC, Last);

    // Cleanup phase
    for (auto It = ThisState.ASOContent.begin();
         It != ThisState.ASOContent.end();) {
      const ASSet &ThisASWC = ThisState.addressSpaceWideContent();
      if (!It->second.hasDirectContent()
          && It->second.indirectContent() == ThisASWC) {
        It = ThisState.eraseASO(It);
      } else {
        It++;
      }

    }


  }

private:
  friend class IntraproceduralAnalysis;
  friend void run(BasicBlock *BB);

  // The following vectors are indexed with ASID
  Container State;
};

/// \brief Helper class to manipulate a compact representation of flags for the
///        parts of an element of the lattice
///
/// Given a context, this class assigns to each of its elements and a flag for
/// each registered slot.
class ASIndexer {
public:
  class LoadLog {
  public:
    LoadLog &operator&=(const LoadLog &Other) {
      ASWL &= Other.ASWL;
      Slots &= Other.Slots;
      return *this;
    }

    LoadLog &operator|=(const LoadLog &Other) {
      ASWL |= Other.ASWL;
      Slots |= Other.Slots;
      return *this;
    }

    bool operator==(const LoadLog &Other) const {
      return std::tie(ASWL, Slots) == std::tie(Other.ASWL, Other.Slots);
    }

  private:
    friend ASIndexer;

    LazySmallBitVector ASWL;
    LazySmallBitVector Slots;
  };

public:
  ASIndexer() { }
  ASIndexer(const Element &L) {
    size_t Index = 0;
    size_t ASIndex = 0;
    for (const AddressSpace &AS : L) {
      for (const auto &P : AS)
        Map[ASSlot(ASID(ASIndex), P.first)] = Index++;

      ASIndex++;
    }

  }

  bool contains(ASSlot Slot) const { return Map.count(Slot) != 0; }

  void set(LoadLog &Target, ASSet ASF) const {
    for (const ASID ID : ASF)
      set(Target, ID);
  }

  void set(LoadLog &Target, ASID ID) const {
    if (ID == ASID::deadStackID())
      return;

    Target.ASWL.set(ID.id());
  }

  void set(LoadLog &Target, ASSlot Slot) {
    if (Slot.addressSpace() == ASID::deadStackID())
      return;

    Target.Slots.set(indexForSlot(Slot));
  }

  bool get(const LoadLog &Target, ASID ID) const {
    return Target.ASWL[ID.id()];
  }

  bool get(const LoadLog &Target, ASSlot ID) const {
    Optional<size_t> Index = indexForSlot(ID);
    if (!Index)
      return false;
    else
      return Target.Slots[*Index];
  }

  const LazySmallBitVector &getASWL(const LoadLog &Target) {
    return Target.ASWL;
  }

  std::vector<ASSlot> getSlots(const LoadLog &Target) {
    std::vector<ASSlot> Result;

    for (auto &P : Map) {
      unsigned Index = P.second;
      if (Target.Slots[Index])
        Result.push_back(P.first);
    }

    return Result;
  }

  ASIndexer shiftAddressSpaces(int32_t N, LoadLog &Target) {
    assert(N == -1);

    ASIndexer Result;

    // Shift all the elements of the map
    for (auto &P : Map) {
      ASSlot Slot = P.first;
      Slot.shiftAddressSpaces(N);
      Result.Map[Slot] = P.second;
    }

    // Shift the ASWL part
    ASSet::shiftAddressSpaces(N, Target.ASWL);

    return Result;
  }

  void dump(const LoadLog &Target) const {
    dump(Target, dbg);
  }

  void dump(const LoadLog &Target, std::ostream &Output) const {
    unsigned Count = Target.ASWL.requiredBits();
    for (auto &P : Map)
      Count = std::max(Count, P.first.addressSpace().id() + 1);

    for (unsigned I = 0; I < Count; I++) {
      ASID ID(I);
      ID.dump(Output);
      Output << " ASWL " << Target.ASWL[ID.id()];
      Output << "\n";
      for (auto &P : Map) {
        if (P.first.addressSpace() == ID) {
          Output << "  ";
          P.first.dump(nullptr, Output);
          Output << ": ";
          Output << Target.Slots[P.second];
          Output << "\n";
        }
      }
    }
  }

  LazySmallBitVector getLoadButUntracked(const LoadLog &Target,
                                         const Element &L) const {
    LazySmallBitVector Result;
    std::vector<const ASSlot *> Slots;
    for (auto &P : Map) {
      unsigned Index = P.second;
      if (Index >= Slots.size())
        Slots.resize(Index + 1);
      Slots[Index] = &P.first;
    }

    for (unsigned Index : Target.Slots)
      if (!L.contains(*Slots[Index]))
        Result.set(Slots[Index]->addressSpace().id());

    return Result;
  }

private:
  Optional<size_t> indexForSlot(ASSlot Slot) const {
    auto It = Map.find(Slot);
    if (It == Map.end())
      return Optional<size_t>();
    else
      return It->second;
  }

  size_t indexForSlot(ASSlot Slot) {
    auto It = Map.find(Slot);
    if (It != Map.end()) {
      return It->second;
    } else {
      return Map[Slot] = Map.size();
    }
  }

private:
  friend class Element;

  std::map<ASSlot, size_t> Map;
};

/// \brief Log of all the operations performed by a function in terms of
///        load/stores
///
/// This class is used to keep track of all the loads (stored in a bit vector)
/// and stores (stored in a Element) performed by a function.
///
/// \note That the store log is relative to the callee, while the load log is
///       relative to the caller (or, more precisely, to the call context).
struct LoadStoreLog {
  LoadStoreLog() { }
  LoadStoreLog(Element Store) : Store(Store), Load() { }

  Element Store;
  ASIndexer::LoadLog Load;

  /// \brief Shift the information stored in this log to the context of the
  ///        caller
  ASIndexer moveToCaller(Element &Context, ASIndexer &OldIndexer) {
    Store.replaceTaggedValues(Context);

    Store.shiftAddressSpaces(-1);

    return OldIndexer.shiftAddressSpaces(-1, Load);
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    Output << "Store log:\n";
    Store.dump(M, Output);
    Output << "\n";
  }

};

void Element::pruneAddressSpace(const LoadStoreLog *L,
                                const ASIndexer *Indexer,
                                ASID ToPruneID,
                                const LazySmallBitVector &Against,
                                const LazySmallBitVector &LoadButUntracked) {
  const LoadStoreLog &Log = *L;
  AddressSpace &S = State[ToPruneID.id()];
  unsigned Last = State.size() - 1;

  // Is there at least an indirect load from this address space?
  bool ASWL = llvm::any_of(Against, [&] (unsigned ID) {
      return Indexer->get(Log.Load, ASID(ID));
    });

  // Is there at least a load from a slot which was not present in this?
  bool LFUS = llvm::any_of(Against, [&] (unsigned ID) {
      return LoadButUntracked[ID];
    });

  // Three situations:
  //
  // * !ASWL and !LFUS (Case1): we can set the address space-wide content to top
  //   and discard all the slots which have not been explicitly read.
  // * if !ASWL but LFUS (Case2): we can't touch the address space-wide content,
  //   but we can discard all the slots which have not been explicitly read.
  // * otherwise (Case3): we can't touch the address space-wide content. We send
  //   to top (i.e., flatten + ASWC) all the slots that have not been read
  //   explicitly.

  bool Case1 = !ASWL && !LFUS;
  bool Case2 = !ASWL && LFUS;
  bool Case3 = !Case1 && !Case2;

  if (Case1) {
    DBG("sa", {
        dbg << "Enlarging ASWC of ";
        ToPruneID.dump();
        dbg << " from ";
        S.AddressSpaceWideContent.dump();
        dbg << " to ";
        all().dump();
        dbg << "\n";
      });
    S.AddressSpaceWideContent = all();
  }

  // TODO: this could be done more efficiently
  // Note: we're relying on std::set::erase not invalidating iterators
  for (auto It = S.ASOContent.begin(); It != S.ASOContent.end(); /**/) {
    auto &P = *It;

    bool Read = llvm::any_of(Against, [&] (unsigned ID) {
        return Indexer->get(Log.Load, ASSlot(ASID(ID), P.first));
      });;

    if (!Read) {
      if (Case1 || Case2) {
        DBG("sa", {
            dbg << "Pruning ";
            ASSlot(ToPruneID, P.first).dump(nullptr);
            dbg << "\n";
          });
        It = S.ASOContent.erase(It);
        continue;
      } else {
        assert(Case3);
        ASSet NewValue = P.second.flatten();
        NewValue.combine(S.AddressSpaceWideContent, Last);
        DBG("sa", {
            if (P.second != NewValue) {
              dbg << "Enlarging ";
              ASSlot(ToPruneID, P.first).dump(nullptr);
              dbg << " from ";
              P.second.dump(nullptr);
              dbg << " to ";
              NewValue.dump();
              dbg << "\n";
            }
          });
        P.second = Value(NewValue);
      }
    }

    It++;
  }

}

void Element::pruneImpl(const LoadStoreLog *L, const ASIndexer *Indexer) {
  const LoadStoreLog &Log = *L;
  LazySmallBitVector LoadButUntracked = Indexer->getLoadButUntracked(Log.Load,
                                                                     *this);

  // Three cases:
  //
  // 1. this and the log handle the same number of address spaces
  // 2. this has more stacks than the log
  // 3. the log has more stacks than this

  // The Log.Store always an address space more than the context
  // TODO: are we 100% sure?
  unsigned LogSize = Log.Store.size() - 1;
  unsigned ThisSize = State.size();
  unsigned CommonSize = 0;
  if (ThisSize == LogSize) {
    CommonSize = ThisSize;
  } else {
    // Merge up to RST, excluded
    assert(ASID::RestOfTheStackID == ASID::LastStackID - 1);
    CommonSize = ASID::RestOfTheStackID;
  }

  for (unsigned I = 0; I < CommonSize; I++) {
    ASID ID(I);
    LazySmallBitVector Against;
    Against.set(I);
    pruneAddressSpace(L, Indexer, ID, Against, LoadButUntracked);
  }

  if (ThisSize > LogSize) {
    // Each extra stack (including RST itself)  must be compared against RST
    for (unsigned I = CommonSize; I < ThisSize; I++) {
      ASID ID(I);
      LazySmallBitVector Against;
      Against.set(ASID::RestOfTheStackID);
      pruneAddressSpace(L, Indexer, ID, Against, LoadButUntracked);
    }
  } else if (ThisSize < LogSize) {
    // RST can be pruned only if all the extra stacks (RST included) allow it
    ASID ID(ASID::RestOfTheStackID);
    LazySmallBitVector Against;
    for (unsigned I = CommonSize; I < LogSize; I++)
      Against.set(I);
    pruneAddressSpace(L, Indexer, ID, Against, LoadButUntracked);
  }

}

/// \brief Wrapper around Element to log the actions performed on the underlying
///        lattice element
///
/// ElementProxy provides two features:
///
/// 1. It keeps a version of the Element starting from bottom instead than from
///    the context. It can be employed as a store log which can be applied by
///    the callers instead of directly replacing the current state, which might
///    incur in an information loss.
///
/// 2. It keeps track of which parts of the context have been read, so that we
///    can discard the unused portions of the state. Discarding, means
///    considering them top and make our results more general than the initially
///    provided context. To do so accurately we also need to keep track of which
///    ASSlots have been clobbered by a store, so that we don't consider
///    subsequent read from that address as reads from the original
///    context. This can be accomplished with the Element initialized to bottom,
///    mentioned above.
///
/// The underlying Element is the Actual field, while the two above mentioned
/// logs are embedded in the Log field.
class ElementProxy {
public:
  ElementProxy() { }

  ElementProxy(const Element &InitialState, ASIndexer *Indexer) :
    Actual(InitialState),
    Indexer(Indexer),
    Log(InitialState.last()) { }

  /// \brief Access the underlying Element
  const Element &actual() const { return Actual; }

  /// \brief Access the load/store log
  const LoadStoreLog &log() const { return Log; }

  ASIndexer *indexer() { return Indexer; }

  /// \brief Given the current context, compute the worst case scenario of what
  ///        a function could do
  ///
  /// This function simulates a function that writes wherever it can whatever it
  /// knows about. Basically this functions keeps loading from a set of address
  /// spaces initialized with GLB, CPU and SP0/RST and keeps updating the list
  /// with the newly met address spaces until a fixed point. Then, this address
  /// space set is stored to itself.
  void top() {
    ASSet AccessibleAS;
    AccessibleAS.add(ASID::cpuID());
    AccessibleAS.add(ASID::globalID());
    if (ASID::lastStackID().id() >= Actual.size())
      AccessibleAS.add(ASID::restOfTheStackID());
    else
      AccessibleAS.add(ASID::lastStackID());
    ASSet OldAccessibleAS = AccessibleAS;
    do {
      OldAccessibleAS = AccessibleAS;
      ASSet Loaded = load(Value(AccessibleAS)).indirectContent();
      AccessibleAS.combine(Loaded);
    } while(OldAccessibleAS != AccessibleAS);

    Value Address(AccessibleAS);
    AccessibleAS.drop(ASID::cpuID());
    Value V(AccessibleAS);
    store(Address, V);
  }

  /// \brief Perform a load with logging
  ///
  /// This method mainly wraps the Element::load method. In addition it also
  /// logs the parts of the original context being accessed and transforms
  /// positive stack-relative memory accesses in accesses to the ARG address
  /// space.
  Value load(Value Address) {
    logLoad(Address);
    return Actual.load(Address);
  }

  /// \brief Perform a store
  ///
  /// This method simply performs a store to the underlying Element and to the
  /// store log.
  void store(Value Address, Value StoredValue) {
    Actual.store(Address, StoredValue);
    Log.Store.store(Address, StoredValue);
  }

  ElementProxy &combine(const ElementProxy &Other) {
    Log.Load |= Other.Log.Load;
    Actual.combine(Other.Actual);
    Log.Store.combine(Other.Log.Store);
    return *this;
  }

  void applyStores(const Element &StoreLog) {
    Actual.apply(StoreLog);
    Log.Store.apply(StoreLog);
  }

  void applyLoads(ASIndexer &CalleeIndexer, const ASIndexer::LoadLog LoadLog) {
    for (unsigned I : CalleeIndexer.getASWL(LoadLog)) {
      ASID ID(I);
      if (!ID.isDeadStack())
        Indexer->set(this->Log.Load, ID);
    }

    for (ASSlot Slot : CalleeIndexer.getSlots(LoadLog)) {
      if (!Slot.addressSpace().isDeadStack())
        Indexer->set(this->Log.Load, Slot);
    }
  }

  std::set<ASSlot> computeCalleeSavedSlots() const {
    return Actual.computeCalleeSavedSlots();
  }

  bool lowerThanOrEqual(const ElementProxy &RHS) const {
    auto And = Log.Load;
    And &= RHS.Log.Load;
    return And == RHS.Log.Load && Actual.lowerThanOrEqual(RHS.Actual);
  }

  bool greaterThan(const ElementProxy &RHS) const {
    return !this->lowerThanOrEqual(RHS);
  }

  bool verify() const {
    return Actual.verify();
  }

  void dump(const Module *M) const {
    dump(M, dbg);
  }

  void dump(const Module *M, std::ostream &Output) const {
    Output << "Actual:\n";
    Actual.dump(M, Output);
    Output << "Log:\n";
    Log.dump(M, Output);
  }

private:
  void logLoad(const Value &TargetAddress) {
    // TODO: here we're duplicating some of the logic of Element::load

    // We need a version of the address from the caller's point of view
    Value CallerAddress = TargetAddress;
    // CallerAddress.shiftAddressSpaces(-1);

    if (const ASSlot *CallerSlot = CallerAddress.getASSlot()) {
      const ASSlot *CalleeSlot = TargetAddress.getASSlot();
      assert(CalleeSlot != nullptr);

      // We're reading from an address in the form of a slot
      if (!Log.Store.contains(*CalleeSlot))
        Indexer->set(Log.Load, *CallerSlot);
    } else {
      // We have a load from a set of address spaces, record this fact for each
      // of them
      Indexer->set(Log.Load, CallerAddress.flatten());
    }
  }

private:
  Element Actual;
  ASIndexer *Indexer;
  LoadStoreLog Log;
};

}

#endif // _STACKANALYSIS_IMPL_H
