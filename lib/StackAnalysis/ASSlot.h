#ifndef ASSLOT_H
#define ASSLOT_H

// Standard includes
#include <limits>

// LLVM includes
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

// Local libraries includes
#include "revng/Support/Debug.h"

extern Logger<> SaDiffLog;

namespace StackAnalysis {

template<typename T>
struct debug_cmp {
  /// \brief Perform a comparison between \p This and \p Other printing out all
  ///        the differences
  ///
  /// We need this so that types not fully under our control (e.g.,
  /// MonotoneFrameworkSet) can implement this method too. This has to be part
  /// of a struct so that we can perform partial template specialization.
  static unsigned cmp(const T &This, const T &Other, const llvm::Module *M) {
    return This.template cmp<true, false>(Other, M);
  }
};

/// \brief Assert LHS.lowerThanOrEqual(RHS), and, if not, print the differences
template<typename T>
inline void
assertLowerThanOrEqual(const T &LHS, const T &RHS, const llvm::Module *M) {
#ifndef NDEBUG
  bool Result = LHS.lowerThanOrEqual(RHS);
  if (!Result) {
    SaDiffLog.enable();
    debug_cmp<T>::cmp(LHS, RHS, M);
    revng_abort();
  }
#endif
}

// Note: the following classes are not part of the Intraprocedural namespace.

/// \brief Identifier of an address space
class ASID {
private:
  uint32_t ID;

private:
  enum {
    /// The address space representing the CPU state, registers in particular
    CPUAddressSpaceID,
    /// The address space containing literal addresses (and numbers)
    GlobalID,
    /// The stack frame we're tracking (SP0)
    LastStackID,
    InvalidID,
    LastID
  };

public:
  explicit ASID(uint32_t ID) : ID(ID) { revng_assert(ID < LastID); }

  // Factory methods
  static ASID invalidID() { return ASID(InvalidID); }
  static ASID cpuID() { return ASID(CPUAddressSpaceID); }
  static ASID stackID() { return ASID(LastStackID); }
  static ASID globalID() { return ASID(GlobalID); }

public:
  uint32_t id() const { return ID; }

  bool operator<(const ASID &Other) const { return ID < Other.ID; }
  bool operator==(const ASID &Other) const { return ID == Other.ID; }
  bool operator!=(const ASID &Other) const { return not(*this == Other); }

  size_t hash() const;

  /// \brief Perform a comparison according to the analysis' lattice
  ///
  /// \note Address spaces identifiers are not comparable, they are just unique
  ///       identifiers. The CPU address space is not "more informative" or
  ///       "less conservative" than the GLB address space. Therefore we just
  ///       perform an equality comparison here.
  bool lowerThanOrEqual(const ASID &Other) const { return ID == Other.ID; }

  bool greaterThan(const ASID &Other) const {
    return not lowerThanOrEqual(Other);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    switch (ID) {
    case CPUAddressSpaceID:
      Output << "CPU";
      break;
    case LastStackID:
      Output << "SP0";
      break;
    case GlobalID:
      Output << "GLB";
      break;
    case InvalidID:
      Output << "INV";
      break;
    }
  }

  bool isStack() const { return ID == LastStackID; }
  bool isValid() const { return ID != InvalidID; }
};

/// \brief Class representing the address of an address space slot
class ASSlot {
private:
  ASID AS;
  int32_t Offset;

private:
  ASSlot(ASID ID, int32_t Offset) : AS(ID), Offset(Offset) {}

public:
  static ASSlot invalid() { return ASSlot(ASID::invalidID(), 0); }
  static ASSlot create(ASID ID, int32_t Offset) {
    revng_assert(ID.isValid());
    return ASSlot(ID, Offset);
  }

public:
  /// \brief Perform a comparison according to the analysis' lattice
  bool lowerThanOrEqual(const ASSlot &Other) const;

  template<bool Diff, bool EarlyExit>
  unsigned cmp(const ASSlot &Other, const llvm::Module *M) const;

  size_t hash() const;

  bool greaterThan(const ASSlot &Other) const {
    return not lowerThanOrEqual(Other);
  }

  bool operator==(const ASSlot &Other) const {
    return std::tie(AS, Offset) == std::tie(Other.AS, Other.Offset);
  }

  bool operator!=(const ASSlot &Other) const { return not(*this == Other); }

  bool operator<(const ASSlot &Other) const {
    auto ThisTuple = std::make_pair(AS.id(), Offset);
    auto OtherTuple = std::make_pair(Other.AS.id(), Other.Offset);
    return ThisTuple < OtherTuple;
  }

  int32_t offset() const { return Offset; }
  ASID addressSpace() const { return AS; }
  bool isInvalid() const { return AS == ASID::invalidID(); }

  /// \brief Add a constant to the offset associated with this slot
  void add(int32_t Addend) { Offset += Addend; }

  /// \brief Mask the offset associated to this slot with a value
  void mask(uint64_t Operand) { Offset = Offset & Operand; }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    AS.dump(Output);
    if (Offset >= 0)
      Output << "+";
    dumpOffset(M, AS, Offset, Output);
  }

public:
  static void dumpOffset(const llvm::Module *M, ASID AS, int32_t Offset) {
    dumpOffset(M, AS, Offset, dbg);
  }

  template<typename T>
  static void
  dumpOffset(const llvm::Module *M, ASID AS, int32_t Offset, T &Output) {
    if (M != nullptr && AS == ASID::cpuID()) {
      auto Name = csvNameByOffset(Offset, M);
      if (Name) {
        Output << *Name;
        return;
      }

      Output << "alloca_";
    }

    if (Offset < 0) {
      Offset = -Offset;
      Output << "-";
    }
    Output << "0x" << std::hex << Offset << std::dec;
  }

private:
  static llvm::Optional<std::string>
  csvNameByOffset(int32_t Offset, const llvm::Module *M) {
    int32_t I = 1;
    for (const llvm::GlobalVariable &GV : M->globals()) {
      if (Offset == I)
        return { GV.getName().str() };
      I++;
    }

    return llvm::Optional<std::string>();
  }
};

} // namespace StackAnalysis

namespace std {

template<>
struct hash<StackAnalysis::ASSlot> {
  size_t operator()(const StackAnalysis::ASSlot &K) const { return K.hash(); }
};

template<>
struct hash<StackAnalysis::ASID> {
  size_t operator()(const StackAnalysis::ASID &K) const { return K.hash(); }
};

} // namespace std

// All of these could probably be reimplemented using lambdas, however I haven't
// assessed the performance of lambdas, and this is quite performance critical,
// therefore I don't want to risk for now. Moreover, lambdas are not super
// elegant either. Coroutines would be the best here.

/// Compute \p Expression, if non-zero:
///
/// * check if Diff == true, if so run \p OnDiff
/// * check if EarlyExit == true, if so return 1, otherwise increment of \p
///   Expression the Result variable
///
/// This is supposed to be used for performing a comparison between objects,
/// possibly printing a diagnostics on why they are different, and allowing the
/// user to choose whether to return on the first call to ROA that evaluates to
/// non-zero or proceed and accumulate the number of differences in the Result
/// variable.
#define ROA(Expression, OnDiff)            \
  do {                                     \
    if (unsigned C = (Expression)) {       \
                                           \
      if (SaDiffLog.isEnabled() && Diff) { \
        OnDiff                             \
      }                                    \
                                           \
      if (EarlyExit) {                     \
        return 1;                          \
      } else {                             \
        Result += C;                       \
      }                                    \
    }                                      \
  } while (false)

#endif // ASSLOT_H
