#ifndef FUNCTIONABI_H
#define FUNCTIONABI_H

// Standard includes
#include <sstream>

// Local includes
#include "ABIDataFlows.h"
#include "ASSlot.h"
#include "BasicBlockInstructionPair.h"
#include "revng/ADT/SmallMap.h"
#include "revng/StackAnalysis/FunctionsSummary.h"
#include "revng/Support/Statistics.h"

extern Logger<> SaABI;

/// \brief Average number of registers tracked by the ABI analysis
extern RunningStatistics ABIRegistersCountStats;

namespace StackAnalysis {

// Forward declarations
namespace ABIAnalysis {
template<typename E>
class Element;
}

class ABIFunction;

struct CombineHelper {

  /// \brief Combine with URAOF
  template<bool FunctionCall>
  static void combine(RegisterArgument<FunctionCall> &This,
                      UsedArgumentsOfFunction::Values V) {
    revng_assert(!FunctionCall);

    if (V == UsedArgumentsOfFunction::Yes) {
      switch (This.Value) {
      case RegisterArgument<FunctionCall>::NoOrDead:
        This.Value = RegisterArgument<FunctionCall>::Contradiction;
        break;
      case RegisterArgument<FunctionCall>::Maybe:
        This.Value = RegisterArgument<FunctionCall>::Yes;
        break;
      case RegisterArgument<FunctionCall>::No:
        // No comes from ECS and wins over everything
        break;
      case RegisterArgument<FunctionCall>::Dead:
      case RegisterArgument<FunctionCall>::Yes:
      case RegisterArgument<FunctionCall>::Contradiction:
        revng_abort();
      }
    }
  }

  /// \brief Combine with DRAOF
  template<bool FunctionCall>
  static void combine(RegisterArgument<FunctionCall> &This,
                      DeadRegisterArgumentsOfFunction::Values V) {
    revng_assert(not FunctionCall);

    if (V == DeadRegisterArgumentsOfFunction::NoOrDead) {
      switch (This.Value) {
      case RegisterArgument<FunctionCall>::Maybe:
        This.Value = RegisterArgument<FunctionCall>::NoOrDead;
        break;
      case RegisterArgument<FunctionCall>::Yes:
        This.Value = RegisterArgument<FunctionCall>::Contradiction;
        break;
      case RegisterArgument<FunctionCall>::No:
        // No comes from ECS and wins over everything
        break;
      case RegisterArgument<FunctionCall>::NoOrDead:
      case RegisterArgument<FunctionCall>::Dead:
      case RegisterArgument<FunctionCall>::Contradiction:
        revng_abort();
      }
    }
  }

  /// \brief Combine with RAOFC
  template<bool FunctionCall>
  static void combine(RegisterArgument<FunctionCall> &This,
                      RegisterArgumentsOfFunctionCall::Values V) {
    revng_assert(FunctionCall);

    if (V == RegisterArgumentsOfFunctionCall::Yes) {
      switch (This.Value) {
      case RegisterArgument<FunctionCall>::NoOrDead:
        This.Value = RegisterArgument<FunctionCall>::Dead;
        break;
      case RegisterArgument<FunctionCall>::Maybe:
        This.Value = RegisterArgument<FunctionCall>::Yes;
        break;
      case RegisterArgument<FunctionCall>::Yes:
      case RegisterArgument<FunctionCall>::Dead:
      case RegisterArgument<FunctionCall>::Contradiction:
        break;
      case RegisterArgument<FunctionCall>::No:
        // No comes from ECS and wins over everything
        break;
      }
    }
  }

  /// \brief Combine with URVOF
  static void
  combine(FunctionReturnValue &This, UsedReturnValuesOfFunction::Values V) {
    if (V == UsedReturnValuesOfFunction::Yes) {
      switch (This.Value) {
      case FunctionReturnValue::Maybe:
        This.Value = FunctionReturnValue::YesCandidate;
        break;
      case FunctionReturnValue::No:
        // No comes from ECS and wins over everything
        break;
      case FunctionReturnValue::Yes:
      case FunctionReturnValue::YesCandidate:
      case FunctionReturnValue::NoOrDead:
      case FunctionReturnValue::Dead:
      case FunctionReturnValue::Contradiction:
        revng_abort();
      }
    }
  }

  /// \brief Combine with DRVOFC
  static void combine(FunctionCallReturnValue &This,
                      DeadReturnValuesOfFunctionCall::Values V) {
    if (V == DeadReturnValuesOfFunctionCall::NoOrDead) {
      switch (This.Value) {
      case FunctionCallReturnValue::Maybe:
        This.Value = FunctionCallReturnValue::NoOrDead;
        break;
      case FunctionCallReturnValue::Yes:
        This.Value = FunctionCallReturnValue::Dead;
        break;
      case FunctionCallReturnValue::No:
        // No comes from ECS and wins over everything
        break;
      case FunctionCallReturnValue::Contradiction:
      case FunctionCallReturnValue::Dead:
      case FunctionCallReturnValue::NoOrDead:
        revng_abort();
      }
    }
  }

  // Combine with URVOFC
  static void combine(FunctionCallReturnValue &This,
                      UsedReturnValuesOfFunctionCall::Values V) {
    if (V == UsedReturnValuesOfFunctionCall::Yes) {
      switch (This.Value) {
      case FunctionCallReturnValue::Dead:
      case FunctionCallReturnValue::NoOrDead:
        This.Value = FunctionCallReturnValue::Contradiction;
        break;
      case FunctionCallReturnValue::Maybe:
      case FunctionCallReturnValue::Yes:
        This.Value = FunctionCallReturnValue::Yes;
        break;
      case FunctionCallReturnValue::No:
        // No comes from ECS and wins over everything
        break;
      case FunctionCallReturnValue::Contradiction:
        revng_abort();
      }
    }
  }
};

/// \brief Map with an updatable default value
///
/// This is a map that can be used to lazily handle K elements: proceed with
/// your processing using the Default member, then when K is met, record the
/// state of Default in the map and proceed.
template<typename K, typename V, size_t N = 40>
class DefaultMap {
public:
  // TODO: the size of the SmallMap needs to be fine tuned
  using Container = SmallMap<K, V, N>;
  using const_iterator = typename Container::const_iterator;
  using iterator = typename Container::iterator;

public:
  V Default;

private:
  Container M;

public:
  DefaultMap() : Default() {}

  DefaultMap(const DefaultMap &) = default;
  DefaultMap &operator=(const DefaultMap &) = default;

  DefaultMap(DefaultMap &&) = default;
  DefaultMap &operator=(DefaultMap &&) = default;

public:
  void clear() {
    Default = V();
    M.clear();
  }

  void clear(V NewDefault) {
    Default = V(NewDefault);
    M.clear();
  }

  void sort() const { M.sort(); }

  size_t size() const { return M.size(); }

  bool contains(K Key) const { return M.count(Key) != 0; }

  void erase(K Key) { M.erase(Key); }

  V &operator[](const K Key) {
    return (*M.insert({ Key, Default }).first).second;
  }

  const V &get(const K Key) const {
    auto It = M.find(Key);
    revng_assert(It != M.end());
    return It->second;
  }

  const V &getOrDefault(const K Key) const {
    auto It = M.find(Key);
    if (It == M.end())
      return Default;
    else
      return It->second;
  }

  const_iterator begin() const { return M.begin(); }
  const_iterator end() const { return M.end(); }

  iterator begin() { return M.begin(); }
  iterator end() { return M.end(); }
};

template<typename V, typename T>
inline void dumpAnalysis(T &Output, const char *Prefix, const V &Analysis) {
  Output << Prefix << V::name() << ": ";
  Analysis.dump(Output);
  Output << "\n";
}

/// \brief State of a register in terms of being an argument or a return value
///        in a certain call site
class CallSiteRegisterState {
private:
  RegisterArgumentsOfFunctionCall RAOFC;
  UsedReturnValuesOfFunctionCall URVOFC;
  DeadReturnValuesOfFunctionCall DRVOFC;

public:
  template<typename T>
  CallSiteRegisterState &assign(const T &Other) {
    T::H::assign(*this, Other.Analyses);
    return *this;
  }

  template<typename T>
  CallSiteRegisterState &combine(const T &Other) {
    T::H::combine(*this, Other.Analyses);
    return *this;
  }

  void resetToUnknown() {
    RAOFC = decltype(RAOFC)::initial();
    RAOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
    URVOFC = decltype(URVOFC)::initial();
    URVOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
    DRVOFC = decltype(DRVOFC)::initial();
    DRVOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
  }

  void applyResults(FunctionCallRegisterArgument &V) const {
    CombineHelper::combine(V, RAOFC.value());
  }

  void applyResults(FunctionCallReturnValue &V) const {
    CombineHelper::combine(V, DRVOFC.value());
    CombineHelper::combine(V, URVOFC.value());
  }

  void dump(const char *Prefix) const debug_function { dump(dbg, Prefix); }

  template<typename T>
  void dump(T &Output, const char *Prefix) const {
    dumpAnalysis(Output, Prefix, RAOFC);
    dumpAnalysis(Output, Prefix, URVOFC);
    dumpAnalysis(Output, Prefix, DRVOFC);
  }

private:
  template<typename A, typename B, bool C, bool D, size_t E>
  friend struct AnalysesWrapperHelpers;

  template<typename T>
  T &getByType();

  template<typename T>
  const T &getByType() const;
};

template<>
inline RegisterArgumentsOfFunctionCall &CallSiteRegisterState::getByType() {
  return RAOFC;
}

template<>
inline UsedReturnValuesOfFunctionCall &CallSiteRegisterState::getByType() {
  return URVOFC;
}

template<>
inline DeadReturnValuesOfFunctionCall &CallSiteRegisterState::getByType() {
  return DRVOFC;
}

template<>
inline const RegisterArgumentsOfFunctionCall &
CallSiteRegisterState::getByType() const {
  return RAOFC;
}

template<>
inline const UsedReturnValuesOfFunctionCall &
CallSiteRegisterState::getByType() const {
  return URVOFC;
}

template<>
inline const DeadReturnValuesOfFunctionCall &
CallSiteRegisterState::getByType() const {
  return DRVOFC;
}

/// \brief State of a register in terms of being an argument or a return value
class RegisterState {
private:
  // Core analyses
  DeadRegisterArgumentsOfFunction DRAOF;
  UsedArgumentsOfFunction URAOF;
  UsedReturnValuesOfFunction URVOF;

  // Function-call related analyses, done only for intrerprocedural reasons
  UsedReturnValuesOfFunctionCall URVOFC;
  DeadReturnValuesOfFunctionCall DRVOFC;
  RegisterArgumentsOfFunctionCall RAOFC;

public:
  void resetToUnknown() {
    DRAOF = decltype(DRAOF)::initial();
    DRAOF.transfer(GeneralTransferFunction::UnknownFunctionCall);
    URAOF = decltype(URAOF)::initial();
    URAOF.transfer(GeneralTransferFunction::UnknownFunctionCall);
    URVOF = decltype(URVOF)::initial();
    URVOF.transfer(GeneralTransferFunction::UnknownFunctionCall);

    URVOFC = decltype(URVOFC)::initial();
    URVOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
    DRVOFC = decltype(DRVOFC)::initial();
    DRVOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
    RAOFC = decltype(RAOFC)::initial();
    RAOFC.transfer(GeneralTransferFunction::UnknownFunctionCall);
  }

  template<typename T>
  RegisterState &assign(const T &Other) {
    T::H::assign(*this, Other.Analyses);
    return *this;
  }

  void applyResults(FunctionRegisterArgument &V) const {
    CombineHelper::combine(V, URAOF.value());
    CombineHelper::combine(V, DRAOF.value());
  }

  void applyResults(FunctionReturnValue &V) const {
    CombineHelper::combine(V, URVOF.value());
  }

  bool isArgument() const {
    return URAOF.value() == UsedArgumentsOfFunction::Yes;
  }

  bool isReturnValue() const {
    return URVOF.value() == UsedReturnValuesOfFunction::Yes;
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    dumpAnalysis(Output, "  ", DRAOF);
    dumpAnalysis(Output, "  ", URAOF);
    dumpAnalysis(Output, "  ", URVOF);
    dumpAnalysis(Output, "  ", URVOFC);
    dumpAnalysis(Output, "  ", DRVOFC);
    dumpAnalysis(Output, "  ", RAOFC);
  }

private:
  template<typename A, typename B, bool C, bool D, size_t E>
  friend struct AnalysesWrapperHelpers;

  template<typename T>
  T &getByType();

  template<typename T>
  const T &getByType() const;
};

template<>
inline DeadRegisterArgumentsOfFunction &RegisterState::getByType() {
  return DRAOF;
}

template<>
inline UsedArgumentsOfFunction &RegisterState::getByType() {
  return URAOF;
}

template<>
inline UsedReturnValuesOfFunction &RegisterState::getByType() {
  return URVOF;
}

template<>
inline UsedReturnValuesOfFunctionCall &RegisterState::getByType() {
  return URVOFC;
}

template<>
inline DeadReturnValuesOfFunctionCall &RegisterState::getByType() {
  return DRVOFC;
}

template<>
inline RegisterArgumentsOfFunctionCall &RegisterState::getByType() {
  return RAOFC;
}

template<>
inline const DeadRegisterArgumentsOfFunction &RegisterState::getByType() const {
  return DRAOF;
}

template<>
inline const UsedArgumentsOfFunction &RegisterState::getByType() const {
  return URAOF;
}

template<>
inline const UsedReturnValuesOfFunction &RegisterState::getByType() const {
  return URVOF;
}

template<>
inline const UsedReturnValuesOfFunctionCall &RegisterState::getByType() const {
  return URVOFC;
}

template<>
inline const DeadReturnValuesOfFunctionCall &RegisterState::getByType() const {
  return DRVOFC;
}

template<>
inline const RegisterArgumentsOfFunctionCall &RegisterState::getByType() const {
  return RAOFC;
}

/// \brief Class to track the ABI, i.e., the status of a register as an
///        argument/return value
class FunctionABI {
  template<typename Enabled>
  friend class ABIAnalysis::Element;

private:
  struct CallsAnalyses {
    DefaultMap<int32_t, CallSiteRegisterState, 20> Registers;
  };

private:
  DefaultMap<int32_t, RegisterState, 20> RegisterAnalyses;
  DefaultMap<FunctionCall, CallsAnalyses, 5> Calls;

public:
  FunctionABI() {}

  /// \brief Explicit copy constructor
  FunctionABI copy() const {
    FunctionABI Result;
    Result.RegisterAnalyses = RegisterAnalyses;
    return Result;
  }

  FunctionABI(const FunctionABI &) = delete;
  FunctionABI &operator=(const FunctionABI &) = delete;

  FunctionABI(FunctionABI &&) = default;
  FunctionABI &operator=(FunctionABI &&) = default;

  ~FunctionABI() { ABIRegistersCountStats.push(RegisterAnalyses.size()); }

public:
  /// \brief Perform the ABI analysis
  void analyze(const ABIFunction &TheFunction);

  template<typename E>
  void combine(const ABIAnalysis::Element<E> &Other) {
    for (auto &P : Other.RegisterAnalyses)
      RegisterAnalyses[P.first].assign(P.second);

    for (auto &P : Other.FunctionCallRegisterAnalyses)
      for (auto &Q : P.second)
        Calls[P.first].Registers[Q.first].assign(Q.second);
  }

  void drop(ASSlot Slot) {
    if (Slot.addressSpace() == ASID::cpuID())
      RegisterAnalyses.erase(Slot.offset());
    else
      revng_abort();
  }

  void resetToUnknown(ASSlot Slot) {
    if (Slot.addressSpace() == ASID::cpuID())
      RegisterAnalyses[Slot.offset()].resetToUnknown();
    else
      revng_abort();
  }

  void applyResults(FunctionRegisterArgument &V, int32_t Offset) const {
    if (RegisterAnalyses.contains(Offset))
      RegisterAnalyses.get(Offset).applyResults(V);
  }

  void applyResults(FunctionCallRegisterArgument &V,
                    FunctionCall Call,
                    int32_t Offset) const {
    if (Calls.contains(Call))
      if (Calls.get(Call).Registers.contains(Offset))
        Calls.get(Call).Registers.get(Offset).applyResults(V);
  }

  void applyResults(FunctionReturnValue &V, int32_t Offset) const {
    if (RegisterAnalyses.contains(Offset))
      RegisterAnalyses.get(Offset).applyResults(V);
  }

  void applyResults(FunctionCallReturnValue &V,
                    FunctionCall Call,
                    int32_t Offset) const {
    if (Calls.contains(Call))
      if (Calls.get(Call).Registers.contains(Offset))
        Calls.get(Call).Registers.get(Offset).applyResults(V);
  }

  /// \brief Collect all the slots involved in this instance
  void collectLocalSlots(std::set<ASSlot> &SlotsPool) const {
    for (auto &P : RegisterAnalyses)
      SlotsPool.insert(ASSlot::create(ASID::cpuID(), P.first));
  }

  std::pair<std::set<int32_t>, std::set<int32_t>> collectYesRegisters() const {
    std::set<int32_t> Arguments;
    std::set<int32_t> ReturnValues;
    for (auto &P : RegisterAnalyses) {
      if (P.second.isArgument())
        Arguments.insert(P.first);
      if (P.second.isReturnValue())
        ReturnValues.insert(P.first);
    }

    return { Arguments, ReturnValues };
  }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    std::stringstream Stream;
    dumpInternal(M, Stream);
    Output << Stream.str();
  }

private:
  void dumpInternal(const llvm::Module *M, std::stringstream &Output) const;
};

} // namespace StackAnalysis

#endif // FUNCTIONABI_H
