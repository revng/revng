/// \file functionabi.cpp
/// \brief Implementation of the ABI analysis

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/Support/MonotoneFramework.h"

// Local includes
#include "ABIIR.h"
#include "FunctionABI.h"

using std::conditional;
using std::tuple;
using std::tuple_element;
using std::tuple_size;

using llvm::Module;

Logger<> SaABI("sa-abi");

namespace StackAnalysis {

using ABIIRBB = ABIIRBasicBlock;

static ASID CPU = ASID::cpuID();

template<typename K1, size_t N1, typename K2, typename V2, size_t N2>
using MapOfMaps = DefaultMap<K1, DefaultMap<K2, V2, N2>, N1>;

/// \brief A set of helper functions related to DefaultMap
namespace MapHelpers {

enum Comparison { Lower = -1, Equal = 0, Greater = 1 };

/// \brief Similar to Rust cmp
template<typename T>
static inline Comparison compare(T A, T B) {
  return A == B ? Equal : (A < B ? Lower : Greater);
}

template<typename K, typename V, bool Diff, bool EarlyExit, size_t N>
unsigned
cmp(const DefaultMap<K, V, N> &This, const DefaultMap<K, V, N> &Other) {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  for (auto &P : This) {
    P.second.template cmp<Diff, EarlyExit>(Other.getOrDefault(P.first));
    ROA((P.second.template cmp<Diff, EarlyExit>(Other.getOrDefault(P.first))),
        { revng_log(SaDiffLog, P.first); });
  }

  for (auto &P : Other) {
    ROA((This.getOrDefault(P.first).template cmp<Diff, EarlyExit>(P.second)),
        { revng_log(SaDiffLog, P.first); });
  }

  return Result;
}

template<typename K, typename V, bool Diff, bool EarlyExit, size_t N>
unsigned cmpWithModule(const DefaultMap<K, V, N> &This,
                       const DefaultMap<K, V, N> &Other,
                       ASID ID,
                       const Module *M) {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  for (auto &P : This) {
    ROA((P.second.template cmp<Diff, EarlyExit>(Other.getOrDefault(P.first))), {
      ASSlot::create(ID, P.first).dump(M, SaDiffLog);
      SaDiffLog << DoLog;
    });
  }

  for (auto &P : Other) {
    ROA((This.getOrDefault(P.first).template cmp<Diff, EarlyExit>(P.second)), {
      ASSlot::create(ID, P.first).dump(M, SaDiffLog);
      SaDiffLog << DoLog;
    });
  }

  return Result;
}

template<typename K,
         typename V,
         bool Diff,
         bool EarlyExit,
         size_t N1,
         size_t N2>
unsigned nestedCmpWithModule(const MapOfMaps<FunctionCall, N1, K, V, N2> &This,
                             const MapOfMaps<FunctionCall, N1, K, V, N2> &Other,
                             ASID ID,
                             const Module *M) {
  LoggerIndent<> Y(SaDiffLog);
  unsigned Result = 0;

  for (auto &P : This) {
    ROA((cmpWithModule<K, V, Diff, EarlyExit>(P.second,
                                              Other.getOrDefault(P.first),
                                              ID,
                                              M)),
        {
          P.first.dump(SaDiffLog);
          SaDiffLog << DoLog;
        });
  }

  for (auto &P : Other) {
    ROA((cmpWithModule<K, V, Diff, EarlyExit>(This.getOrDefault(P.first),
                                              P.second,
                                              ID,
                                              M)),
        {
          P.first.dump(SaDiffLog);
          SaDiffLog << DoLog;
        });
  }

  return Result;
}

template<typename V, typename Q>
static void combine(V &This, const Q &Other) {
  This.combine(Other);
}

template<typename K, typename V, typename Q, size_t N>
static void
combine(DefaultMap<K, V, N> &This, const DefaultMap<K, Q, N> &Other) {

  combine(This.Default, Other.Default);

  This.sort();
  Other.sort();
  llvm::SmallVector<const std::pair<const K, Q> *, N> Missing;
  auto ThisIt = This.begin();
  auto ThisEnd = This.end();
  auto OtherIt = Other.begin();
  auto OtherEnd = Other.end();

  // Iterate over the two maps pairwise
  while (OtherIt != OtherEnd && ThisIt != ThisEnd) {
    switch (compare(ThisIt->first, OtherIt->first)) {
    case Greater:
      // Missing, add later (can't change This while iterating)
      Missing.push_back(&*OtherIt);
      OtherIt++;
      break;
    case Equal:
      // Merge
      combine(ThisIt->second, OtherIt->second);
      ThisIt++;
      OtherIt++;
      break;
    case Lower:
      // Only ours, merge with default
      combine(ThisIt->second, Other.Default);
      ThisIt++;
      break;
    }
  }

  // Handle the remaining elements of Other
  while (OtherIt != OtherEnd) {
    combine(This[OtherIt->first], OtherIt->second);
    OtherIt++;
  }

  // Handle the remaining elements of This
  while (ThisIt != ThisEnd) {
    combine(ThisIt->second, Other.Default);
    ThisIt++;
  }

  // Handle the elements we registered
  for (auto *P : Missing)
    combine(This[P->first], P->second);
}

template<typename V, typename T, size_t N>
inline void
dump(const Module *M, T &Output, const DefaultMap<int32_t, V, N> &D, ASID ID) {
  for (auto &P : D) {
    ASSlot::create(ID, P.first).dump(M, Output);
    Output << ":\n";
    P.second.dump(Output);
    Output << "\n";
  }
}

template<typename V, typename T, size_t N>
inline void dump(const Module *M,
                 T &Output,
                 const DefaultMap<int32_t, V, N> &D,
                 ASID ID,
                 const char *Prefix) {
  std::string Longer(Prefix);
  Longer += "  ";

  Output << Prefix << "Default:\n";
  D.Default.dump(Output, Longer.data());
  Output << "\n";

  for (auto &P : D) {
    Output << Prefix;
    ASSlot::create(ID, P.first).dump(M, Output);
    Output << ":\n";
    P.second.dump(Output, Longer.data());
    Output << "\n";
  }
}

template<typename V, typename T, size_t N1, size_t N2>
inline void dump(const Module *M,
                 T &Output,
                 const MapOfMaps<FunctionCall, N1, int32_t, V, N2> &D,
                 ASID ID,
                 const char *Prefix) {
  std::string Longer(Prefix);
  Longer += "  ";

  Output << Prefix << "Default:\n";
  dump(M, Output, D.Default, ID, Longer.data());
  Output << "\n";

  for (auto &P : D) {
    Output << Prefix;
    P.first.dump(Output);
    Output << ":\n";
    dump(M, Output, P.second, ID, Longer.data());
    Output << "\n";
  }
}

template<typename V, typename Q>
static void returnFromCall(V &This, const Q &Other) {
  This.returnFromCall(Other);
}

template<typename K, typename V, typename Q, size_t N>
static void
returnFromCall(DefaultMap<K, V, N> &This, const DefaultMap<K, Q, N> &Other) {

  returnFromCall(This.Default, Other.Default);

  This.sort();
  Other.sort();
  llvm::SmallVector<const std::pair<const K, Q> *, N> Missing;
  auto ThisIt = This.begin();
  auto ThisEnd = This.end();
  auto OtherIt = Other.begin();
  auto OtherEnd = Other.end();

  // Iterate over the two maps pairwise
  while (OtherIt != OtherEnd && ThisIt != ThisEnd) {
    switch (compare(ThisIt->first, OtherIt->first)) {
    case Greater:
      // Missing, add later (can't change This while iterating)
      Missing.push_back(&*OtherIt);
      OtherIt++;
      break;
    case Equal:
      // Merge
      returnFromCall(ThisIt->second, OtherIt->second);
      ThisIt++;
      OtherIt++;
      break;
    case Lower:
      // Only ours, merge with default
      returnFromCall(ThisIt->second, Other.Default);
      ThisIt++;
      break;
    }
  }

  // Handle the remaining elements of Other
  while (OtherIt != OtherEnd) {
    returnFromCall(This[OtherIt->first], OtherIt->second);
    OtherIt++;
  }

  // Handle the remaining elements of This
  while (ThisIt != ThisEnd) {
    returnFromCall(ThisIt->second, Other.Default);
    ThisIt++;
  }

  // Handle the elements we registered
  for (auto *P : Missing)
    returnFromCall(This[P->first], P->second);
}

template<typename K, typename T1, size_t N>
void unknownFunctionCall(DefaultMap<K, T1, N> &This) {
  This.Default.unknownFunctionCall();
  for (auto &P : This)
    P.second.unknownFunctionCall();
}

template<typename K, typename T1, size_t N>
void disable(DefaultMap<K, T1, N> &This) {
  This.Default.disable();
  for (auto &P : This)
    P.second.disable();
}

template<typename K, typename T1, size_t N>
void enable(DefaultMap<K, T1, N> &This) {
  This.Default.enable();
  for (auto &P : This)
    P.second.enable();
}

} // namespace MapHelpers

/// \brief Wrapper for an analysis that can inhibit it
template<class S>
class Inhibitor : public S {
public:
  using Base = S;

public:
  bool Enabled;

public:
  Inhibitor() : S(), Enabled(false) {}
  explicit Inhibitor(typename S::Values V) : S(V), Enabled(false) {}
  explicit Inhibitor(typename S::Values V, bool Enabled) :
    S(V),
    Enabled(Enabled) {}

  bool isEnabled() const { return Enabled; }

  void enable() { Enabled = true; }
  void disable() { Enabled = false; }

  void combine(const Inhibitor &Other) {
    // TODO: we should assert the non-enabled one is bottom, or just ignore it
    S::combine(Other);
    Enabled = Enabled || Other.Enabled;
  }

  bool greaterThan(const Inhibitor &Other) const {
    return not lowerThanOrEqual(Other);
  }

  bool lowerThanOrEqual(const Inhibitor &Other) const {
    if (isEnabled() and not Other.isEnabled())
      return false;
    else
      return S::lowerThanOrEqual(Other);
  }

  void transfer(typename S::TransferFunction T) {
    if (isEnabled())
      S::transfer(T);
  }

  void transfer(GeneralTransferFunction T) {
    if (isEnabled())
      S::transfer(T);
  }

  void dump() const { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    // If analysis is inhibited, simply wrap it in parenthesis
    if (not isEnabled())
      Output << "(";
    S::dump(Output);
    if (not isEnabled())
      Output << ")";
  }
};

/// \brief Return whether a certain analysis should start from return labels
///        only
template<typename T>
static constexpr bool isReturnOnly() {
  return false;
}

// Currently only URVOF is supposed to start from return points only
template<>
constexpr bool isReturnOnly<UsedReturnValuesOfFunction>() {
  return true;
}

/// \brief Recursive template class to apply certain methods on all the analyses
///        in Tuple
///
/// This class has many template argument which are used only in certain
/// functions. This saves from partial function specialization and from having
/// on class per function.
///
/// \tparam Tuple the tuple of analysis to use
/// \tparam T see dumpAnalysis
/// \tparam Diff see dumpAnalysis
/// \tparam EarlyExit see dumpAnalysis
/// \tparam NextIndex index of the tuple type, used for the recursion
template<typename Tuple,
         typename T = int,
         bool Diff = false,
         bool EarlyExit = false,
         size_t NextIndex = tuple_size<Tuple>::value>
struct AnalysesWrapperHelpers {

  using Next = AnalysesWrapperHelpers<Tuple, T, Diff, EarlyExit, NextIndex - 1>;
  static const size_t Index = NextIndex - 1;
  using Type = typename tuple_element<Index, Tuple>::type::Base;

  static typename tuple_element<Index, Tuple>::type &get(Tuple &This) {
    return std::get<Index>(This);
  }

  static const typename tuple_element<Index, Tuple>::type &
  get(const Tuple &This) {
    return std::get<Index>(This);
  }

  static void initial(Tuple &This, bool IsReturn) {
    bool Enable = isReturnOnly<Type>() ? IsReturn : true;
    get(This) = Inhibitor<Type>(Type::initial(), Enable);
    Next::initial(This, IsReturn);
  }

  static void combine(Tuple &This, const Tuple &Other) {
    get(This).combine(std::get<Index>(Other));
    Next::combine(This, Other);
  }

  // TODO: maybe we should call these "collect"
  static void assign(RegisterState &This, const Tuple &Other) {
    This.getByType<Type>() = std::get<Index>(Other);
    Next::assign(This, Other);
  }

  static void assign(CallSiteRegisterState &This, const Tuple &Other) {
    This.getByType<Type>() = std::get<Index>(Other);
    Next::assign(This, Other);
  }

  static void disable(Tuple &This) {
    get(This).disable();
    Next::disable(This);
  }

  static void enable(Tuple &This) {
    get(This).enable();
    Next::enable(This);
  }

  static void transfer(Tuple &This, GeneralTransferFunction TF) {
    get(This).transfer(TF);
    Next::transfer(This, TF);
  }

  static void dumpAnalysis(const Tuple &This, T &Output, const char *Prefix) {
    StackAnalysis::dumpAnalysis(Output, Prefix, get(This));
    Next::dumpAnalysis(This, Output, Prefix);
  }

  static void returnFromCall(Tuple &This, const RegisterState &Other) {
    get(This).transfer(Other.getByType<Type>().returnTransferFunction());
    Next::returnFromCall(This, Other);
  }

  static unsigned cmp(const Tuple &This, const Tuple &Other) {
    unsigned Result = 0;
    Result = !get(This).lowerThanOrEqual(std::get<Index>(Other));
    if (Result != 0) {
      if (EarlyExit)
        return Result;

      if (SaDiffLog.isEnabled() and Diff) {
        SaDiffLog << Type::name() << ": ";
        get(This).dump(SaDiffLog);
        SaDiffLog << " and ";
        std::get<Index>(Other).dump(SaDiffLog);
        SaDiffLog << DoLog;
      }
    }

    return Result + Next::cmp(This, Other);
  }
};

/// \brief Specialization for the base case (NextIndex == 0)
template<typename Tuple, typename T, bool Diff, bool EarlyExit>
struct AnalysesWrapperHelpers<Tuple, T, Diff, EarlyExit, 0> {
  static void initial(Tuple &, bool) {}
  static void assign(Tuple &, const Tuple &) {}
  static void combine(Tuple &, const Tuple &) {}
  static void assign(RegisterState &, const Tuple &) {}
  static void assign(CallSiteRegisterState &, const Tuple &) {}
  static void disable(Tuple &) {}
  static void enable(Tuple &) {}
  static void transfer(Tuple &, GeneralTransferFunction) {}
  static void dumpAnalysis(const Tuple &, T &, const char *) {}
  static void returnFromCall(Tuple &, const RegisterState &) {}
  static unsigned cmp(const Tuple &, const Tuple &) { return 0; }
};

/// \brief Helper class to dispatch methods required by Element onto the
///        low-level analyses
template<typename Tuple>
class AnalysesWrapper {
  friend class RegisterState;
  friend class CallSiteRegisterState;

public:
  Tuple Analyses;

private:
  using H = AnalysesWrapperHelpers<Tuple>;
  using AnalysesType = Tuple;

public:
  static AnalysesWrapper initial(bool IsReturn) {
    AnalysesWrapper Result;
    H::initial(Result.Analyses, IsReturn);
    return Result;
  }

  AnalysesWrapper &combine(const AnalysesWrapper &Other) {
    H::combine(this->Analyses, Other.Analyses);
    return *this;
  }

  void disable() { H::disable(this->Analyses); }
  void enable() { H::enable(this->Analyses); }

  void write() { H::transfer(this->Analyses, GeneralTransferFunction::Write); }

  void read() { H::transfer(this->Analyses, GeneralTransferFunction::Read); }

  void unknownFunctionCall() {
    H::transfer(this->Analyses, GeneralTransferFunction::UnknownFunctionCall);
  }

  void returnFromCall(const RegisterState &Other) {
    H::returnFromCall(this->Analyses, Other);
  }

  template<bool Diff, bool EarlyExit>
  unsigned cmp(const AnalysesWrapper &Other) const {
    using H = AnalysesWrapperHelpers<Tuple, int, Diff, EarlyExit>;
    LoggerIndent<> Y(SaDiffLog);
    return H::cmp(this->Analyses, Other.Analyses);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output, const char *Prefix = "  ") const {
    using H = AnalysesWrapperHelpers<Tuple, T>;
    H::dumpAnalysis(this->Analyses, Output, Prefix);
  }
};

/// Namespace for the classes composing the monotone framework of the ABI
/// analysis (and helper classes)
namespace ABIAnalysis {

/// \brief Element of the lattice of the monotone framework, tracks the result
///        of the various analysis for each label
///
/// This class basically acts as a dispatcher of the various actions/transfer
/// functions towards the underlying analysis specified in Analyses
///
/// \tparam Analyses an AnalysesList type listing all the function and funcion
///         call analysis to perform.
template<typename Analyses>
class Element {
  friend class ::StackAnalysis::FunctionABI;

private:
  using AWF = AnalysesWrapper<typename Analyses::Function>;
  using AWFC = AnalysesWrapper<typename Analyses::FunctionCall>;

private:
  /// Map tracking the status of registers from the point of view of the current
  /// function
  DefaultMap<int32_t, AWF, 20> RegisterAnalyses;

  /// Map tracking the status of registers from the point of view of the each
  /// function call
  // TODO: We could have as well have a vector here, considering calls are
  //       relatively rare
  MapOfMaps<FunctionCall, 5, int32_t, AWFC, 20> FunctionCallRegisterAnalyses;

public:
  Element() {}

  static Element bottom() { return Element(); }

  /// \brief Explicit copy constructor
  Element copy() const {
    Element Result;
    Result.RegisterAnalyses = RegisterAnalyses;
    Result.FunctionCallRegisterAnalyses = FunctionCallRegisterAnalyses;
    return Result;
  }

  Element(const Element &) = delete;
  Element &operator=(const Element &) = delete;

  Element(Element &&) = default;
  Element &operator=(Element &&) = default;

public:
  /// Reset and enable all the function analyses
  ///
  /// This function enables all the function analyses except those that need to
  /// start from a return basic block. In such cases, the analysis is enabled
  /// only if \p IsReturn is true.
  ///
  /// \param IsReturn whether the current block is a return basic block or not
  void resetFunctionAnalyses(bool IsReturn) {
    RegisterAnalyses.clear(AWF::initial(IsReturn));
  }

  /// \brief Enable all the function call analyses associated to \p TheCall
  void resetFunctionCallAnalyses(FunctionCall TheCall) {
    MapHelpers::unknownFunctionCall(FunctionCallRegisterAnalyses[TheCall]);
    MapHelpers::enable(FunctionCallRegisterAnalyses[TheCall]);
    FunctionCallRegisterAnalyses[TheCall].clear(AWFC::initial(true));
  }

  bool lowerThanOrEqual(const Element &Other) const {
    return cmp<false, true>(Other) == 0;
  }

  // TODO: review
  template<bool Diff, bool EarlyExit>
  unsigned cmp(const Element &Other, const Module *M = nullptr) const {
    using namespace MapHelpers;
    LoggerIndent<> Y(SaDiffLog);
    unsigned Result = 0;

    auto registerCmp = cmpWithModule<int32_t, AWF, Diff, EarlyExit, 20>;

    ROA((registerCmp(RegisterAnalyses, Other.RegisterAnalyses, CPU, M)),
        { revng_log(SaDiffLog, "RegisterAnalyses"); });

    auto X = nestedCmpWithModule<int32_t, AWFC, Diff, EarlyExit, 5, 20>;
    ROA((X(FunctionCallRegisterAnalyses,
           Other.FunctionCallRegisterAnalyses,
           CPU,
           M)),
        { revng_log(SaDiffLog, "RegisterAnalyses"); });

    return Result;
  }

  bool greaterThan(const Element &Other) const {
    return not lowerThanOrEqual(Other);
  }

  Element &combine(const Element &Other) {
    MapHelpers::combine(RegisterAnalyses, Other.RegisterAnalyses);
    MapHelpers::combine(FunctionCallRegisterAnalyses,
                        Other.FunctionCallRegisterAnalyses);
    return *this;
  }

  /// \brief Record that \p Slot has been written
  void write(ASSlot Slot) {
    // It should touch the slot at the given offset plus the slot in all the
    // function call analyses, including default.

    if (Slot.addressSpace() == CPU) {
      RegisterAnalyses[Slot.offset()].write();
      FunctionCallRegisterAnalyses.Default[Slot.offset()].write();
      for (auto &P : FunctionCallRegisterAnalyses)
        P.second[Slot.offset()].write();
    }
  }

  /// \brief Record that \p Slot has been read
  void read(ASSlot Slot) {
    // It should touch the slot at the given offset plus the slot in all the
    // function call analyses, including default.

    if (Slot.addressSpace() == CPU) {
      RegisterAnalyses[Slot.offset()].read();
      FunctionCallRegisterAnalyses.Default[Slot.offset()].read();
      for (auto &P : FunctionCallRegisterAnalyses)
        P.second[Slot.offset()].read();
    }
  }

  /// \brief Handle a call to a function for which the ABI analysis produced
  ///        \p Other
  void directCall(const FunctionABI &CalleeABI) {
    // It should touch all the register/stack slots plus all the register of
    // every function call (including default).

    // All register analyses
    MapHelpers::returnFromCall(RegisterAnalyses, CalleeABI.RegisterAnalyses);

    // All the register analyses of all the function calls (including default)
    MapHelpers::returnFromCall(FunctionCallRegisterAnalyses.Default,
                               CalleeABI.RegisterAnalyses);
    for (auto &P : FunctionCallRegisterAnalyses)
      MapHelpers::returnFromCall(P.second, CalleeABI.RegisterAnalyses);
  }

  void indirectCall() {
    // It should touch all the register plus all the register/stack slots of
    // every function call (including default).

    // All register analyses
    MapHelpers::unknownFunctionCall(RegisterAnalyses);

    // All the register analyses of all the function calls (including default)
    MapHelpers::unknownFunctionCall(FunctionCallRegisterAnalyses.Default);
    for (auto &P : FunctionCallRegisterAnalyses)
      MapHelpers::unknownFunctionCall(P.second);
  }

  void dump(const Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const Module *M, T &Output) const {
    std::stringstream Stream;
    dumpInternal(M, Stream);
    Output << Stream.str();
  }

private:
  void dumpInternal(const Module *M, std::stringstream &Output) const {
    MapHelpers::dump(M, Output, RegisterAnalyses, CPU);
    MapHelpers::dump(M, Output, FunctionCallRegisterAnalyses, CPU, "  ");
  }
};

/// \brief Given a tuple, produce a new tuple where each element is wrapped in
///        another template class
///
/// \tparam Wrapper the template class to use for wrapping the elements of the
///         tuple.
/// \tparam Tuple the tuple to wrap.
template<template<typename X> class Wrapper,
         typename Tuple,
         int I = tuple_size<Tuple>::value,
         typename... Types>
class WrapIn {
public:
  /// The resulting tuple
  using Wrapped = Wrapper<typename tuple_element<I - 1, Tuple>::type>;
  using type = typename WrapIn<Wrapper, Tuple, I - 1, Wrapped, Types...>::type;
};

template<template<typename X> class Wrapper, typename Tuple, typename... Types>
class WrapIn<Wrapper, Tuple, 0, Types...> {
public:
  using type = std::tuple<Types...>;
};

/// \brief Compile-time container for a set of function and function call
///        analyses
///
/// \tparam A tuple of function analyses
/// \tparam A tuple of function call analyses
template<typename F, typename FC>
class AnalysesList {
public:
  using Function = typename WrapIn<Inhibitor, F>::type;
  using FunctionCall = typename WrapIn<Inhibitor, FC>::type;
};

template<typename E>
class Interrupt {
private:
  enum Reason { Regular, Return, NoReturn, Summary };

private:
  Reason TheReason;
  Element<E> Result;

private:
  explicit Interrupt(Reason TheReason, Element<E> Result) :
    TheReason(TheReason),
    Result(std::move(Result)) {}

  explicit Interrupt(Reason TheReason) : TheReason(TheReason), Result() {}

public:
  static Interrupt createRegular(Element<E> Result) {
    return Interrupt(Regular, std::move(Result));
  }

  static Interrupt createReturn(Element<E> Result) {
    return Interrupt(Return, std::move(Result));
  }

  static Interrupt createNoReturn() { return Interrupt(NoReturn); }

  static Interrupt createSummary(Element<E> Result) {
    return Interrupt(Summary, std::move(Result));
  }

public:
  bool requiresInterproceduralHandling() {
    switch (TheReason) {
    case Regular:
    case Return:
      return false;
    case NoReturn:
    case Summary:
      return true;
    }

    revng_abort();
  }

  bool isReturn() const {
    revng_assert(TheReason == Regular or TheReason == Return);
    return TheReason == Return;
  }

  Element<E> &&extractResult() { return std::move(Result); }
};

/// \brief The core of the ABI analysis
///
/// This monotone framework implements the ABI analysis.
///
/// \tparam IsForward whether the analysis should be performed forward or not
/// \tparam E an AnalysesList type listing all the function and funcion call
///         analysis to perform.
///
/// \note Don't reset and re-run this analysis
template<bool IsForward, typename E>
class Analysis
  : public MonotoneFramework<ABIIRBasicBlock *,
                             Element<E>,
                             Interrupt<E>,
                             Analysis<IsForward, E>,
                             ABIIRBasicBlock::links_const_range,
                             IsForward ? ReversePostOrder : PostOrder> {

private:
  using DirectedLabelRange = typename conditional<IsForward,
                                                  ABIIRBB::range,
                                                  ABIIRBB::reverse_range>::type;

public:
  using Base = MonotoneFramework<ABIIRBasicBlock *,
                                 Element<E>,
                                 Interrupt<E>,
                                 Analysis<IsForward, E>,
                                 ABIIRBasicBlock::links_const_range,
                                 IsForward ? ReversePostOrder : PostOrder>;
  using LabelRange = typename Base::LabelRange;

private:
  /// The entry basic block of the function
  ABIIRBasicBlock *FunctionEntry;

  /// Counter for basic block visits, for statistical purposes
  unsigned VisitsCount;

  /// Flag to prevent the analysis from being run more than once
  bool FirstRun;

public:
  Analysis(ABIIRBasicBlock *FunctionEntry) :
    Base(FunctionEntry),
    FunctionEntry(FunctionEntry),
    VisitsCount(0),
    FirstRun(true) {}

public:
  void assertLowerThanOrEqual(const Element<E> &A, const Element<E> &B) const {
    const Module *M = getModule(FunctionEntry->basicBlock());
    ::StackAnalysis::assertLowerThanOrEqual(A, B, M);
  }

  /// \brief Prevent the analysis from running twice
  void initialize() {
    revng_assert(FirstRun, "The ABIAnalysis cannot be run twice");
    FirstRun = false;
    Base::initialize();
  }

  void dumpFinalState() const {}

  llvm::Optional<Element<E>> handleEdge(const Element<E> &Original,
                                        ABIIRBasicBlock *Source,
                                        ABIIRBasicBlock *Destination) const {
    return llvm::Optional<Element<E>>();
  }

  ABIIRBasicBlock::links_const_range
  successors(ABIIRBasicBlock *BB, Interrupt<E> &) const {
    return BB->next<IsForward>();
  }

  size_t successor_size(ABIIRBasicBlock *BB, Interrupt<E> &) const {
    return BB->next_size<IsForward>();
  }

  Interrupt<E> createSummaryInterrupt() {
    return Interrupt<E>::createSummary(std::move(this->FinalResult));
  }

  Interrupt<E> createNoReturnInterrupt() const {
    return Interrupt<E>::createNoReturn();
  }

  Element<E> extremalValue(ABIIRBasicBlock *BB) const {
    Element<E> Result;

    // Initialize to `::initial()` and enable all the function-related
    // analyses. Some of the backward analyses are available only if we're
    // starting from a proper return.
    Result.resetFunctionAnalyses(BB->isReturn());

    return Result;
  }

  unsigned visitsCount() const { return VisitsCount; }

  Interrupt<E> transfer(ABIIRBasicBlock *BB) {
    revng_log(SaABI, "Analyzing " << BB->basicBlock());
    Element<E> Result = this->State[BB].copy();

    VisitsCount++;

    for (ABIIRInstruction &I : range(BB)) {

      // Result is Element<E>
      switch (I.opcode()) {
      case ABIIRInstruction::Load:
        Result.read(I.target());
        break;

      case ABIIRInstruction::Store:
        Result.write(I.target());
        break;

      case ABIIRInstruction::DirectCall:
        Result.directCall(I.abi());
        break;

      case ABIIRInstruction::IndirectCall:
        Result.indirectCall();
        break;
      }

      // Once we get to a function call, if it's the first time we meet it, its
      // analyses are going to be disabled. Here we first activate the unknown
      // function call transfer function (while it might still be disabled) and
      // then we enable all the analyses.
      if (I.opcode() == ABIIRInstruction::DirectCall
          or I.opcode() == ABIIRInstruction::IndirectCall) {
        Result.resetFunctionCallAnalyses(I.call());
      }
    }

    // We don't check BB->isReturn() since there are basic blocks that have no
    // successors but are not returns. And we want to consider those too, unlike
    // what happens with the stack analysis, where we are intersted in
    // understanding what happens from the point of view of the caller (e.g., if
    // a callee-saved register is not restored on a noreturn path, we don't
    // care).
    if ((IsForward and BB->successor_size() == 0)
        or (not IsForward and BB->predecessor_size() == 0))
      return Interrupt<E>::createReturn(std::move(Result));
    else
      return Interrupt<E>::createRegular(std::move(Result));
  }

private:
  DirectedLabelRange range(ABIIRBasicBlock *BB) {
    return instructionRange<DirectedLabelRange, IsForward>(BB);
  }
};

} // namespace ABIAnalysis

//
// FunctionaABI methods
//

void FunctionABI::analyze(const ABIFunction &TheFunction) {
  using namespace ABIAnalysis;

  {
    revng_log(SaABI, "Running forward function analyses");

    // List of the forward ABI analyses to perform
    // Note: Among the function analyses we also have an instance of the
    //       function call analyses so that we can use them interproceduraly to
    //       simulate the inling of the called function.
    using DRAOF = DeadRegisterArgumentsOfFunction;
    using UAOF = UsedArgumentsOfFunction;
    using URVOFC = UsedReturnValuesOfFunctionCall;
    using DRVOFC = DeadReturnValuesOfFunctionCall;
    using FunctionWise = tuple<DRAOF, UAOF, URVOFC, DRVOFC>;
    using FunctionCallWise = tuple<URVOFC, DRVOFC>;
    using ForwardList = AnalysesList<FunctionWise, FunctionCallWise>;

    Analysis<true, ForwardList> ForwardFunctionAnalyses(TheFunction.entry());

    ForwardFunctionAnalyses.registerExtremal(TheFunction.entry());

    ForwardFunctionAnalyses.initialize();
    Interrupt<ForwardList> Result = ForwardFunctionAnalyses.run();

    int Average = ForwardFunctionAnalyses.visitsCount() / TheFunction.size();
    revng_log(SaABI,
              "Forward function analyses terminated: "
                << ForwardFunctionAnalyses.visitsCount() << " visits performed"
                << " on " << TheFunction.size() << " blocks ("
                << "average: " << Average << ").");

    this->combine(Result.extractResult());
  }

  {
    revng_log(SaABI,
              "Running backward function analyses ("
                << TheFunction.finals_size() << " return points)");
    /// List of the backward ABI analyses to perform
    using URVOF = UsedReturnValuesOfFunction;
    using RAOFC = RegisterArgumentsOfFunctionCall;
    using FunctionWise = tuple<URVOF, RAOFC>;
    using FunctionCallWise = tuple<RAOFC>;
    using BackwardList = AnalysesList<FunctionWise, FunctionCallWise>;
    Analysis<false, BackwardList> BackwardFunctionAnalyses(TheFunction.entry());

    for (ABIIRBasicBlock *FinalBB : TheFunction.finals())
      BackwardFunctionAnalyses.registerExtremal(FinalBB);

    BackwardFunctionAnalyses.initialize();
    Interrupt<BackwardList> Result = BackwardFunctionAnalyses.run();
    this->combine(Result.extractResult());
  }
}

void FunctionABI::dumpInternal(const Module *M,
                               std::stringstream &Output) const {
  MapHelpers::dump(M, Output, RegisterAnalyses, CPU);

  Output << "Calls:\n\n";
  for (auto &P : Calls) {
    Output << "  ";
    P.first.dump(Output);
    Output << ":\n";
    MapHelpers::dump(M, Output, P.second.Registers, CPU, "    ");
    Output << "\n";
  }
}

} // namespace StackAnalysis
