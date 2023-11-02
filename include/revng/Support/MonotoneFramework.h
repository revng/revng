#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/Queue.h"
#include "revng/Support/Debug.h"

/// Backport of std::map::insert_or_assign
template<typename K, typename V>
inline void insert_or_assign(std::map<K, V> &Map, K Key, V &&Value) {
  auto It = Map.find(Key);
  if (It != Map.end())
    It->second = std::forward<V>(Value);
  else
    Map.emplace(Key, std::forward<V>(Value));
}

enum VisitType {
  /// Breadth first visit, useful if the function body is unknown
  BreadthFirst,
  /// Post order visit, for backward analyses
  PostOrder,
  /// Reverse post order visit, for forward analyses
  ReversePostOrder
};

/// Work list for the monotone framework supporting various visit strategies
template<typename Iterated, VisitType Visit>
class MonotoneFrameworkWorkList {};

// Breadth first implementation
template<typename Iterated>
class MonotoneFrameworkWorkList<Iterated, BreadthFirst> {
private:
  UniquedQueue<Iterated> Queue;

public:
  MonotoneFrameworkWorkList(Iterated) {}
  void clear() { Queue.clear(); }
  void insert(Iterated Entry) { Queue.insert(Entry); }
  bool empty() const { return Queue.empty(); }
  Iterated head() const { return Queue.head(); }
  Iterated pop() { return Queue.pop(); }
  size_t size() const { return Queue.size(); }
};

// (Reverse) post order implementation
template<typename Iterated, VisitType Visit>
  requires(Visit == PostOrder or Visit == ReversePostOrder)
class MonotoneFrameworkWorkList<Iterated, Visit> {
private:
  /// Class for an entry in the work list
  ///
  /// All the basic blocks are always in the list in (reverse) post order. When
  /// an entry is popped it is simply disabled.
  class PostOrderEntry {
  private:
    Iterated Entry;
    bool Enabled;

  public:
    PostOrderEntry(Iterated Entry) : Entry(Entry), Enabled(true) {}
    void enable() { Enabled = true; }
    void disable() { Enabled = false; }
    bool isEnabled() const { return Enabled; }
    Iterated entry() const { return Entry; }
  };

private:
  /// List of all basic blocks in the appropriate order
  std::vector<PostOrderEntry> PostOrderList;

  /// Map to quickly find the index of an entry in PostOrderList
  std::map<Iterated, size_t> PostOrderListIndex;

  /// The next index to consume. This should always point to the lowest enabled
  /// entry in PostOrderList
  size_t Next;

  /// Special value for Next to indicate that the work list is empty
  const static size_t InvalidIndex = std::numeric_limits<size_t>::max();

public:
  MonotoneFrameworkWorkList(const std::vector<Iterated> &RPOT) {
    for (Iterated Entry : RPOT)
      PostOrderList.push_back(PostOrderEntry(Entry));

    initialize();
  }

  MonotoneFrameworkWorkList(const llvm::SmallVectorImpl<Iterated> &RPOT) {
    for (Iterated Entry : RPOT)
      PostOrderList.push_back(PostOrderEntry(Entry));

    initialize();
  }

  MonotoneFrameworkWorkList(Iterated Entry) :
    MonotoneFrameworkWorkList(buildRPOT(Entry)) {}

  size_t size() const {
    revng_assert(verify());

    if (empty())
      return 0;

    size_t Result = 0;
    for (size_t I = Next; I < PostOrderList.size(); I++)
      if (PostOrderList[I].isEnabled())
        Result++;

    return Result;
  }

  void clear() {
    for (PostOrderEntry &Entry : PostOrderList)
      Entry.disable();
    Next = InvalidIndex;
  }

  void insert(Iterated Entry) {
    // Find the entry
    auto It = PostOrderListIndex.find(Entry);
    revng_assert(It != PostOrderListIndex.end());

    // Enable it
    PostOrderList[It->second].enable();

    // Reset next to the lowest enabled index, if necessary
    Next = std::min(Next, It->second);
  }

  bool empty() const { return Next == InvalidIndex; }

  Iterated head() const {
    revng_assert(Next != InvalidIndex);
    return PostOrderList[Next].entry();
  }

  Iterated pop() {
    revng_assert(not empty());
    revng_assert(verify());

    // Start from the next element and look for an enabled element
    size_t I = Next + 1;
    for (; I < PostOrderList.size(); I++)
      if (PostOrderList[I].isEnabled())
        break;

    // Update Next
    size_t OldNext = Next;
    Next = (I >= PostOrderList.size()) ? InvalidIndex : I;

    // Consume the previous next
    PostOrderList[OldNext].disable();

    // Return the consumed entry
    return PostOrderList[OldNext].entry();
  }

private:
  static std::vector<Iterated> buildRPOT(Iterated Entry) {
    // Populate PostOrderList
    std::vector<Iterated> RPOT;
    for (Iterated I : llvm::ReversePostOrderTraversal<Iterated>(Entry))
      RPOT.push_back(I);

    return RPOT;
  }

private:
  void initialize() {
    // Reverse the list in case we don't want the reverse post order
    if (Visit == PostOrder)
      std::reverse(PostOrderList.begin(), PostOrderList.end());

    // Populate the index, used for faster lookups
    for (unsigned I = 0; I < PostOrderList.size(); I++)
      PostOrderListIndex[PostOrderList[I].entry()] = I;

    // Initialize the next index
    Next = (PostOrderList.size() > 0) ? 0 : InvalidIndex;
  }

  bool verify() const {
    if (PostOrderList.size() == 0 and not empty())
      return false;

    if (empty()) {
      // If the worklist is empty, no elements should be enabled
      for (const PostOrderEntry &Entry : PostOrderList)
        if (Entry.isEnabled())
          return false;
    } else {
      // Otherwise, all the elements up to next should be disabled
      for (size_t I = 0; I != Next; I++)
        if (PostOrderList[I].isEnabled())
          return false;

      if (not PostOrderList[Next].isEnabled())
        return false;
    }

    return true;
  }
};

/// CRTP base class for an element of the lattice
///
/// \note This class is more for reference. It's unused.
///
/// \tparam D the derived class.
template<typename D>
class ElementBase {
public:
  /// The partial ordering relation
  bool lowerThanOrEqual(const ElementBase &RHS) const {
    const D &This = *static_cast<const D *>(this);
    const D &Other = static_cast<const D &>(RHS);
    return This.lowerThanOrEqual(Other);
  }

  /// The combination operator
  // TODO: assert monotonicity
  ElementBase &combine(const ElementBase &RHS) {
    return static_cast<const D *>(this)->combine(static_cast<const D &>(RHS));
  }
};

/// Default class to represent a simple Interrupt for a MonotoneFramework
///
/// This class provides the simplest possible implementation for an Interrupt
/// for a Monotone Framework.
///
/// In particular this interrupt is suitable for MonotoneFrameworks that are NOT
/// interprocedural, and that DO NOT need to combine all the results on the
/// terminal labels at the end of the analysis in a single FinalResult.
///
/// With these assumptions, the resulting Interrupt is pretty simple and it just
/// forwards the results of the transfer function.
template<typename LatticeElement>
class DefaultInterrupt {
private:
  explicit DefaultInterrupt(const LatticeElement &Element) :
    Result(Element.copy()) {}
  explicit DefaultInterrupt(LatticeElement &&Element) : Result(Element) {}

public:
  explicit DefaultInterrupt() = default;

  static DefaultInterrupt createInterrupt(const LatticeElement &Element) {
    return DefaultInterrupt(Element);
  }

  static DefaultInterrupt createInterrupt(LatticeElement &&Element) {
    return DefaultInterrupt(Element);
  }

public:
  static bool requiresInterproceduralHandling() { return false; }
  LatticeElement &&extractResult() { return std::move(Result); }
  static bool isPartOfFinalResults() { return false; }

private:
  LatticeElement Result;
};

/// Helper struct for creation of Interrupts for MonotoneFramework
///
/// This is for creating generic Interrupts.
/// In this case we delegate the construction of the Interrupts to the
/// derived class in the CRTP.
/// There is a full specialization for DefaultInterrupt<LatticeElement>.
///
/// \tparam D the CRTP derived class of MonotoneFramework
/// \tparam LatticeElement the type representing an element of the lattice of
///         the MonotoneFramework
/// \tparam InterruptTy the type representing an Interrupt of the
///         MonotoneFramework
template<typename D, typename LatticeElement, typename InterruptTy>
struct InterruptCreator {
  InterruptTy createSummaryInterrupt(D &I) {
    return I.createSummaryInterrupt();
  }

  InterruptTy createNoReturnInterrupt(D &I) {
    return I.createNoReturnInterrupt();
  }
};

/// Specialization of InterruptCreator for DefaultInterrupt
///
/// This is for creating DefaultInterrupt<LatticeElement>.
/// In case of DefaultInterrupt the Summary Interrupt is never created,
/// because there is never a Final State to compute the Summary.
///
/// \tparam D the CRTP derived class of MonotoneFramework
/// \tparam LatticeElement the type representing an element of the lattice of
///         the MonotoneFramework
template<typename D, typename LatticeElement>
struct InterruptCreator<D, LatticeElement, DefaultInterrupt<LatticeElement>> {
  DefaultInterrupt<LatticeElement> createSummaryInterrupt(D &) {
    revng_abort();
    return DefaultInterrupt<LatticeElement>();
  }

  DefaultInterrupt<LatticeElement> createNoReturnInterrupt(D &) {
    return DefaultInterrupt<LatticeElement>();
  }
};
/// CRTP base class for implementing a monotone framework
///
/// This class provides the base structure to implement an analysis based on a
/// monotone framework. It also provides an implementation of the MFP solution.
///
/// For further information about monotone frameworks see "Principles of Program
/// Analysis" (by Nielson, Flemming), Chapter 2.
///
/// To use this class you need to define a Label (typically the basic block of
/// the IR you're working on), a class representing an element of the lattice
/// (LatticeElement, see ElementBase) and a class representing an Interrupt
/// reason of the analysis. It is suggested to create a namespace for these
/// classes and keep their names simple: Analysis for the class inheriting from
/// MonotoneFramework, Element for LatticeElement and Interrupt for Interrupt.
///
/// \tparam Label the type identifying a "label" in the monotone framework,
///         typically an instruction or a basic block.
/// \tparam LatticeElement the type representing an element of the lattice.
/// \tparam Interrupt the type describing why the analysis has been interrupted.
/// \tparam D the derived class.
/// \tparam SuccessorsRange the return type of D::successors.
/// \tparam Visit type of visit to perform.
// TODO: static_assert features of these classes (Interrupt in particular)
template<typename D,
         typename Label,
         typename LatticeElement,
         VisitType Visit,
         typename SuccessorsRange,
         typename Interrupt = DefaultInterrupt<LatticeElement>,
         bool DynamicGraph = false>
class MonotoneFramework {
  static_assert(DynamicGraph ? Visit == BreadthFirst : true,
                "Cannot compute (reverse) post order for dynamic graphs");

protected:
  /// Lattice element where the results on return points of the function are
  /// accumulated
  LatticeElement FinalResult;

  /// Have we already met at least return label? This is used to ensure that the
  /// first final result we get is assigned to FinalResult and we're not
  /// combining with an uninitialized FinalResult.
  ///
  /// \note Unused if DynamicGraph == true
  bool FirstFinalResult;

  MonotoneFrameworkWorkList<Label, Visit> WorkList;

  /// State of the monotone framework, maps a label to a lattice element
  std::map<Label, LatticeElement> State;

  /// List of basic blocks we want to be sure to visit again before the end of
  /// the analysis
  std::set<Label> ToVisit;

  /// Set of extremal (i.e., initial) labels
  std::set<Label> Extremals;

  /// Final states and associated lattice elements
  ///
  /// This is used, in case we're dealing with a dynamic graph, to avoid merging
  /// in FinalResult final states that are not actually reachable from the entry
  /// point in the final graph.
  ///
  /// \note Unused if DynamicGraph == false
  std::vector<std::pair<Label, LatticeElement>> FinalStates;

  /// Keep track of the edges of the control flow graph
  ///
  /// This is used to identify the set of labels reachable from Entry in a
  /// dynamic graph, and merge in FinalResult only the entries of FinalStates
  /// that are actually reachable.
  ///
  /// \note Unused if DynamicGraph == false
  std::map<Label, llvm::SmallVector<Label, 2>> SuccessorsMap;

public:
  using InterruptType = Interrupt;

  MonotoneFramework(Label Entry) :
    FinalResult(LatticeElement::bottom()), WorkList(Entry) {}

  MonotoneFramework(const std::vector<Label> &RPOT) :
    FinalResult(LatticeElement::bottom()), WorkList(RPOT) {}

  MonotoneFramework(const llvm::SmallVectorImpl<Label> &RPOT) :
    FinalResult(LatticeElement::bottom()), WorkList(RPOT) {}

private:
  const D &derived() const { return *static_cast<const D *>(this); }
  D &derived() { return *static_cast<D *>(this); }
  InterruptCreator<D, LatticeElement, InterruptType> TheInterruptCreator;

public:
  /// The transfer function
  ///
  /// Starting from the initial state at \p L provides a new lattice element or
  /// a reason why the analysis has be interrupted.
  ///
  /// \note This method must be implemented by the derived class D
  Interrupt transfer(Label L) { return derived().transfer(L); }

  /// Return the element of the lattice associated with the extremal label \p L
  ///
  /// \note This method must be implemented by the derived class D
  LatticeElement extremalValue(Label L) const {
    return derived().extremalValue(L);
  }

  /// Create a "summary" interrupt, used upon a regular analysis completion
  ///
  /// \note This method must be implemented by the derived class D only if
  ///       Interrupt != DefaultInterrupt<LatticeElement>
  Interrupt createSummaryInterrupt() {
    return TheInterruptCreator.createSummaryInterrupt(derived());
  }

  /// Create a "no return" interrupt, used when the analysis terminates without
  /// identifying a return basic block
  ///
  /// \note This method must be implemented by the derived class D only if
  ///       Interrupt != DefaultInterrupt<LatticeElement>
  Interrupt createNoReturnInterrupt() {
    return TheInterruptCreator.createNoReturnInterrupt(derived());
  }

  /// Dump the final state
  ///
  /// \note This method must be implemented by the derived class D
  void dumpFinalState() const { return derived().dumpFinalState(); }

  /// Get the successors of label \p L
  ///
  /// Also the interrupt \p I is given since it can sometimes be useful to
  /// provide a different set of successors.
  ///
  /// \note This method must be implemented by the derived class D
  SuccessorsRange successors(Label &L, Interrupt &I) const {
    return derived().successors(L, I);
  }

  /// \note This method must be implemented by the derived class D
  size_t successor_size(Label &L, Interrupt &I) const {
    return derived().successor_size(L, I);
  }

  /// Assert that \p A is lower than or equal \p B, useful for debugging
  /// purposes
  ///
  /// \note This method must be implemented by the derived class D
  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    derived().assertLowerThanOrEqual(A, B);
  }

  /// Handle the propagation of \p Original from \p Source to \p Destination
  ///
  /// \return Empty optional value if \p Original is fine, a new LatticeElement
  ///         otherwise.
  std::optional<LatticeElement>
  handleEdge(const LatticeElement &Original, Label Source, Label Destination) {
    return derived().handleEdge(Original, Source, Destination);
  }

  /// Initialize/reset the analysis
  ///
  /// Call this method before invoking run or if you want to reset the state of
  /// the analysis to run it again.
  void initialize() {
    FirstFinalResult = true;
    FinalStates.clear();
    State.clear();
    WorkList.clear();
    ToVisit.clear();

    for (Label ExtremalLabel : Extremals) {
      WorkList.insert(ExtremalLabel);
      insert_or_assign(State, ExtremalLabel, extremalValue(ExtremalLabel));
    }
  }

  /// Registers \p L to be visited before the end of the analysis
  ///
  /// If \p L has already been visited at least once before, it's simply
  /// enqueued in the WorkList, otherwise is registered to be visited at least
  /// once before the end of the analysis.
  ///
  /// This function is required when you want to visit a basic block only if
  /// it's part of the current function, or fail otherwise.
  void registerToVisit(Label L) {
    if (!State.contains(L))
      ToVisit.insert(L);
    else
      WorkList.insert(L);
  }

  /// Number of label analyzed so far
  size_t size() const { return State.size(); }

  /// Register a new extremal label
  void registerExtremal(Label L) { Extremals.insert(L); }

  /// Resolve the data flow analysis problem using the MFP solution
  Interrupt run() {
    using namespace llvm;

    // Proceed until there are elements in the work list
    while (not WorkList.empty()) {
      Label ToAnalyze = WorkList.head();

      // If we've been asked to visit this basic block before the end, consider
      // the requested satisfied
      ToVisit.erase(ToAnalyze);

      // Run the transfer function
      Interrupt Result = transfer(ToAnalyze);

      // Check if we should continue or if we should yield control to the
      // caller, i.e., the interprocedural part of the analysis, if present.
      if (Result.requiresInterproceduralHandling())
        return Result;

      // OK, we can handle this result by ourselves: get the result and pop an
      // element from the work list
      LatticeElement &&NewLatticeElement = Result.extractResult();
      WorkList.pop();

      // Compute the number of successors
      size_t SuccessorsCount = successor_size(ToAnalyze, Result);

      // In case we have a dynamic graph, prepare for registering the successors
      // of the current label
      SmallVector<Label, 2> *Successors = nullptr;
      if (DynamicGraph)
        Successors = &SuccessorsMap[ToAnalyze];

      if (Result.isPartOfFinalResults()) {
        // The current label is a final state (but not necessarily a return or
        // without successors)

        // If so, accumulate the result in FinalResult (or in FinalStates in
        // case of dynamic graph)
        if (DynamicGraph) {
          FinalStates.emplace_back(ToAnalyze, NewLatticeElement.copy());
        } else {
          if (FirstFinalResult)
            FinalResult = std::move(NewLatticeElement);
          else
            FinalResult.combine(NewLatticeElement);
        }

        FirstFinalResult = false;

        dumpFinalState();
      }

      // The current label is NOT a final state

      // Used only if DynamicGraph
      SmallVector<Label, 2> NewSuccessors;

      // If it has successors, check if we have to re-enqueue them
      for (Label Successor : successors(ToAnalyze, Result)) {

        std::optional<LatticeElement> NewElement = handleEdge(NewLatticeElement,
                                                              ToAnalyze,
                                                              Successor);
        bool GotNewElement = NewElement.has_value();
        LatticeElement &ActualElement = GotNewElement ? *NewElement :
                                                        NewLatticeElement;

        if (DynamicGraph)
          NewSuccessors.push_back(Successor);

        auto It = State.find(Successor);
        if (It == State.end()) {
          // We have never seen this Label, register it in the analysis state

          // If this is the only successor or we got a new element we can use
          // move semantics, otherwise create a copy
          if (SuccessorsCount == 1 or GotNewElement)
            insert_or_assign(State, Successor, std::move(ActualElement));
          else
            insert_or_assign(State, Successor, ActualElement.copy());

          // Enqueue the successor
          WorkList.insert(Successor);

        } else if (not ActualElement.lowerThanOrEqual(It->second)) {
          // We have already seen this Label but the result of the transfer
          // function is larger than its previous initial state

          // Update the state merging ActualElement
          It->second.combine(ActualElement);

          // Assert we're now actually lower than or equal
          assertLowerThanOrEqual(ActualElement, It->second);

          // Re-enqueue
          WorkList.insert(Successor);
        }
      }

      // In case of dynamic graph, register successors of this label
      if (DynamicGraph) {
        // The successors must match, unless the current label has become a
        // return label or we initially had no successors
        if (Successors->size() != 0 and NewSuccessors.size() != 0)
          revng_assert(NewSuccessors == *Successors);

        *Successors = std::move(NewSuccessors);
      }
    }

    if (DynamicGraph)
      revng_assert(FirstFinalResult == (FinalStates.size() == 0));
    else
      revng_assert(FinalStates.size() == 0 and SuccessorsMap.size() == 0);

    // The work list is empty
    revng_assert(ToVisit.empty());
    if (FirstFinalResult) {
      // We haven't find any return label
      return createNoReturnInterrupt();
    } else {
      // OK, we already have at least a return label

      if (DynamicGraph) {
        // We have dynamic graph, we need to compute the set of labels reachable
        // from the extremal labels and therefore exclude from FinalResult
        // results obtained from return labels that are no longer reachable.
        //
        // We need to do this since in certain situations we temporarily visit
        // basic block that then turns out not to be part of the graph.  As a
        // concrete example, we might temporarily misdetect an indirect jump as
        // a return and then rectify this later on.

        // Find all the reachable labels
        OnceQueue<Label> ReachableLabels;

        for (Label ExtremalLabel : Extremals)
          ReachableLabels.insert(ExtremalLabel);

        // Recursively visit all the reachable labels
        while (not ReachableLabels.empty()) {
          Label L = ReachableLabels.pop();
          auto It = SuccessorsMap.find(L);
          revng_assert(It != SuccessorsMap.end());
          for (Label Successor : It->second)
            ReachableLabels.insert(Successor);
        }

        // Obtain the set of visited labels
        std::set<Label> Reachable = std::move(ReachableLabels.visited());

        // TODO: if this assert never triggers, all the SuccessorsMaps thingy is
        //       only for debugging purposes and the DynamicGraph template
        //       argument should be replaced with an `#ifndef NDEBUG`.
        if (Reachable.size() != State.size()) {
          for (const Label &L : Reachable)
            if (!State.contains(L))
              revng_abort("A label is Reachable but not present in State");

          for (const auto &P : State)
            if (!Reachable.contains(P.first))
              revng_abort("A label is in State but not Reachable");

          revng_abort();
        }

        // Merge all the final states, if they are reachable
        bool First = true;
        for (auto &P : FinalStates) {
          if (Reachable.contains(P.first)) {

            if (First)
              FinalResult = std::move(P.second);
            else
              FinalResult.combine(P.second);

            First = false;
          }
        }

        revng_assert(not First);
      }

      return createSummaryInterrupt();
    }
  }
};

/// Base class for lattices for MonotoneFrameworks built over a set of T
///
/// You can have a custom lattice for your monotone framework instance, but
/// using sets makes everything quite smooth.
///
/// \tparam T type of the elements of the set
template<typename T>
class MonotoneSet {
public:
  using const_iterator = typename std::set<T>::const_iterator;
  using size_type = typename std::set<T>::size_type;

protected:
  using iterator = typename std::set<T>::iterator;
  std::set<T> Set;

protected:
  MonotoneSet(const MonotoneSet &) = default;

public:
  MonotoneSet() = default;

  MonotoneSet copy() const { return *this; }
  MonotoneSet &operator=(const MonotoneSet &) = default;

  MonotoneSet(MonotoneSet &&) = default;
  MonotoneSet &operator=(MonotoneSet &&) = default;

public:
  const_iterator begin() const { return Set.begin(); }
  const_iterator end() const { return Set.end(); }

  void dump() const { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "{ ";
    for (const T &Value : Set)
      Output << Value << " ";
    Output << " }";
  }

protected:
  size_type size() const { return Set.size(); }

  void insert(const T Key) { Set.insert(Key); }

  size_type erase(const T &El) { return Set.erase(El); }
  const_iterator erase(const_iterator It) { return this->Set.erase(It); }

  bool contains(const T &Key) const { return Set.contains(Key); }
};

/// Lattice for a MonotoneFramework over a set, where combine is set union
template<typename T>
class UnionMonotoneSet : public MonotoneSet<T> {
private:
  UnionMonotoneSet(const UnionMonotoneSet &) = default;

public:
  UnionMonotoneSet() = default;

  UnionMonotoneSet copy() const { return *this; }
  UnionMonotoneSet &operator=(const UnionMonotoneSet &) = default;

  UnionMonotoneSet(UnionMonotoneSet &&) = default;
  UnionMonotoneSet &operator=(UnionMonotoneSet &&) = default;

  static UnionMonotoneSet bottom() { return UnionMonotoneSet(); }

public:
  typename MonotoneSet<T>::size_type size() const {
    return MonotoneSet<T>::size();
  }

  void insert(const T Key) { MonotoneSet<T>::insert(Key); }

  typename MonotoneSet<T>::size_type erase(const T &El) {
    return MonotoneSet<T>::erase(El);
  }
  typename MonotoneSet<T>::const_iterator
  erase(typename MonotoneSet<T>::const_iterator It) {
    return MonotoneSet<T>::erase(It);
  }

  bool contains(const T &Key) const { return MonotoneSet<T>::contains(Key); }

  bool contains(std::function<bool(const T &)> Predicate) const {
    return std::any_of(this->begin(), this->end(), Predicate);
  }
  bool contains_any_of(const std::set<T> &Other) const {
    return contains([&Other](const T &El) { return Other.contains(El); });
  }

  void erase_if(std::function<bool(const T &)> Predicate) {
    for (auto It = this->begin(), End = this->end(); It != End;) {
      if (Predicate(*It)) {
        It = erase(It);
      } else {
        ++It;
      }
    }
  }

  void drop(const T &Key) { this->Set.erase(Key); }

  void combine(const UnionMonotoneSet &Other) {
    // Simply join the sets
    this->Set.insert(Other.begin(), Other.end());
  }

  bool lowerThanOrEqual(const UnionMonotoneSet &Other) const {
    if (size() > Other.size())
      return false;
    return std::includes(Other.begin(),
                         Other.end(),
                         this->begin(),
                         this->end());
  }
};

/// Lattice for a MonotoneFramework over a set, where combine is set
/// intersection
template<typename T>
class IntersectionMonotoneSet : public MonotoneSet<T> {
private:
  bool IsBottom;

private:
  IntersectionMonotoneSet(const IntersectionMonotoneSet &) = default;

public:
  IntersectionMonotoneSet() : MonotoneSet<T>(), IsBottom(true){};

  IntersectionMonotoneSet copy() const { return *this; }
  IntersectionMonotoneSet &operator=(const IntersectionMonotoneSet &) = default;

  IntersectionMonotoneSet(IntersectionMonotoneSet &&) = default;
  IntersectionMonotoneSet &operator=(IntersectionMonotoneSet &&) = default;

  static IntersectionMonotoneSet bottom() { return IntersectionMonotoneSet(); }

  static IntersectionMonotoneSet top() {
    IntersectionMonotoneSet Res = {};
    Res.IsBottom = false;
    return Res;
  }

public:
  typename MonotoneSet<T>::size_type size() const {
    revng_assert(not IsBottom);
    return MonotoneSet<T>::size();
  }

  void insert(const T Key) {
    revng_assert(not IsBottom);
    MonotoneSet<T>::insert(Key);
  }

  typename MonotoneSet<T>::size_type erase(const T &El) {
    revng_assert(not IsBottom);
    return MonotoneSet<T>::erase(El);
  }
  typename MonotoneSet<T>::const_iterator
  erase(typename MonotoneSet<T>::const_iterator It) {
    revng_assert(not IsBottom);
    return MonotoneSet<T>::erase(It);
  }

  bool contains(const T &Key) const {
    revng_assert(not IsBottom);
    return MonotoneSet<T>::contains(Key);
  }

  void combine(const IntersectionMonotoneSet &Other) {
    // Simply intersects the sets
    if (Other.IsBottom)
      return;
    if (IsBottom) {
      this->Set = Other.Set;
      IsBottom = false;
      return;
    }

    llvm::erase_if(this->Set,
                   [&Other](const auto &E) { return !Other.contains(E); });
  }

  bool lowerThanOrEqual(const IntersectionMonotoneSet &Other) const {
    if (IsBottom)
      return true;
    if (Other.IsBottom)
      return false;
    if (size() < Other.size())
      return false;
    return std::includes(this->begin(),
                         this->end(),
                         Other.begin(),
                         Other.end());
  }
};
