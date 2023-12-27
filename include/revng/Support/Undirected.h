#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/iterator.h"

#include "revng/Support/Assert.h"

namespace llvm {

/// This class is used as a marker class to tell the graph iterator to treat the
/// underlying graph as an undirected one
template<class GraphType>
struct Undirected {
  const GraphType &Graph;

  inline Undirected(const GraphType &G) : Graph(G) {}
};

} // namespace llvm

/// Custom child iterator which iterates over the concatenation of the
/// successors and predecessors of a node. This is used for the `Undirected`
/// `GraphTraits` specialization. This is done taking inspiration from the
/// internals of `llvm::concat` implementation, but overcomes one of its
/// limitations. The limitation is caused by the fact that internally, the `*`
/// operator of `llvm::concat` (and more specifically the `getHelper`), first
/// dereferences the internal iterator that is being concatenated, and then
/// takes and returns its pointer (`&*It`). If the iterator used internally,
/// when dereferenced, returns a temporary object with a limited lifetime, which
/// goes out of scope before its used by the user of `llvm::concat`, we have a
/// problem. Instead here, we artificially extend the lifetime of the object
/// pointed to by the internal iterator, wrapping it into the `Proxy` struct.
/// When dereferenciation happens, no problems of lifetime happen.
// TODO: Drop this once we adopt C++26
//       (https://en.cppreference.com/w/cpp/ranges/concat_view) or if we adopt
//       range-v3
//       (https://ericniebler.github.io/range-v3/structranges_1_1views_1_1concat
//        __fn.html).
//       In such case, we can drop the below custom iterator, and use the
//       `concat`  directly in the body of the `struct
//       GraphTraits<llvm::Undirected<T *>>` implementation.
template<typename SuccessorIterator,
         typename PredecessorIterator,
         typename ValueType>
class UndirectedChildIterator : public llvm::iterator_facade_base<
                                  UndirectedChildIterator<SuccessorIterator,
                                                          PredecessorIterator,
                                                          ValueType>,
                                  std::forward_iterator_tag,
                                  ValueType> {
private:
  SuccessorIterator SBegin;
  SuccessorIterator SEnd;
  PredecessorIterator PBegin;
  PredecessorIterator PEnd;

  explicit UndirectedChildIterator(SuccessorIterator SBegin,
                                   SuccessorIterator SEnd,
                                   PredecessorIterator PBegin,
                                   PredecessorIterator PEnd) :
    SBegin(SBegin), SEnd(SEnd), PBegin(PBegin), PEnd(PEnd) {}

public:
  static UndirectedChildIterator<SuccessorIterator,
                                 PredecessorIterator,
                                 ValueType>
  begin(llvm::iterator_range<SuccessorIterator> Successors,
        llvm::iterator_range<PredecessorIterator> Predecessors) {
    return UndirectedChildIterator<SuccessorIterator,
                                   PredecessorIterator,
                                   ValueType>(std::begin(Successors),
                                              std::end(Successors),
                                              std::begin(Predecessors),
                                              std::end(Predecessors));
  }
  static UndirectedChildIterator<SuccessorIterator,
                                 PredecessorIterator,
                                 ValueType>
  end(llvm::iterator_range<SuccessorIterator> Successors,
      llvm::iterator_range<PredecessorIterator> Predecessors) {
    return UndirectedChildIterator<SuccessorIterator,
                                   PredecessorIterator,
                                   ValueType>(std::end(Successors),
                                              std::end(Successors),
                                              std::end(Predecessors),
                                              std::end(Predecessors));
  }

  using UndirectedChildIterator::iterator_facade_base::operator++;

  UndirectedChildIterator &operator++() {
    if (SBegin != SEnd) {
      ++SBegin;
      return *this;
    }

    if (PBegin != PEnd) {
      ++PBegin;
      return *this;
    }

    revng_abort("Attempted to increment an end iterator!");
  }

  decltype(auto) operator*() {
    if (SBegin != SEnd)
      return *SBegin;

    if (PBegin != PEnd)
      return *PBegin;

    revng_abort("Attempted to dereference an end iterator!");
  }
  decltype(auto) operator*() const {
    if (SBegin != SEnd)
      return *SBegin;

    if (PBegin != PEnd)
      return *PBegin;

    revng_abort("Attempted to dereference an end iterator!");
  }

  struct Proxy {
    Proxy(ValueType Value) : Temporary(std::move(Value)) {}
    ValueType *const operator->() { return &Temporary; }
    ValueType const *const operator->() const { return &Temporary; }

  private:
    ValueType Temporary;
  };

  Proxy operator->() { return operator*(); }
  Proxy operator->() const { return operator*(); }

  bool operator==(const UndirectedChildIterator &Another) const {
    auto This = std::tie(SBegin, SEnd, PBegin, PEnd);
    auto That = std::tie(Another.SBegin,
                         Another.SEnd,
                         Another.PBegin,
                         Another.PEnd);
    return This == That;
  }
};
