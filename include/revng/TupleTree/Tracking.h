#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <tuple>

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/TrackingContainer.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleLikeTraits.h"
#include "revng/TupleTree/TupleTreeCompatible.h"
#include "revng/TupleTree/TupleTreePath.h"
#include "revng/TupleTree/Visits.h"

/// Struct returned by Tracking::collect.
/// The field Read is the set of paths of fields that were accessed.
/// The exact vectors contains the paths of all vectors that were marked as
/// requiring being identical.
struct ReadFields {
  std::set<TupleTreePath> Read;
  std::set<TupleTreePath> ExactVectors;
};

namespace revng {

struct Tracking {

private:
  struct PopVisitor {
    template<revng::SetOrKOC Type>
    static void visitKeyedObjectContainer(const Type &CurrentItem) {
      CurrentItem.trackingPop();
    }

    template<typename Type, size_t FieldIndex>
    static void visitTupleElement(const Type &CurrentItem) {
      CurrentItem.template getTracker<FieldIndex>().pop();
    }
  };

  struct PushVisitor {
    template<revng::SetOrKOC Type>
    static void visitKeyedObjectContainer(const Type &CurrentItem) {
      CurrentItem.trackingPush();
    }

    template<typename Type, size_t FieldIndex>
    static void visitTupleElement(const Type &CurrentItem) {
      CurrentItem.template getTracker<FieldIndex>().push();
    }
  };

  struct ClearVisitor {
    template<revng::SetOrKOC Type>
    static void visitKeyedObjectContainer(const Type &CurrentItem) {
      CurrentItem.clearTracking();
    }

    template<typename Type, size_t FieldIndex>
    static void visitTupleElement(const Type &CurrentItem) {
      CurrentItem.template getTracker<FieldIndex>().clear();
    }
  };

private:
  template<typename M, size_t I = 0, typename T>
  static void
  collectTuple(const T &LHS, TupleTreePath &Stack, ReadFields &Info) {
    if constexpr (I < std::tuple_size_v<T>) {

      Stack.push_back(size_t(I));
      if (LHS.template getTracker<I>().isSet())
        Info.Read.insert(Stack);
      collectImpl<M>(LHS.template untrackedGet<I>(), Stack, Info);
      Stack.pop_back();

      // Recur
      collectTuple<M, I + 1>(LHS, Stack, Info);
    }
  }

  template<typename M, StrictSpecializationOf<UpcastablePointer> T>
  static void
  collectImpl(const T &LHS, TupleTreePath &Stack, ReadFields &Info) {
    LHS.upcast([&](auto &Upcasted) { collectImpl<M>(Upcasted, Stack, Info); });
  }

  template<typename M, TupleSizeCompatible T>
  static void
  collectImpl(const T &LHS, TupleTreePath &Stack, ReadFields &Info) {
    collectTuple<M>(LHS, Stack, Info);
  }

  template<typename M, revng::SetOrKOC T>
  static void
  collectImpl(const T &LHS, TupleTreePath &Stack, ReadFields &Info) {
    typename T::TrackingResult TrackingResult = LHS.getTrackingResult();
    if (TrackingResult.Exact)
      Info.ExactVectors.insert(Stack);

    for (auto &Key : TrackingResult.InspectedKeys) {
      Stack.push_back(Key);
      Info.Read.insert(Stack);
      Stack.pop_back();
    }
    for (auto &LHSElement : LHS.Content) {
      using value_type = typename T::value_type;

      if constexpr (TupleSizeCompatible<value_type>)
        Stack.push_back(LHSElement.untrackedKey());
      else
        Stack.push_back(KeyedObjectTraits<value_type>::key(LHSElement));

      collectImpl<M>(LHSElement, Stack, Info);
      Stack.pop_back();
    }
  }

  template<typename M, NotTupleTreeCompatible T>
  static void
  collectImpl(const T &LHS, TupleTreePath &Stack, ReadFields &Info) {}

private:
  template<typename M, typename Visitor, size_t I = 0, typename T>
  static void visitTuple(const T &LHS) {
    if constexpr (I < std::tuple_size_v<T>) {

      visitImpl<M, Visitor>(LHS.template untrackedGet<I>());
      Visitor::template visitTupleElement<T, I>(LHS);

      // Recur
      visitTuple<M, Visitor, I + 1, T>(LHS);
    }
  }

  template<typename M,
           typename Visitor,
           StrictSpecializationOf<UpcastablePointer> T>
  static void visitImpl(T &LHS) {
    LHS.upcast([&](const auto &Upcasted) { visitImpl<M, Visitor>(Upcasted); });
  }

  template<typename M, typename Visitor, TupleSizeCompatible T>
  static void visitImpl(const T &LHS) {
    visitTuple<M, Visitor>(LHS);
  }

  template<typename M, typename Visitor, revng::SetOrKOC T>
  static void visitImpl(const T &LHS) {
    for (auto &LHSElement : LHS.Content) {
      visitImpl<M, Visitor>(LHSElement);
    }
    Visitor::template visitKeyedObjectContainer<T>(LHS);
  }

  template<typename M, typename Visitor, NotTupleTreeCompatible T>
  static void visitImpl(const T &LHS) {}

public:
  template<typename M>
  static ReadFields collect(const M &LHS) {
    TupleTreePath Stack;
    ReadFields Info;
    collectImpl<M>(LHS, Stack, Info);
    return Info;
  }

  template<typename M>
  static void clear(const M &LHS) {
    visitTuple<M, ClearVisitor>(LHS);
  }

  template<typename M>
  static void push(const M &LHS) {
    visitTuple<M, PushVisitor>(LHS);
  }

  template<typename M>
  static void pop(const M &LHS) {
    visitTuple<M, PopVisitor>(LHS);
  }
};

} // namespace revng
