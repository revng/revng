#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/TupleTree/TupleTree.h"

// Note: the purpose of this file is to make sure we opt-in using
//       visitTupleTree in each translation unit to save on compilation time.

namespace detail {
template<typename T>
using ConstVisitor = typename TupleTreeVisitor<T>::ConstVisitorBase;
}

template<TupleTreeCompatible T>
void TupleTree<T>::visitImpl(detail::ConstVisitor<T> &Pre,
                             detail::ConstVisitor<T> &Post) const {
  visitTupleTree(*Root, Pre, Post);
}

template<TupleTreeCompatible T>
void TupleTree<T>::visitImpl(typename TupleTreeVisitor<T>::VisitorBase &Pre,
                             typename TupleTreeVisitor<T>::VisitorBase &Post) {
  visitTupleTree(*Root, Pre, Post);
}

template<TupleTreeCompatible T>
bool TupleTree<T>::verifyReferences(bool Assert) const {
  TrackGuard Guard(*Root);
  bool Result = true;

  visitReferences([&Result,
                   &Assert,
                   RootPointer = Root.get()](const auto &Element) {
    if (Result) {
      auto Check = [&Assert, &Result](bool Condition) {
        if (not Condition) {
          Result = false;
          if (Assert)
            revng_abort();
        }
      };

      if (not Element.empty()) {
        Check(Element.getRoot() == RootPointer);
        Check(not Element.isConst());
        Check(Element.isValid());
      }
    }
  });

  return Result;
}
