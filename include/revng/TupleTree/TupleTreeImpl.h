#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
