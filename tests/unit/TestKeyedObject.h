#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/Model/TupleTree.h"

struct Element {
  uint64_t Key;
  uint64_t Value;

  Element(uint64_t Key) : Key(Key), Value(0) {}
  Element(uint64_t Key, uint64_t Value) : Key(Key), Value(Value) {}

  bool operator==(const Element &Other) const = default;

  uint64_t key() const { return Key; }
  uint64_t value() const { return Value; }
  void setValue(uint64_t NewValue) { Value = NewValue; }
};

template<>
struct KeyedObjectTraits<Element> {
  static uint64_t key(const Element &SE) { return SE.key(); }
  static Element fromKey(uint64_t Key) { return Element(Key); }
};

INTROSPECTION(Element, Key, Value);

template<>
struct llvm::yaml::MappingTraits<Element>
  : public TupleLikeMappingTraits<Element> {};

static_assert(HasKeyObjectTraits<Element>);
static_assert(not IsKeyedObjectContainer<std::vector<std::pair<int, int>>>);
