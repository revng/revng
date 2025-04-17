/// \file TypeBucket.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE TypeBucket
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/TypeBucket.h"
#include "revng/TupleTree/TupleTree.h"

BOOST_AUTO_TEST_CASE(Commit) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto &&[Type, Path] = Bucket.makeStructDefinition();
    Bucket.commit();
  }

  revng_check(Binary->TypeDefinitions().size() == 1);
}

BOOST_AUTO_TEST_CASE(Drop) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto &&[Type, Path] = Bucket.makeStructDefinition();
    Bucket.drop();
  }

  revng_check(Binary->TypeDefinitions().size() == 0);
}

BOOST_AUTO_TEST_CASE(Multiple) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto &&[Type, Path] = Bucket.makeStructDefinition();
    Type.Name() = "First";
    Bucket.commit();
  }

  {
    model::TypeBucket Bucket = *Binary;
    auto &&[Type, Path] = Bucket.makeStructDefinition();
    Type.Name() = "Second";
    Bucket.commit();
  }

  revng_check(Binary->TypeDefinitions().size() == 2);
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check((*FirstIterator)->Name() == "First");
  revng_check((*std::next(FirstIterator))->Name() == "Second");
}

BOOST_AUTO_TEST_CASE(Reused) {
  TupleTree<model::Binary> Binary;

  model::TypeBucket Bucket = *Binary;
  auto &&[Type, Path] = Bucket.makeStructDefinition();
  Type.Name() = "First";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->Name() == "First");

  auto &&[AnotherType, AP] = Bucket.makeStructDefinition();
  AnotherType.Name() = "Second";
  Bucket.drop();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->Name() == "First");

  auto &&[OneMoreType, _] = Bucket.makeStructDefinition();
  OneMoreType.Name() = "Third";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 2);
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check((*FirstIterator)->Name() == "First");
  revng_check((*std::next(FirstIterator))->Name() == "Third");
}

BOOST_AUTO_TEST_CASE(Paths) {
  TupleTree<model::Binary> Binary;

  model::UpcastableType Saved;
  {
    model::TypeBucket Bucket = *Binary;
    auto &&[Definition, Type] = Bucket.makeStructDefinition();
    Definition.Name() = "Valid";

    revng_check(Binary->TypeDefinitions().size() == 0);
    // Not valid yet.
    // revng_check(Type->tryGetAsDefinition()->Name() == "Valid");
    Bucket.commit();
    revng_check(Binary->TypeDefinitions().size() == 1);
    // Becomes valid after the commit.
    revng_check(Type->tryGetAsDefinition()->Name() == "Valid");

    Saved = Type.copy();
    revng_check(Saved->tryGetAsDefinition()->Name() == "Valid");

    //`Type` gets destroyed here,
  }

  // But `Saved` is still valid.
  revng_check(Binary->TypeDefinitions().size() == 1);
  auto TD = Binary->TypeDefinitions().begin();
  model::TypeDefinition &SavedDef = *Saved->tryGetAsDefinition();
  revng_check(SavedDef.Name() == (*TD)->Name());
  revng_check(SavedDef.Name() == "Valid");
}
