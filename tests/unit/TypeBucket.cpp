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
    auto [Type, Path] = Bucket.makeStructDefinition();
    Bucket.commit();
  }

  revng_check(Binary->TypeDefinitions().size() == 1);
}

BOOST_AUTO_TEST_CASE(Drop) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeStructDefinition();
    Bucket.drop();
  }

  revng_check(Binary->TypeDefinitions().size() == 0);
}

BOOST_AUTO_TEST_CASE(Multiple) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeStructDefinition();
    Type.OriginalName() = "First";
    Bucket.commit();
  }

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeStructDefinition();
    Type.OriginalName() = "Second";
    Bucket.commit();
  }

  revng_check(Binary->TypeDefinitions().size() == 2);
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check((*FirstIterator)->OriginalName() == "First");
  revng_check((*std::next(FirstIterator))->OriginalName() == "Second");
}

BOOST_AUTO_TEST_CASE(Reused) {
  TupleTree<model::Binary> Binary;

  model::TypeBucket Bucket = *Binary;
  auto [Type, Path] = Bucket.makeStructDefinition();
  Type.OriginalName() = "First";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->OriginalName() == "First");

  auto [AnotherType, AP] = Bucket.makeStructDefinition();
  AnotherType.OriginalName() = "Second";
  Bucket.drop();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->OriginalName() == "First");

  auto [OneMoreType, _] = Bucket.makeStructDefinition();
  OneMoreType.OriginalName() = "Third";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 2);
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check((*FirstIterator)->OriginalName() == "First");
  revng_check((*std::next(FirstIterator))->OriginalName() == "Third");
}

BOOST_AUTO_TEST_CASE(Paths) {
  TupleTree<model::Binary> Binary;

  model::UpcastableType Saved;
  {
    model::TypeBucket Bucket = *Binary;
    auto [Definition, Type] = Bucket.makeStructDefinition();
    Definition.OriginalName() = "Valid";

    revng_check(Binary->TypeDefinitions().size() == 0);
    // Not valid yet.
    // revng_check(Type->tryGetAsDefinition()->OriginalName() == "Valid");
    Bucket.commit();
    revng_check(Binary->TypeDefinitions().size() == 1);
    // Becomes valid after the commit.
    revng_check(Type->tryGetAsDefinition()->OriginalName() == "Valid");

    Saved = Type.copy();
    revng_check(Saved->tryGetAsDefinition()->OriginalName() == "Valid");

    //`Type` gets destroyed here,
  }

  // But `Saved` is still valid.
  revng_check(Binary->TypeDefinitions().size() == 1);
  auto TD = Binary->TypeDefinitions().begin();
  model::TypeDefinition &SavedDef = *Saved->tryGetAsDefinition();
  revng_check(SavedDef.OriginalName() == (*TD)->OriginalName());
  revng_check(SavedDef.OriginalName() == "Valid");
}
