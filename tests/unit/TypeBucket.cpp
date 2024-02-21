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
    auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
    Bucket.commit();
  }

  revng_check(Binary->TypeDefinitions().size() == 1);
}

BOOST_AUTO_TEST_CASE(Drop) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
    Bucket.drop();
  }

  revng_check(Binary->TypeDefinitions().size() == 0);
}

BOOST_AUTO_TEST_CASE(Multiple) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
    Type.OriginalName() = "First";
    Bucket.commit();
  }

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
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
  auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
  Type.OriginalName() = "First";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->OriginalName() == "First");

  auto [AnotherType, AP] = Bucket.makeTypeDefinition<model::StructDefinition>();
  AnotherType.OriginalName() = "Second";
  Bucket.drop();

  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check((*Binary->TypeDefinitions().begin())->OriginalName() == "First");

  auto [OneMoreType, _] = Bucket.makeTypeDefinition<model::StructDefinition>();
  OneMoreType.OriginalName() = "Third";
  Bucket.commit();

  revng_check(Binary->TypeDefinitions().size() == 2);
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check((*FirstIterator)->OriginalName() == "First");
  revng_check((*std::next(FirstIterator))->OriginalName() == "Third");
}

BOOST_AUTO_TEST_CASE(Paths) {
  TupleTree<model::Binary> Binary;

  model::DefinitionReference Saved;
  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeTypeDefinition<model::StructDefinition>();
    Type.OriginalName() = "Valid";

    revng_check(Binary->TypeDefinitions().size() == 0);
    // Not valid yet.
    // revng_check(Path.get()->OriginalName() == "Valid");
    Bucket.commit();
    revng_check(Binary->TypeDefinitions().size() == 1);
    // Becomes valid after the commit.
    revng_check(Path.get()->OriginalName() == "Valid");

    Saved = Path;
    revng_check(Saved.get()->OriginalName() == "Valid");
  }

  // Still valid.
  revng_check(Binary->TypeDefinitions().size() == 1);
  revng_check(Saved.get()->OriginalName() == "Valid");
  auto FirstIterator = Binary->TypeDefinitions().begin();
  revng_check(Saved.get()->OriginalName() == (*FirstIterator)->OriginalName());
}
