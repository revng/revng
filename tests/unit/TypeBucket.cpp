/// \file TypeBucket.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
    auto [Type, Path] = Bucket.makeType<model::StructType>();
    Bucket.commit();
  }

  revng_check(Binary->Types().size() == 1);
}

BOOST_AUTO_TEST_CASE(Drop) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeType<model::StructType>();
    Bucket.drop();
  }

  revng_check(Binary->Types().size() == 0);
}

BOOST_AUTO_TEST_CASE(Multiple) {
  TupleTree<model::Binary> Binary;

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeType<model::StructType>();
    Type.OriginalName() = "First";
    Bucket.commit();
  }

  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeType<model::StructType>();
    Type.OriginalName() = "Second";
    Bucket.commit();
  }

  revng_check(Binary->Types().size() == 2);
  auto FirstIterator = Binary->Types().begin();
  revng_check((*FirstIterator)->OriginalName() == "First");
  revng_check((*std::next(FirstIterator))->OriginalName() == "Second");
}

BOOST_AUTO_TEST_CASE(Reused) {
  TupleTree<model::Binary> Binary;

  model::TypeBucket Bucket = *Binary;
  auto [Type, Path] = Bucket.makeType<model::StructType>();
  Type.OriginalName() = "First";
  Bucket.commit();

  revng_check(Binary->Types().size() == 1);
  revng_check((*Binary->Types().begin())->OriginalName() == "First");

  auto [AnotherType, AnotherPath] = Bucket.makeType<model::StructType>();
  AnotherType.OriginalName() = "Second";
  Bucket.drop();

  revng_check(Binary->Types().size() == 1);
  revng_check((*Binary->Types().begin())->OriginalName() == "First");

  auto [OneMoreType, OneMorePath] = Bucket.makeType<model::StructType>();
  OneMoreType.OriginalName() = "Third";
  Bucket.commit();

  revng_check(Binary->Types().size() == 2);
  auto FirstIterator = Binary->Types().begin();
  revng_check((*FirstIterator)->OriginalName() == "First");
  revng_check((*std::next(FirstIterator))->OriginalName() == "Third");
}

BOOST_AUTO_TEST_CASE(Paths) {
  TupleTree<model::Binary> Binary;

  model::TypePath Saved;
  {
    model::TypeBucket Bucket = *Binary;
    auto [Type, Path] = Bucket.makeType<model::StructType>();
    Type.OriginalName() = "Valid";

    revng_check(Binary->Types().size() == 0);
    // Not valid yet.
    // revng_check(Path.get()->OriginalName() == "Valid");
    Bucket.commit();
    revng_check(Binary->Types().size() == 1);
    // Becomes valid after the commit.
    revng_check(Path.get()->OriginalName() == "Valid");

    Saved = Path;
    revng_check(Saved.get()->OriginalName() == "Valid");
  }

  // Still valid.
  revng_check(Binary->Types().size() == 1);
  revng_check(Saved.get()->OriginalName() == "Valid");
  auto FirstIterator = Binary->Types().begin();
  revng_check(Saved.get()->OriginalName() == (*FirstIterator)->OriginalName());
}
