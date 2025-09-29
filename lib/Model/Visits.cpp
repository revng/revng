/// \file Visits.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/TupleTree/VisitsImpl.h"

//
// To optimize compile times, we explicitly instantiate the template
// specializations of the relevant types, here, by hand.
//

template const model::TypeDefinition *
getByPath<const model::TypeDefinition,
          const model::Binary>(const TupleTreePath &Path,
                               const model::Binary &M);

template const model::TypeDefinition *
getByPath<model::TypeDefinition, const model::Binary>(const TupleTreePath &Path,
                                                      const model::Binary &M);

template model::TypeDefinition *
getByPath<model::TypeDefinition, model::Binary>(const TupleTreePath &Path,
                                                model::Binary &M);

template const model::BinaryIdentifier *
getByPath<const model::BinaryIdentifier,
          const model::Binary>(const TupleTreePath &Path,
                               const model::Binary &M);

template const model::BinaryIdentifier *
getByPath<model::BinaryIdentifier, const model::Binary>(const TupleTreePath
                                                          &Path,
                                                        const model::Binary &M);

template model::BinaryIdentifier *
getByPath<model::BinaryIdentifier, model::Binary>(const TupleTreePath &Path,
                                                  model::Binary &M);
