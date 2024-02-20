/// \file Visits.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/TupleTree/VisitsImpl.h"

template const model::TypeDefinition *
getByPath<const model::TypeDefinition,
          const model::Binary>(const TupleTreePath &Path,
                               const model::Binary &M);

template model::TypeDefinition *
getByPath<model::TypeDefinition, const model::Binary>(const TupleTreePath &Path,
                                                      const model::Binary &M);

template model::TypeDefinition *
getByPath<model::TypeDefinition, model::Binary>(const TupleTreePath &Path,
                                                model::Binary &M);
