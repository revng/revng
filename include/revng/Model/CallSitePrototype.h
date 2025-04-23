#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/CommonFunctionMethods.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng/Model/Generated/Early/CallSitePrototype.h"

class model::CallSitePrototype
  : public model::generated::CallSitePrototype,
    public model::CommonFunctionMethods<CallSitePrototype> {
public:
  using generated::CallSitePrototype::CallSitePrototype;
  CallSitePrototype(MetaAddress CallerAddress,
                    model::UpcastableType &&Prototype,
                    bool IsTailCall) :
    generated::CallSitePrototype(CallerAddress,
                                 std::move(Prototype),
                                 IsTailCall,
                                 {}) {}

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  bool isDirect() const { return Prototype().isEmpty(); }
};

#include "revng/Model/Generated/Late/CallSitePrototype.h"
