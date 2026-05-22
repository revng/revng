#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers::python {

template<HasCustomInvalidation T>
inline nanobind::object
processCustomInvalidation(T &Handle,
                          nanobind::list PyInvalidationData,
                          nanobind::handle_t<ModelDiff> Diff) {
  // Convert the invalidation data from python to C++
  InvalidationData Data;
  for (nanobind::handle List : PyInvalidationData) {
    auto &CppList = Data.emplace_back();
    for (nanobind::handle Entry : nanobind::cast<nanobind::list>(List)) {
      nanobind::tuple Tuple = nanobind::cast<nanobind::tuple>(Entry);
      nanobind::bytes Bytes = nanobind::cast<nanobind::bytes>(Tuple[1]);
      CppList.push_back({ nanobind::cast<ObjectID *>(Tuple[0]),
                          { reinterpret_cast<const uint8_t *>(Bytes.data()),
                            Bytes.size() } });
    }
  }

  // Run the actual invalidate method
  ModelDiff *CppDiff = nanobind::cast<ModelDiff *>(Diff);
  std::vector<std::set<ObjectID>>
    CppResult = Handle.processCustomInvalidation(Data, *CppDiff);

  // Convert the result to a list[ObjectSet]
  using Traits = PipeRunTraits<T>;
  nanobind::object ObjectSetCls = importObject("revng.pypeline.object."
                                               "ObjectSet");
  nanobind::list Result;
  for (size_t I = 0; I < CppResult.size(); I++) {
    compile_time::repeat<Traits::ContainerCount>([&]<size_t J>() {
      if (I != J)
        return;

      using CT = std::tuple_element_t<J, typename Traits::ContainerTypes>;
      nanobind::object PyKind = nanobind::cast(CT::Kind);
      nanobind::list ObjectList;
      for (const ObjectID &Object : CppResult[I])
        ObjectList.append(Object);
      nanobind::object ObjectSet = ObjectSetCls(PyKind,
                                                nanobind::set(ObjectList));
      Result.append(ObjectSet);
    });
  }
  return Result;
}

} // namespace revng::pypeline::helpers::python
