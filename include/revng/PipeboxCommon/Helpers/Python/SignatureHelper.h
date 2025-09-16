#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "nanobind/nanobind.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"

namespace revng::pypeline::helpers::python {

template<typename T>
  requires IsPipe<T> or IsAnalysis<T>
class SignatureHelper {
public:
  /// Function that gets the signature of a given Pipe or Analysis type. Since
  /// the signature might be requested many times, the result is cached in the
  /// `_signature` attribute inside the class.
  static nanobind::object getSignature() {
    // Get the PyObject of the specified class T.
    // We need to `borrow` it since we want to increase the refcount for the
    // duration of this function.
    nanobind::object Class = nanobind::borrow(nanobind::type<T>());
    // Check that it's actually not NULL (e.g. not registered through nanobind)
    revng_assert(Class.is_valid());
    // Check if we previously computed the signature, if so return that
    nanobind::object Signature = nanobind::getattr(Class,
                                                   "_signature",
                                                   nanobind::none());

    if (not Signature.is_none())
      return Signature;

    // If here we've yet to compute the signature, do that and save it as an
    // attribute inside the class
    if constexpr (IsAnalysis<T>)
      Signature = parseAnalysisSignature();
    else
      Signature = parsePipeSignature();
    nanobind::setattr(Class, "_signature", Signature);
    return Signature;
  }

private:
  /// Given a pipe type T, compute its signature
  static nanobind::object parsePipeSignature() {
    nanobind::object TaskArgument = importObject("revng.pypeline.task.task."
                                                 "TaskArgument");
    nanobind::object TaskArgumentAccess = importObject("revng.pypeline.task."
                                                       "task."
                                                       "TaskArgumentAccess");
    // Result list, this will be appended with a TaskArgument object for each
    // argument of the pipe
    nanobind::list Result;

    using CT = PipeRunTraits<T>::ContainerTypes;
    forEach<CT>([&Result,
                 &TaskArgument,
                 &TaskArgumentAccess]<typename A, size_t I>() {
      using Argument = std::tuple_element_t<I,
                                            typename T::ArgumentsDocumentation>;
      // Create a Kwargs dictionary, this will be passed to the constructor of
      // TaskArgument
      nanobind::dict Kwargs;

      Kwargs["name"] = nanobind::str(Argument::Name.data(),
                                     Argument::Name.size());
      Kwargs["help_text"] = nanobind::str(Argument::HelpText.data(),
                                          Argument::HelpText.size());
      // nanobind::type<T> returns a reference, so we need to borrow it and
      // increase the reference count
      Kwargs["container_type"] = nanobind::borrow(nanobind::type<A>());
      // If the argument is const then the access is READ, otherwise READ_WRITE
      if constexpr (std::is_const_v<A>)
        Kwargs["access"] = nanobind::getattr(TaskArgumentAccess, "READ");
      else
        Kwargs["access"] = nanobind::getattr(TaskArgumentAccess, "READ_WRITE");

      // kwargs_proxy is a special nanobind class that allows passing a
      // nanobind::dict as kwargs, this is equivalent to doing
      // `TaskArgument(**Kwargs)`.
      Result.append(TaskArgument(nanobind::detail::kwargs_proxy(Kwargs)));
    });

    // This version of the nanobind::tuple constructor automatically converts
    // the nanobind::list to a tuple.
    return nanobind::tuple(Result);
  }

  /// Given an analysis type T, compute its signature
  static nanobind::object parseAnalysisSignature() {
    nanobind::list Result;

    using CT = AnalysisRunTraits<T>::ContainerTypes;
    forEach<CT>([&Result]<typename A, size_t I>() {
      // For each tuple element, add the python type to the Result list
      Result.append(nanobind::borrow(nanobind::type<A>()));
    });

    return nanobind::tuple(Result);
  }
};

} // namespace revng::pypeline::helpers::python
