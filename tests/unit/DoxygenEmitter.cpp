//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE DoxygenEmitter
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/CDoxygenEmitter.h"

BOOST_AUTO_TEST_CASE(LineComment) {
  std::string Content;
  {
    llvm::raw_string_ostream Out(Content);
    ptml::CTokenEmitter C(Out, ptml::Tagging::Disabled);
    auto Doxygen = ptml::emitDoxygenLineComment(C);

    Doxygen.emitKeyword("brief");
    Doxygen.emit(" This does something.");
    Doxygen.emit("\n");
    Doxygen.emit("\n\n");
    Doxygen.emit("And also note this other thing.");
  }
  constexpr auto &Expected = "/// \\brief This does something.\n"
                             "///\n"
                             "///\n"
                             "/// And also note this other thing.\n";
  revng_assert(Content == Expected);
}

BOOST_AUTO_TEST_CASE(BlockComment) {
  std::string Content;
  {
    llvm::raw_string_ostream Out(Content);
    ptml::CTokenEmitter C(Out, ptml::Tagging::Disabled);
    auto Doxygen = ptml::emitDoxygenBlockComment(C);

    Doxygen.emitKeyword("brief");
    Doxygen.emit(" This does something.");
    Doxygen.emit("\n");
    Doxygen.emit("\n\n");
    Doxygen.emit("And also note this other thing.");
  }
  constexpr auto Expected = "/**\n"
                            " * \\brief This does something.\n"
                            " *\n"
                            " *\n"
                            " * And also note this other thing.\n"
                            " */";
  revng_assert(Content == Expected);
}
