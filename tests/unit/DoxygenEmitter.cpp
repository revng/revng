//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#define BOOST_TEST_MODULE DoxygenEmitter
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/CDoxygenEmitter.h"

BOOST_AUTO_TEST_CASE(LineComment) {
  std::string Emitted;
  {
    llvm::raw_string_ostream Out(Emitted);
    ptml::CTokenEmitter C(Out, ptml::Tagging::Disabled);
    auto Doxygen = ptml::CDoxygenEmitter::emitLineComment(C);

    Doxygen.emitKeyword("brief");
    Doxygen.emit(" This does something.");
    Doxygen.emit("\n");
    Doxygen.emit("\n\n");
    Doxygen.emit("And also note this other thing.");
  }
  constexpr llvm::StringRef Expected = "/// \\brief This does something.\n"
                                       "///\n"
                                       "///\n"
                                       "/// And also note this other thing.\n";

  if (Emitted != Expected) {
    std::string Error = "Emitted string does *not* match the expected one!\n"
                        "> Emitted:\n"
                        + Emitted + "\n> Expected:\n" + Expected.str() + '\n';
    revng_abort(Error.c_str());
  }
}

BOOST_AUTO_TEST_CASE(BlockComment) {
  std::string Emitted;
  {
    llvm::raw_string_ostream Out(Emitted);
    ptml::CTokenEmitter C(Out, ptml::Tagging::Disabled);
    auto Doxygen = ptml::CDoxygenEmitter::emitBlockComment(C);

    Doxygen.emitKeyword("brief");
    Doxygen.emit(" This does something.");
    Doxygen.emit("\n");
    Doxygen.emit("\n\n");
    Doxygen.emit("And also note this other thing.");
  }
  constexpr llvm::StringRef Expected = "/**\n"
                                       " * \\brief This does something.\n"
                                       " *\n"
                                       " *\n"
                                       " * And also note this other thing.\n"
                                       " */";

  if (Emitted != Expected) {
    std::string Error = "Emitted string does *not* match the expected one!\n"
                        "> Emitted:\n"
                        + Emitted + "\n> Expected:\n" + Expected.str() + '\n';
    revng_abort(Error.c_str());
  }
}
