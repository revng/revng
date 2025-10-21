//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

%b1 = clift.make_label

clift.for break %b1 body {
  clift.break_to %b1
}

%c2 = clift.make_label

clift.for continue %c2 body {
  clift.continue_to %c2
}

%b3 = clift.make_label
%c3 = clift.make_label

clift.for break %b3 continue %c3 body {}

clift.for body {
  clift.goto %b3
}

clift.for body {
  clift.goto %c3
}
