#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

template<typename T, typename U>
concept same_as = std::is_same_v<T, U>;
