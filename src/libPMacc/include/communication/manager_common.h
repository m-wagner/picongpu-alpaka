/**
 * Copyright 2013-2016 Rene Widera, Wolfgang Hoenig, Axel Huebl
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

const int GridManagerRank = 0;

enum {
  gridInitTag = 1,
  gridHostnameTag = 2,
  gridHostRankTag = 3,
  gridExitTag = 4,
  gridExchangeTag = 5
};

#define MPI_CHECK(cmd) {int error = cmd; if(error!=MPI_SUCCESS){std::cerr << "<" << __FILE__ << ">:" << __LINE__; throw std::runtime_error(std::string("[MPI] Error"));}}
#define MPI_CHECK_BLOCKING(cmd) PMacc::Environment<>::get().TransactionManager().getTransactionEvent().waitForFinished(); MPI_CHECK(cmd)
