/**
 * Copyright 2013-2016 Felix Schmitt, Heiko Burau, Rene Widera
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


#include "types.h"
#include <string>
#include <ostream>
#include <math_functions.h>

namespace PMacc
{

template<
    typename T_Acc,
    typename T_Type
>
DINLINE void atomicAddWrapper(const T_Acc& acc, T_Type * const address, const T_Type& value)
{
    ::alpaka::atomic::atomicOp<::alpaka::atomic::op::Add>(acc, address, value);
}

} //namespace PMacc

/* CUDA STD structs and CPP STD ostream */
template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const double3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}

template <class T>
std::basic_ostream<T, std::char_traits<T> >& operator<<(std::basic_ostream<T, std::char_traits<T> >& out, const float3& v)
{
    out << "{" << v.x << " " << v.y << " " << v.z << "}";
    return out;
}


