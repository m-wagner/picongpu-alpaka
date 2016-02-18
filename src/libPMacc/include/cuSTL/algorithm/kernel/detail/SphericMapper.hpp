/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "math/vector/Size_t.hpp"
#include "types.h"

#include <boost/mpl/void.hpp>

namespace PMacc
{
namespace algorithm
{
namespace kernel
{
namespace detail
{

namespace mpl = boost::mpl;

/** The SphericMapper maps from cuda blockIdx and/or threadIdx to the cell index
 * \tparam dim dimension
 * \tparam BlockSize compile-time vector of the cuda block size (optional)
 * \tparam dummy neccesary to implement the optional BlockSize parameter
 *
 * If BlockSize is given the cuda variable blockDim is not used which is faster.
 */
template<int dim, typename BlockSize = mpl::void_, typename dummy = mpl::void_>
struct SphericMapper;

/* Compile-time BlockSize */

template<typename BlockSize>
struct SphericMapper<1, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 1;

    using IndexVec = ::alpaka::Vec<
        alpaka::Dim<dim>,
        alpaka::IdxSize
    >;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<dim>(size.x() / BlockSize::x::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                              const math::Int<dim>& _threadIdx) const
    {
        return _blockIdx.x() * BlockSize::x::value + _threadIdx.x();
    }

    HDINLINE
    math::Int<dim> operator()(const IndexVec& _blockIdx, const IndexVec& _threadIdx = IndexVec(0)) const
    {
        return operator()(math::Int<dim>(_blockIdx[0]),
                          math::Int<dim>(_threadIdx[0]));
    }
};

template<typename BlockSize>
struct SphericMapper<2, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 2;

    using IndexVec = ::alpaka::Vec<
        alpaka::Dim<dim>,
        alpaka::IdxSize
    >;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<dim>(size.x() / BlockSize::x::value,
                              size.y() / BlockSize::y::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                              const math::Int<dim>& _threadIdx) const
    {
        return math::Int<dim>( _blockIdx.x() * BlockSize::x::value + _threadIdx.x(),
                               _blockIdx.y() * BlockSize::y::value + _threadIdx.y() );
    }

    HDINLINE
    math::Int<dim> operator()(const IndexVec& _blockIdx, const IndexVec& _threadIdx = IndexVec(0,0)) const
    {
        // alpaka index y == 0 and x == 1
        return operator()(math::Int<dim>(_blockIdx[1], _blockIdx[0]),
                          math::Int<dim>(_threadIdx[1], _threadIdx[0]));
    }
};

template<typename BlockSize>
struct SphericMapper<3, BlockSize>
{
    BOOST_STATIC_CONSTEXPR int dim = 3;

    using IndexVec = ::alpaka::Vec<
        alpaka::Dim<dim>,
        alpaka::IdxSize
    >;

    DataSpace<dim> gridDim(const math::Size_t<dim>& size) const
    {
        return DataSpace<dim>(size.x() / BlockSize::x::value,
                            size.y() / BlockSize::y::value,
                            size.z() / BlockSize::z::value);
    }

    HDINLINE
    math::Int<dim> operator()(const math::Int<dim>& _blockIdx,
                             const math::Int<dim>& _threadIdx) const
    {
        return math::Int<dim>( _blockIdx * (math::Int<dim>)BlockSize().toRT() + _threadIdx );
    }


    HDINLINE
    math::Int<dim> operator()(const IndexVec& _blockIdx, const IndexVec& _threadIdx = IndexVec(0,0,0)) const
    {
        // alpaka index z == 0 , y == 1 and x ==2
        return operator()(math::Int<dim>(_blockIdx[2], _blockIdx[1], _blockIdx[0]),
                          math::Int<dim>(_threadIdx[2], _threadIdx[1], _threadIdx[0]));
    }
};

} // detail
} // kernel
} // algorithm
} // PMacc
