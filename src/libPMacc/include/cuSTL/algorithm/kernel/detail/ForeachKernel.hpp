/**
 * Copyright 2013-2016 Heiko Burau
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

namespace PMacc
{
namespace algorithm
{
namespace kernel
{

namespace detail
{

    struct kernelForeach
    {
        static constexpr uint32_t kernelDim = DIM3;
        //-----------------------------------------------------------------------------
        //! The kernel.
        //-----------------------------------------------------------------------------
        template<
            typename T_Acc,
            typename T_Mapper,
            typename T_Functor,
            typename... T_Type>
        ALPAKA_FN_ACC void operator()(
            const T_Acc& acc,
            const T_Mapper& mapper,
            const T_Functor& functor,
            T_Type ... ts) const
        {
            math::Int<T_Mapper::dim> cellIndex(
                mapper(
                    ::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc),
                    ::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc)));

            functor(forward(ts[cellIndex])...);
        }
    };

} // namespace detail
} // namespace kernel
} // namespace algorithm
} // namespace PMacc
