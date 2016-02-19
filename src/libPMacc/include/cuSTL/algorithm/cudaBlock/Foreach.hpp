/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Axel Huebl
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
#include "algorithms/TypeCast.hpp"
#include "math/vector/Int.hpp"
#include "math/Vector.hpp"
#include "math/VectorOperations.hpp"
#include "forward.hpp"

namespace PMacc
{
namespace algorithm
{
namespace cudaBlock
{

/** Foreach algorithm that is executed by one cuda thread block
 *
 * \tparam BlockDim 3D compile-time vector (PMacc::math::CT::Int) of the size of the cuda blockDim.
 *
 * BlockDim could also be obtained from cuda itself at runtime but
 * it is faster to know it at compile-time.
 */
template<typename BlockDim>
struct Foreach
{
private:
    const int linearThreadIdx;
public:

     DINLINE Foreach(int linearThreadIdx) : linearThreadIdx(linearThreadIdx) {}

    /* operator()(zone, cursor0, cursor1, ..., cursorN-1, functor or lambdaFun)
     *
     * \param zone compile-time zone object, see zone::CT::SphericZone. (e.g. ContainerType::Zone())
     * \param cursorN cursor for the N-th data source (e.g. containerObj.origin())
     * \param functor or lambdaFun either a functor with N arguments or a N-ary lambda function (e.g. _1 = _2)
     *
     * The functor or lambdaFun is called for each cell within the zone.
     * It is called like functor(*cursor0(cellId), ..., *cursorN(cellId))
     *
     */

    template<typename Zone, typename Functor, typename... T_Types>
    DINLINE void operator()(Zone, const Functor& functor, T_Types ... ts)
    {
        BOOST_AUTO(functor_, lambda::make_Functor(functor));
        const int dataVolume = math::CT::volume<typename Zone::Size>::type::value;
        const int blockVolume = math::CT::volume<BlockDim>::type::value;

        typedef typename math::Int<Zone::dim> PosType;
        using namespace PMacc::algorithms::precisionCast;

        for(int i = this->linearThreadIdx; i < dataVolume; i += blockVolume)
        {
            PosType pos = Zone::Offset().toRT() +
                          precisionCast<typename PosType::type>(
                            math::MapToPos<Zone::dim>()( Zone::Size(), i ) );
            functor_( forward(ts[pos]) ...);
        }
    }
};

} // cudaBlock
} // algorithm
} // PMacc
