/**
 * Copyright 2013-2016 Rene Widera, Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "simulation_defines.hpp"


namespace picongpu
{
namespace particles
{
namespace manipulators
{
namespace detail
{
template<typename T_Functor>
struct FreeImpl
{

    typedef T_Functor Functor;

    HINLINE FreeImpl() = default;

    template<typename T_Particle1, typename T_Particle2, typename T_Acc>
    DINLINE void operator()(const T_Acc& acc,
                            T_Particle1& particleSpecies1, T_Particle2& particleSpecies2,
                            const bool isParticle1, const bool isParticle2)
    {
        if (isParticle1 && isParticle2)
        {
            Functor userFunctor;
            userFunctor(acc, particleSpecies1, particleSpecies2);
        }
    }

};

} //namespace detail

template<typename T_Functor>
struct FreeImpl
{
    template<typename T_SpeciesType>
    struct apply
    {
        typedef FreeImpl< T_Functor > type;
    };

    HINLINE FreeImpl(const uint32_t)
    {
    }

    template<typename T_Acc>
    struct Get
    {
        typedef detail::FreeImpl<T_Functor> type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& localCellIdx) const
    {
        typedef typename Get<T_Acc>::type Functor;

        return Functor();
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
