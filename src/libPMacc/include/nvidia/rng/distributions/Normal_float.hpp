/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

#include "pmacc_types.hpp"

namespace PMacc
{
    namespace nvidia
    {
        namespace rng
        {
            namespace distributions
            {

                /*Return normally distributed floats with mean 0.0f and standard deviation 1.0f
                 */
                template< typename T_Acc>
                class Normal_float
                {
                public:
                    typedef float Type;
                private:
                    using Dist =
                        decltype(
                            ::alpaka::rand::distribution::createNormalReal<Type>(
                                std::declval<T_Acc const &>()));
                    PMACC_ALIGN(dist, Dist);
                public:
                    HDINLINE Normal_float()
                    {
                    }

                    HDINLINE Normal_float(const T_Acc& acc) : dist(::alpaka::rand::distribution::createNormalReal<Type>(acc))
                    {
                    }

                    template<class RNGState>
                    DINLINE Type operator()(RNGState& state)
                    {
                        return dist(state);
                    }

                };
            }
        }
    }
}
