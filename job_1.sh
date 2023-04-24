#!/bin/bash
source /opt/intel/oneapi/setvars.sh
cd $PBS_O_WORKDIR

icpx -fsycl bloom_sycl.cpp -o bloom_sycl
./bloom_sycl
