#!/bin/bash

NAME="FCANN"
EXEC="$({ test -d 'cmake-build-release' && find 'cmake-build-release' -executable -name $NAME || find . -executable -name $NAME; } | head -n 1)"
echo "Using executable ${EXEC}"

datasets=( datasets/* )


# Fully connected networks with 1 hidden layer.
for dataset in "${datasets[@]}"; do
  output="results/full-1hl-$(basename "$dataset" ".txt").txt"
  echo "dataset,neurons,accuracy" >"$output"
  for neurons in {1..20}; do
    for i in {1..10}; do
      echo "Fully connected networks, neurons=${neurons}, iteration=${i}, dataset=${dataset}"
      result=( $(${EXEC} "${dataset}" full $neurons 2>/dev/null) )
      if [ $? -ne 0 ]; then break; fi
      echo "$dataset,${result[0]},${result[1]}" >>"$output"
    done
  done
done


# Fully connected networks with 2 hidden layers of equal sizes.
for dataset in "${datasets[@]}"; do
  output="results/full-2hl-$(basename "$dataset" ".txt").txt"
  echo "dataset,neurons,accuracy" >"$output"
  for neurons in {1..10}; do
    for i in {1..10}; do
      echo "Fully connected networks, neurons=[${neurons}, ${neurons}], iteration=${i}, dataset=${dataset}"
      result=( $(${EXEC} "${dataset}" full $neurons $neurons 2>/dev/null) )
      if [ $? -ne 0 ]; then break; fi
      echo "$dataset,${result[0]},${result[1]}" >>"$output"
    done
  done
done


# FCA-based network with condition min_support.
for dataset in "${datasets[@]}"; do
  output="results/full-min_supp-$(basename "$dataset" ".txt").txt"
  echo "dataset,min_supp,max_level,neurons,accuracy" >"$output"
  for min_supp in $(seq 0.1 0.05 0.9); do
    for max_level in {1..5}; do
      for i in {1..10}; do
        echo "min_support based network, min_supp=${min_supp}, max_level=${max_level}, iteration=${i}, dataset=${dataset}"
        result=( $(${EXEC} "${dataset}" min_supp $min_supp $max_level 2>/dev/null) )
        if [ $? -ne 0 ]; then break; fi
        echo "$dataset,$min_supp,$max_level,${result[0]},${result[1]}" >>"$output"
      done
    done
  done
done


# FCA-based network with condition min_cv.
for dataset in "${datasets[@]}"; do
  output="results/full-min_cv-$(basename "$dataset" ".txt").txt"
  echo "dataset,min_cv,max_level,neurons,accuracy" >"$output"
  for min_cv in $(seq 0.1 0.05 0.9); do
    for max_level in {1..5}; do
      for i in {1..10}; do
        echo "min_cv based network, min_cv=${min_cv}, max_level=${max_level}, iteration=${i}, dataset=${dataset}"
        result=( $(${EXEC} "${dataset}" cv $min_cv $max_level 2>/dev/null) )
        if [ $? -ne 0 ]; then break; fi
        echo "$dataset,$min_cv,$max_level,${result[0]},${result[1]}" >>"$output"
      done
    done
  done
done


# FCA-based network with condition min_cfc.
for dataset in "${datasets[@]}"; do
  output="results/full-min_cfc-$(basename "$dataset" ".txt").txt"
  echo "dataset,min_cfc,max_level,neurons,accuracy" >"$output"
  for min_cfc in $(seq 0.1 0.05 0.9); do
    for max_level in {1..5}; do
      for i in {1..10}; do
        echo "min_cfc based network, min_cfc=${min_cfc}, max_level=${max_level}, iteration=${i}, dataset=${dataset}"
        result=( $(${EXEC} "${dataset}" cfc $min_cfc $max_level 2>/dev/null) )
        if [ $? -ne 0 ]; then break; fi
        echo "$dataset,$min_cfc,$max_level,${result[0]},${result[1]}" >>"$output"
      done
    done
  done
done
