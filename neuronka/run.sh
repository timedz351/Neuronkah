##!/bin/bash
## change this file to your needs
echo "checking dotnet version"
dotnet --version

echo "#################"
echo "    COMPILING    "
echo "#################"

## We will use JIT compilation -> tends to be faster

## compile the code -> dotnet native AOT compilation
#dotnet publish src/NNProject/NNProject.csproj --sc -c Release -o ./build
#chmod u+x ./build/NNProject


echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network

#nice -n 19 ./build/NNProject

nice -n 19 dotnet run --project ./neuronka.csproj -c Release
