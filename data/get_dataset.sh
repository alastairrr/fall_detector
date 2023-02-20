#!bin/bash

echo Downloading from http://fenix.univ.rzeszow.pl/~mkepski/ds

if [ ! -d "dataset" ]; then
    mkdir dataset
    cd dataset
    mkdir fall not_fall
    cd fall

else
    cd dataset
    cd fall
fi

echo -e "\nGetting fall dataset...\n"

for i in {01..30}
do
    file_link="http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-${i}-cam0-rgb.zip"
    echo Now attempting to retrieve ${file_link}
    wget ${file_link}
    unzip "fall-${i}-cam0-rgb.zip"
    cd "fall-${i}-cam0-rgb"
    wget "http://fenix.univ.rzeszow.pl/~mkepski/ds/data/fall-${i}-data.csv"
    cd ../
    rm "fall-${i}-cam0-rgb.zip"
done

cd ../not_fall

echo -e "\nGetting not fall dataset...\n"

for i in {01..40}
do
    file_link="http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-${i}-cam0-rgb.zip"
    echo Now attempting to retrieve ${file_link}
    wget ${file_link}
    unzip "adl-${i}-cam0-rgb.zip"
    cd "adl-${i}-cam0-rgb"
    wget "http://fenix.univ.rzeszow.pl/~mkepski/ds/data/adl-${i}-data.csv"
    cd ../
    rm "adl-${i}-cam0-rgb.zip"
done

echo -e "\nDownload complete!\n"