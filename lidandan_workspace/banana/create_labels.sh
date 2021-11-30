#!/bin/bash
jsonfiles=`ls *.json`
for jsonfile in $jsonfiles
do
    labelme_json_to_dataset $jsonfile
    cd ${jsonfile/.json/_json}
    mv label.png ../${jsonfile/.json/.png}
    cd ..
    rm -r ${jsonfile/.json/_json}
    echo ${jsonfile/.json/.png} created!
done

