# downloadWeights.sh


FILE=$1
DIR=./models/saves

if [ "$FILE" == "UNet_16" ]; then 
    URL=""
    wget $URL -P $DIR

elif [ "$FILE" == "UNet_32" ]; then
    URL=""
    wget $URL -P $DIR

elif [ "$FILE" == "UNet_64" ]; then 
    URL=""
    wget $URL -P $DIR

else
    echo "Options: UNet_16, UNet_32, UNet_64"
    exit 1
fi 



