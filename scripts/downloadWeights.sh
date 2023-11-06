# downloadWeights.sh

#TODO ADD URLS
FILE=$1
DIR=./models/saves

if [ "$FILE" == "unet_16" ]; then 
    URL="https://huggingface.co/Lingram/DocuSegment-Pytorch/resolve/main/unet_16.pth"
    wget $URL -P $DIR

elif [ "$FILE" == "unet_32" ]; then
    URL="https://huggingface.co/Lingram/DocuSegment-Pytorch/resolve/main/unet_32.pth"
    wget $URL -P $DIR

else
    echo "Options: unet_16, unet_32"
    exit 1
fi 



