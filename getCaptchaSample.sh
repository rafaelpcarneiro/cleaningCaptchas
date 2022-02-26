#!/bin/bash
fileName=`date "+%Y%m%d%H%M%S"`

wget $1 -O "$fileName"".jpg"
