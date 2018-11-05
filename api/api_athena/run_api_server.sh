#!/bin/bash

if [[ -z $ATHENA_INSTALL ]]
then
    echo "ATHENA_INSTALL not set"
    exit 1
fi

if [[ -z ATHENA_HOME ]] 
then
    echo "ATHENA_HOME not set"
    exit 1
fi

API_SERVER=$ATHENA_INSTALL/api/release/bin/api_server

$API_SERVER $1 $2
