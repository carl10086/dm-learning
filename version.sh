#!/usr/bin/env bash
version=$(date +%y%m%d.%H%M)
echo $version
mvn versions:set -DnewVersion=$version
mvn versions:commit
