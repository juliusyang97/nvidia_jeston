#!/bin/bash
SLEEPINTERVAL=1
while true;
do
    sysctl -w vm.drop_caches=3
    sleep $SLEEPINTERVAL
done
