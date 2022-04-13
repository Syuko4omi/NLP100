#!/bin/zsh

expand -t 1 hoge.txt

cat hoge.txt | tr '\t' ' '

cat hoge.txt | sed s/$'\t'/' '/g  
