#!bin/zsh

cut -f 1 -d $'\t' popular-names.txt | sort | uniq
