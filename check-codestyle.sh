#!/usr/bin/env bash

fails=()
successes=()

checkmark="✔"
cross="✘"

function check() {
    echo "+ $@"
    "$@" && {
        successes+=("${checkmark} ${1}")
    } || {
        fails+=("${cross} ${1}")
    }
}

check flake8
check mypy --config-file mypy.ini
check black . --check --diff --color

echo
if [ "${#successes[@]}" -gt "0" ]; then
    echo "Successful checks:"
    echo ${successes[@]}
fi
if [ "${#fails[@]}" -gt "0" ]; then
    echo "The following checks failed (please check the output above):"
    echo ${fails[@]}
    exit 1
else
    echo
    echo "All looking good!"
fi
