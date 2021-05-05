#!/usr/bin/env sh

if test "$#" -ne 4; then
    echo 'Usage: sync_repo.sh <remote_machine_username> <remote_machine_address> <local_fl_dir> <remote_fl_dir>'
    exit 2
fi

USER=$1
MACHINE=$2
IBMFL_DIR_LOC="$3/"
IBMFL_DIR_REM="$4"
rsync -av \
    --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
    --include 'examples/configs' --include 'examples/datasets' --include 'examples/data' --exclude 'examples/*' \
    "$IBMFL_DIR_LOC" "$USER@$MACHINE:$IBMFL_DIR_REM"
