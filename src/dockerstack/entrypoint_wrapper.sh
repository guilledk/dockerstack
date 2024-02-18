#!/bin/sh

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <username> <command>" >&2
    exit 1
fi

USERNAME="$1"
shift  # remove the first argument which is the username

# combine remaining arguments into a single command
COMMAND="$*"
STARTED=0

echo "Entrypoint Wrapper, \"exec su $USERNAME -c $SHELL -c \\\"$COMMAND\\\"\" $DS_LOG_FILE"

execute_command_as_user() {
    echo "Signal received, executing the command as user $USERNAME..."
    if [ -n "$DS_LOG_FILE" ]; then
        # If DS_LOG_FILE is set, redirect output to the file specified
        echo "Redirecting output to $DS_LOG_FILE"
        exec su "$USERNAME" -c "$SHELL -c \"$COMMAND >> /logs/"$DS_LOG_FILE" 2>&1\""
    else
        exec su "$USERNAME" -c "$SHELL -c \"$COMMAND\""
    fi
    STARTED=1
}

trap 'execute_command_as_user' USR2

echo "Waiting for SIGUSR2 signal..."

while true; do
    sleep 1
    if [ $STARTED -eq 1 ]; then
        echo "Process started!"
        break
    fi
done
