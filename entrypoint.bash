#!/bin/bash

echo "Entry point ---> Executing sshfs "
sshfs -o StrictHostKeyChecking=accept-new,default_permissions,ssh_command='ssh -i /home/drigoni/repository/volta/id_dkm' digis-precision79202.fbk.eu:/raid/home/dkm/drigoni/repository/volta/data ./data


exec "$@"
