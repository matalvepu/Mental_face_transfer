module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /scratch/echowdh2/mongodb_storage/deception --port $port > mongod.log &