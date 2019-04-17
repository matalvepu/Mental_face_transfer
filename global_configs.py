running_as_job_array = False
conf_prototype=False
conf_url_database = 'bhc0085:27017'

if(conf_prototype == True):
    conf_mongo_database_name = 'prototype'
else:
    conf_mongo_database_name = 'baseline'
#bce_score
#The first run is stored in baseline
#run by:node_modules/.bin/omniboard -m  bhc0085:27017:baseline