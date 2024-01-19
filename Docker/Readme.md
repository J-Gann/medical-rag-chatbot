# Running Opensearch with Docker  

# Links
- https://docs.docker.com/get-docker/ 
- https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/


# Instruction 

1. Install docker 
2. Pull opensearch images  
    sudo docker pull opensearchproject/opensearch:2
    sudo docker pull opensearchproject/opensearch-dashboards:2
3. Create docker-compose.yml
4. Run docker file 
    sudo docker-compose up -d
5. Verify running docker 
    docker-compose ps
6. Stop docker file 
    docker-compose down
