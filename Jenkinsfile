pipeline {
    agent any
    environment {
        MLFLOW_TRACKING_URI = 'http://localhost:5000'
        SLACK_CHANNEL = '#mlflow-cicd'
        BACKUP_DIR = 'kubernetes/backups'
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/PhilippeMitch/mlflow-cicd-pipeline.git'
            }
        }
        stage('Test Model') {
            steps {
                script {
                    sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pytest tests/test_model.py --mock-mlflow
                    '''
                }
            }
            post {
                failure {
                    slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "Test stage failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                }
            }
        }

        stage('Build and Push Dev') {
            steps {
                sh 'docker login -u $DOCKERHUB_CREDENTIALS_USR -p $DOCKERHUB_CREDENTIALS_PSW'
                sh '''
                    docker build -t $DOCKERHUB_CREDENTIALS_USR/adult-classifier:dev-1.0.0 \
                        --build-arg MODEL_NAME=adult-classifier \
                        --build-arg MODEL_VERSION=1.0.0 -f docker/serve/Dockerfile .
                    docker tag $DOCKERHUB_CREDENTIALS_USR/adult-classifier:dev-1.0.0 \
                        $DOCKERHUB_CREDENTIALS_USR/adult-classifier:dev-latest
                    docker push $DOCKERHUB_CREDENTIALS_USR/adult-classifier:dev-1.0.0
                    docker push $DOCKERHUB_CREDENTIALS_USR/adult-classifier:dev-latest
                '''
            }
        }
    }
    post {
        failure {
            slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "Pipeline failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
        }
    }
}