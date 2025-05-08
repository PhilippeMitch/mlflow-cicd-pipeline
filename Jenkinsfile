pipeline {
    agent any
    environment {
        MLFLOW_TRACKING_URI = 'https://lmitch-mlops.duckdns.org/mlflow'
        SLACK_CHANNEL = '#mlflow-cicd'
        BACKUP_DIR = 'kubernetes/backups'
        REDIS_HOST = 'redis'
        REDIS_PASSWORD = credentials('redis-password')
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/PhilippeMitch/mlflow-cicd-pipeline.git'
            }
        }
        stage('Train Model') {
            steps {
                sh 'python scripts/train_model.py'
            }
        }
        stage('Check Model Metadata') {
            steps {
                sh '''
                    python -c "from mlflow.tracking import MlflowClient; \\
                    client = MlflowClient(); \\
                    versions = client.search_model_versions(\\"name='adult-classifier'\\"); \\
                    for v in versions: \\
                        print(f'Version: {v.version}, Created by: {v.tags.get(\\'created_by\\')}, \\
                        Stage: {v.current_stage}, Description: {v.description}')"
                '''
            }
        }
        stage('Test Model') {
            steps {
                script {
                    sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pytest tests/test_model.py
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