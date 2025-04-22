pipeline {
    agent any
    environment {
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
        SLACK_CHANNEL = '#mlflow-cicd'
        BACKUP_DIR = 'kubernetes/backups'
    }
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/USER/REPO.git'
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
        stage('Deploy to Dev') {
            steps {
                script {
                    sh """
                    mkdir -p ${BACKUP_DIR}
                    cp kubernetes/dev-deployment.yaml ${BACKUP_DIR}/dev-deployment-backup.yaml || true
                    kubectl apply -f kubernetes/dev-deployment.yaml
                    """
                    slackSend channel: env.SLACK_CHANNEL, message: "Model deployed to Dev successfully."
                }
            }
            post {
                failure {
                    script {
                        sh """
                        if [ -f ${BACKUP_DIR}/dev-deployment-backup.yaml ]; then
                            kubectl apply -f ${BACKUP_DIR}/dev-deployment-backup.yaml
                            echo "Rolled back Dev deployment to previous state."
                        else
                            echo "No backup found for Dev rollback."
                        fi
                        """
                        slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "Dev deployment failed and rolled back: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                    }
                }
            }
        }
        stage('Deploy to UAT') {
            when {
                expression { return currentBuild.currentResult == 'SUCCESS' }
            }
            steps {
                input message: 'Approve deployment to UAT?', ok: 'Deploy'
                script {
                    sh """
                    mkdir -p ${BACKUP_DIR}
                    cp kubernetes/uat-deployment.yaml ${BACKUP_DIR}/uat-deployment-backup.yaml || true
                    kubectl apply -f kubernetes/uat-deployment.yaml
                    """
                    slackSend channel: env.SLACK_CHANNEL, message: "Model deployed to UAT successfully."
                }
            }
            post {
                failure {
                    script {
                        sh """
                        if [ -f ${BACKUP_DIR}/uat-deployment-backup.yaml ]; then
                            kubectl apply -f ${BACKUP_DIR}/uat-deployment-backup.yaml
                            echo "Rolled back UAT deployment to previous state."
                        else
                            echo "No backup found for UAT rollback."
                        fi
                        """
                        slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "UAT deployment failed and rolled back: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                    }
                }
            }
        }
        stage('Deploy to Prod') {
            when {
                expression { return currentBuild.currentResult == 'SUCCESS' }
            }
            steps {
                input message: 'Approve deployment to Prod?', ok: 'Deploy'
                script {
                    sh """
                    mkdir -p ${BACKUP_DIR}
                    cp kubernetes/prod-deployment.yaml ${BACKUP_DIR}/prod-deployment-backup.yaml || true
                    kubectl apply -f kubernetes/prod-deployment.yaml
                    """
                    slackSend channel: env.SLACK_CHANNEL, message: "Model deployed to Prod successfully."
                }
            }
            post {
                failure {
                    script {
                        sh """
                        if [ -f ${BACKUP_DIR}/prod-deployment-backup.yaml ]; then
                            kubectl apply -f ${BACKUP_DIR}/prod-deployment-backup.yaml
                            echo "Rolled back Prod deployment to previous state."
                        else
                            echo "No backup found for Prod rollback."
                        fi
                        """
                        slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "Prod deployment failed and rolled back: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                    }
                }
            }
        }
    }
    post {
        failure {
            slackSend channel: env.SLACK_CHANNEL, color: 'danger', message: "Pipeline failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
        }
    }
}