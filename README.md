# MLOps Pipeline for Adult Classifier

This project implements an end-to-end MLOps pipeline for training, validating, and deploying a machine learning model (XGBoost classifier) on the UCI Adult dataset to predict income levels (>50K or <=50K). The pipeline leverages modern MLOps tools for model management, CI/CD automation, containerization, and secure access, with services accessible via a DuckDNS domain (`https://<your-domain-name>.duckdns.org`).

## Features
- **Model Training**: Trains an XGBoost classifier with hyperparameter tuning using Scikit-learn.
- **Model Management**: Tracks experiments, logs metrics, and manages model versions with MLflow.
- **Validation**: Validates models against a baseline (DummyClassifier) and previous production versions.
- **CI/CD**: Automates training and deployment with GitHub Actions and Jenkins.
- **Containerization**: Runs services (MLflow, Nginx, Redis, Jenkins) in Docker containers.
- **Secure Access**: Proxies MLflow (`/mlflow`) and Jenkins (`/jenkins`) via Nginx with SSL.
- **Caching**: Stores model metrics and metadata in Redis with password protection.
- **Notifications**: Integrates Slack notifications for pipeline status (configured in Jenkins).
- **Scalability**: Supports deployment to Kubernetes for production.

## Tools and Technologies
- **MLflow**: Manages the ML lifecycle (tracking, model registry, deployment).
- **Scikit-learn**: Provides machine learning algorithms and preprocessing (XGBoost, DummyClassifier).
- **CI/CD**: Automates pipelines with GitHub Actions (training) and Jenkins (orchestration).
- **GitHub Actions**: Runs CI/CD workflows for model training and validation.
- **Docker**: Containerizes services for portability and consistency.
- **Nginx**: Acts as a reverse proxy for MLflow and Jenkins with SSL termination.
- **PostgreSQL**: (Planned) Replaces SQLite for scalable MLflow backend storage.
- **Redis**: Caches model metrics and metadata with password protection.
- **Jenkins**: Orchestrates CI/CD pipelines and sends Slack notifications.
- **Slack Notification**: Alerts team on pipeline success/failure via Jenkins.

## Architecture
The pipeline consists of:
- **MLflow Server**: Tracks experiments and manages models (`https://<your-domain-name>.duckdns.org/mlflow`).
- **Nginx**: Proxies requests to MLflow (`/mlflow`) and Jenkins (`/jenkins`) with SSL.
- **Redis**: Caches metrics and metadata, secured with a password.
- **Jenkins**: Runs CI/CD pipelines and sends Slack notifications.
- **Training Script**: `train_model.py` trains, validates, and registers models.
- **Rollback Script**: `rollback_model.py` reverts to previous production models.
- **Docker Compose**: Orchestrates local services (MLflow, Nginx, Redis, Jenkins).
- **GitHub Actions**: Automates training and validation in CI/CD.
- **Kubernetes**: (Optional) Deploys the model for production.

## Prerequisites
- **Docker** and **Docker Compose**: For containerized services.
- **Python 3.9**: For running scripts locally.
- **Git**: For cloning the repository.
- **DuckDNS Account**: For domain setup (`<your-domain-name>.duckdns.org`).
- **SSL Certificates**: Let’s Encrypt certificates (`fullchain.pem`, `privkey.pem`).
- **Minikube/Kubernetes**: (Optional) For production deployment.
- **Slack Webhook**: For Jenkins notifications.
- **Windows/WSL2**: Supported for local development.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/adult-classifier-mlops.git
cd adult-classifier-mlops
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
REDIS_PASSWORD=your_secure_password
```
*Note*: Add `.env` to `.gitignore` to avoid committing sensitive data.

### 3. Set Up SSL Certificates
Place Let’s Encrypt certificates in `nginx/certs/`:
```
nginx/certs/fullchain.pem
nginx/certs/privkey.pem
```
Ensure they are valid for `<your-domain-name>.duckdns.org`.

### 4. Install Dependencies
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install required packages:
```bash
pip install mlflow==2.21.3 scikit-learn>=1.0.0 xgboost>=1.7.0 pandas>=1.5.0 numpy>=1.23.0 redis>=4.5.0
```

### 5. Start Docker Services
Update `docker-compose.yml` with your Redis password and run:
```bash
docker-compose up -d
```
This starts:
- MLflow (`http://mlflow:5000`)
- Nginx (`https://<your-domain-name>.duckdns.org`)
- Redis (`redis:6379`)
- Jenkins (`http://jenkins:8080/jenkins`)

### 6. Configure Jenkins
1. Access `http://localhost:8080/jenkins` to complete Jenkins setup:
   - Unlock Jenkins using the initial admin password (found in `jenkins/secrets/initialAdminPassword`).
   - Install recommended plugins.
2. Set up a pipeline using the provided `Jenkinsfile` or configure manually:
   ```groovy
   pipeline {
       agent any
       environment {
           MLFLOW_TRACKING_URI = 'https://<your-domain-name>.duckdns.org/mlflow'
           REDIS_HOST = 'redis'
           REDIS_PASSWORD = credentials('redis-password')
       }
       stages {
           stage('Train Model') {
               steps {
                   sh 'python scripts/train_model.py'
               }
           }
           stage('Verify Model Metadata') {
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
       }
       post {
           always {
               slackSend(channel: '#mlops', message: "Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER} completed: ${currentBuild.currentResult}")
           }
       }
   }
   ```
3. Add `redis-password` to Jenkins credentials with the value from `.env`.
4. Configure Slack notifications:
   - Install the Slack Notification plugin.
   - Add your Slack webhook URL in Jenkins settings (Manage Jenkins → Configure System → Slack).

### 7. Configure GitHub Actions
Update `.github/workflows/mlflow-cicd.yml` with your repository settings:
```yaml
name: MLflow CI/CD Pipeline
on:
  push:
    branches: [ main ]
jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7.0
        ports: [ 6379:6379 ]
        env:
          REDIS_PASSWORD: ${{ secrets.REDIS_PASSWORD }}
          REDIS_ARGS: --requirepass ${{ secrets.REDIS_PASSWORD }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.9'
      - run: |
          pip install -r requirements.txt
          pip install mlflow==2.21.3 scikit-learn>=1.0.0 xgboost>=1.7.0 pandas>=1.5.0 numpy>=1.23.0 redis>=4.5.0
      - run: python scripts/train_model.py
        env:
          REDIS_HOST: localhost
          REDIS_PASSWORD: ${{ secrets.REDIS_PASSWORD }}
```
Add `REDIS_PASSWORD` to GitHub Secrets (Settings → Secrets and variables → Actions).

### 8. (Optional) Deploy to Kubernetes
1. Create a Kubernetes Secret:
   ```bash
   kubectl create secret generic redis-secret --from-literal=REDIS_PASSWORD=your_secure_password
   ```
2. Apply the deployment:
   ```bash
   minikube start
   kubectl apply -f kubernetes/dev-deployment.yaml
   minikube service adult-classifier-service
   ```

### 9. (Planned) Configure PostgreSQL
Replace SQLite with PostgreSQL for MLflow:
```yaml
services:
  mlflow:
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://user:password@postgres:5432/mlflow
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlflow
    volumes:
      - D:/ci-cd-volumes/docker-data/postgres:/var/lib/postgresql/data
    networks:
      - mlflow-network
```

## Usage
1. **Train and Deploy Model**:
   ```bash
   export REDIS_HOST=redis
   export REDIS_PASSWORD=your_secure_password
   python scripts/train_model.py
   ```
   - Trains an XGBoost model, validates against a baseline, and promotes to `Production` if it outperforms the previous version.
   - Logs metrics and metadata to MLflow and Redis.

2. **Rollback Model**:
   ```bash
   python scripts/rollback_model.py
   ```
   - Reverts to the last archived `Production` model.

3. **Access Services**:
   - MLflow: `https://<your-domain-name>.duckdns.org/mlflow`
   - Jenkins: `https://<your-domain-name>.duckdns.org/jenkins`
   - Check pipeline status in Slack (`#mlops` channel).

4. **Monitor Redis**:
   ```bash
   docker exec <redis-container-name> redis-cli -a your_secure_password keys "model:adult-classifier:*"
   ```

## Directory Structure
```
.
├── .env                    # Environment variables (REDIS_PASSWORD)
├── docker-compose.yml      # Docker Compose configuration
├── nginx/
│   ├── nginx.conf          # Nginx reverse proxy config
│   ├── certs/              # SSL certificates
├── mlflow-data/            # MLflow artifacts and SQLite DB
├── docker/
│   ├── jenkins/
│   │   ├── Dockerfile.jenkins  # Custom Jenkins image
├── scripts/
│   ├── train_model.py      # Model training script
│   ├── rollback_model.py   # Model rollback script
├── feature_store/
│   ├── features.csv        # Transformed features
│   ├── preprocessor.joblib # Preprocessor artifact
├── .github/
│   ├── workflows/
│   │   ├── mlflow-cicd.yml # GitHub Actions workflow
├── kubernetes/
│   ├── dev-deployment.yaml # Kubernetes deployment
│   ├── redis-secret.yaml   # Kubernetes Secret
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

## Troubleshooting
- **Nginx Errors**:
  ```bash
  docker-compose logs nginx
  ```
  Check `nginx.conf` syntax and certificate paths.

- **Jenkins Not Accessible**:
  Verify `--prefix=/jenkins` in `docker-compose.yml`:
  ```bash
  curl http://localhost:8080/jenkins
  ```

- **Redis Connection Issues**:
  ```bash
  docker exec <redis-container-name> redis-cli -a your_secure_password ping
  ```
  Ensure `REDIS_PASSWORD` matches `.env`.

- **MLflow UI Not Loading**:
  Check MLflow service:
  ```bash
  docker-compose logs mlflow
  ```

## Security Best Practices
- **Protect Secrets**: Store `REDIS_PASSWORD` in `.env`, GitHub Secrets, or Kubernetes Secrets.
- **Restrict Ports**:
  ```yaml
  redis:
    ports:
      - 127.0.0.1:6379:6379
  jenkins:
    ports:
      - 127.0.0.1:8080:8080
  ```
- **Rotate Certificates**: Renew Let’s Encrypt certificates regularly.
- **Backup Data**:
  ```bash
  cp -r jenkins jenkins-backup
  docker exec <redis-container-name> redis-cli -a your_secure_password save
  ```

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License
MIT License (see LICENSE file).

## Contact
For issues or questions, open a GitHub issue or contact the team via Slack (`#mlops`).