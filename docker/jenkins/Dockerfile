FROM jenkins/jenkins:2.504.1-jdk21

USER root

# Install required tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# Optional: Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/login || exit 1

# Restore Jenkins user
USER jenkins