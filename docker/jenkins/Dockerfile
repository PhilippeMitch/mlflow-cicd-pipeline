FROM jenkins/jenkins:2.426.3-lts-jdk17

# Update packages to address vulnerabilities
USER root
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Add a healthcheck to monitor the container's health
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8080/login || exit 1

USER root
# Fix permissions on volume at startup
RUN echo '#!/bin/bash\n\
  chown -R 1000:1000 /var/jenkins_home\n\
  chmod -R u+rw /var/jenkins_home\n\
  exec gosu jenkins "$@"' > /usr/local/bin/jenkins-entrypoint.sh && \
  chmod +x /usr/local/bin/jenkins-entrypoint.sh

USER jenkins
ENTRYPOINT ["/usr/local/bin/jenkins-entrypoint.sh"]
CMD ["/sbin/tini", "--", "/usr/local/bin/jenkins.sh"]