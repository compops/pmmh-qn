# Use an official Python runtime as a parent image
FROM python:3.6.3

# Set the working directory to /app
WORKDIR /app/python

# Copy the current directory contents into the container at /app
ADD ./python /app/python
ADD ./requirements1.txt /app/python
ADD ./requirements2.txt /app/python

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/python/requirements1.txt
RUN pip install -r /app/python/requirements2.txt

# Run script on launch
CMD bash /app/python/run_script.sh

# Build
# docker build -t pmmh-qn .
# docker images
# docker tag <<TAG>> compops/pmmh-qn:draft1
# docker tag <<TAG>> compops/pmmh-qn:latest
# docker login --username=yourhubusername --email=youremail@provider.com
# docker push compops/pmmh-qn
