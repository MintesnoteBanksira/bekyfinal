# Use an official Python runtime as a parent image
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the Django project directory into the container at /app
COPY bekyDjango/ /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Upgrade pip and setuptools, and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt --verbose

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
