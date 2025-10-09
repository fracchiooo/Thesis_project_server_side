# Opzione 1: Maven 3.9 con Java 21 (Eclipse Temurin)
FROM maven:3.9-eclipse-temurin-21 as build

# Set the working directory in the container
WORKDIR /app

# Copy the pom.xml and download dependencies
COPY pom.xml .

# Copy the rest of the application source code and build the application
COPY src ./src
RUN mvn clean install -DskipTests

# Start a new image for the runtime with only the JRE
FROM eclipse-temurin:21-jre

# Set the working directory in the runtime image
WORKDIR /app

# Copy the built application from the Maven build stage
COPY --from=build /app/target/*.jar app.jar

# Expose the application port
EXPOSE 3000

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]