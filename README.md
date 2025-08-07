# Multi-Tenant SaaS Voice Agent Backend

This is a boilerplate for a multi-tenant SaaS application backend built with Python, FastAPI, and PostgreSQL. It includes a basic structure for handling multiple tenants, user authentication, and a foundation for building a voice agent service.

## Features

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **PostgreSQL**: A powerful, open-source object-relational database system.
- **SQLAlchemy**: The Python SQL Toolkit and Object Relational Mapper.
- **Alembic**: A lightweight database migration tool for SQLAlchemy.
- **Pydantic**: Data validation and settings management using Python type annotations.
- **Multi-Tenancy**: Basic support for multi-tenancy using a schema-based approach.
- **Docker**: Containerization support for easy development and deployment.

## Project Structure

```
.
├── alembic/              # Alembic migration scripts
├── app/                  # Main application directory
│   ├── api/              # API versioning and endpoint routers
│   ├── core/             # Application core configuration
│   ├── db/               # Database session management and base models
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic schemas for data validation
│   ├── services/         # Business logic and services
│   └── routers/          # Application-level routers (e.g., health checks)
├── .env                  # Environment variables (needs to be created)
├── .gitignore            # Git ignore file
├── alembic.ini           # Alembic configuration
├── Dockerfile            # Docker configuration
├── main.py               # Main application entry point
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized setup)
- PostgreSQL

### Local Development Setup

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file:**
    Copy the example environment variables into a new `.env` file.
    ```sh
    cp .env.example .env
    ```
    Update the `DATABASE_URL` if your PostgreSQL configuration is different.

4.  **Set up the database and run migrations:**
    - Make sure your PostgreSQL server is running.
    - Create the database specified in your `.env` file.
    - Run the following Alembic command to apply migrations:
      ```sh
      alembic upgrade head
      ```

5.  **Run the application:**
    ```sh
    uvicorn app.main:app --reload
    ```
    The application will be available at `http://localhost:8000`.

### Docker Setup

1.  **Build the Docker image:**
    ```sh
    docker build -t voice-agent-backend .
    ```

2.  **Run the container:**
    You'll need a running PostgreSQL instance accessible from the container. You can use Docker Compose to manage both the application and database services.

    ```sh
    docker run -p 8000:8000 --env-file .env voice-agent-backend
    ```

## API Endpoints

-   **Health Check**: `GET /health`
    -   Returns a status message to indicate the service is running.
-   **API v1 Docs**: `http://localhost:8000/api/v1/docs`
    -   Swagger UI for the v1 API.

## Code Quality

This project uses `black` for code formatting and `ruff` for linting.

-   **Format code:**
    ```sh
    black .
    ```
-   **Check for linting errors:**
    ```sh
    ruff check .
    ``` 