@echo off
REM Start PostgreSQL in Docker with matching .env variables
REM Run this script from the Sign-Language-main folder

echo Starting PostgreSQL Docker container...

docker run -d ^
    --name signlang-postgres ^
    -e POSTGRES_USER=signlang_user ^
    -e POSTGRES_PASSWORD=signlang_secret_2024 ^
    -e POSTGRES_DB=signlanguage ^
    -p 5432:5432 ^
    -v signlang-pgdata:/var/lib/postgresql/data ^
    postgres:16-alpine

echo.
echo PostgreSQL container started!
echo.
echo Connection details (matching backend/.env):
echo   Host: localhost
echo   Port: 5432
echo   User: signlang_user
echo   Password: signlang_secret_2024
echo   Database: signlanguage
echo.
echo To stop:  docker stop signlang-postgres
echo To start: docker start signlang-postgres
echo To remove: docker rm signlang-postgres
echo.
pause
