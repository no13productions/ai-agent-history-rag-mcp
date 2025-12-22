"""Technology pattern definitions for query analysis.

This module contains comprehensive technology keyword patterns used to detect
programming languages, frameworks, tools, and platforms mentioned in search queries.

Each technology is mapped to a list of keywords that indicate its presence.
These patterns are used by the QueryAnalyzer for technology detection.
"""

# Technology patterns for detection
# Maps technology names to keywords that indicate that technology
TECHNOLOGY_PATTERNS: dict[str, list[str]] = {
    # Programming Languages
    "python": [
        "python",
        "py",
        "django",
        "flask",
        "fastapi",
        "pandas",
        "numpy",
        "pytorch",
        "tensorflow",
        "pytest",
        "pip",
        "venv",
        "conda",
        "pydantic",
        "asyncio",
    ],
    "javascript": [
        "javascript",
        "js",
        "node",
        "nodejs",
        "npm",
        "yarn",
        "react",
        "vue",
        "angular",
        "express",
        "next",
        "nextjs",
        "webpack",
        "babel",
        "eslint",
    ],
    "typescript": [
        "typescript",
        "ts",
        "tsx",
        "tsc",
        "tsconfig",
    ],
    "java": [
        "java",
        "spring",
        "springboot",
        "maven",
        "gradle",
        "junit",
        "hibernate",
        "jvm",
    ],
    "csharp": [
        "c#",
        "csharp",
        "dotnet",
        ".net",
        "asp.net",
        "entity framework",
        "nuget",
    ],
    "go": [
        "golang",
        "go mod",
        "goroutine",
        "gin",
        "echo",
    ],
    "rust": [
        "rust",
        "cargo",
        "rustc",
        "tokio",
        "serde",
        "actix",
    ],
    "ruby": [
        "ruby",
        "rails",
        "gem",
        "bundler",
        "rake",
    ],
    "php": [
        "php",
        "laravel",
        "symfony",
        "composer",
        "wordpress",
    ],
    "swift": [
        "swift",
        "swiftui",
        "xcode",
        "cocoapods",
        "ios",
    ],
    "kotlin": [
        "kotlin",
        "android",
        "gradle",
        "jetpack",
    ],
    "dart": [
        "dart",
        "flutter",
        "pubspec",
    ],
    # Infrastructure & DevOps
    "docker": [
        "docker",
        "dockerfile",
        "container",
        "docker-compose",
        "containerize",
    ],
    "kubernetes": [
        "kubernetes",
        "k8s",
        "kubectl",
        "helm",
        "pod",
        "deployment",
        "service",
    ],
    "terraform": [
        "terraform",
        "tf",
        "hcl",
        "infrastructure as code",
    ],
    "ansible": [
        "ansible",
        "playbook",
        "inventory",
    ],
    # Cloud Providers
    "aws": [
        "aws",
        "amazon",
        "ec2",
        "s3",
        "lambda",
        "cloudformation",
        "dynamodb",
        "rds",
        "ecs",
        "eks",
    ],
    "azure": [
        "azure",
        "microsoft cloud",
        "app service",
        "azure functions",
    ],
    "gcp": [
        "gcp",
        "google cloud",
        "cloud run",
        "bigquery",
        "gke",
    ],
    # Databases
    "postgresql": [
        "postgresql",
        "postgres",
        "pg",
        "psql",
    ],
    "mysql": [
        "mysql",
        "mariadb",
    ],
    "mongodb": [
        "mongodb",
        "mongo",
        "mongoose",
    ],
    "redis": [
        "redis",
        "cache",
    ],
    "elasticsearch": [
        "elasticsearch",
        "elastic",
        "kibana",
    ],
    # Version Control & CI/CD
    "git": [
        "git",
        "github",
        "gitlab",
        "bitbucket",
        "branch",
        "commit",
        "merge",
        "rebase",
        "pull request",
        "pr",
    ],
    "github_actions": [
        "github actions",
        "gh actions",
        "workflow",
        "ci/cd",
    ],
    # Testing
    "pytest": [
        "pytest",
        "py.test",
    ],
    "jest": [
        "jest",
        "react testing library",
    ],
    # AI/ML
    "ai_ml": [
        "huggingface",
        "transformers",
        "langchain",
        "openai",
        "anthropic",
        "claude",
        "gpt",
        "llm",
        "embedding",
        "vector search",
    ],
    # Vector Databases
    "vector_db": [
        "lancedb",
        "pinecone",
        "weaviate",
        "qdrant",
        "chroma",
        "faiss",
        "milvus",
    ],
    # Modern Python/JS Tools
    "modern_tools": [
        "fastmcp",
        "pydantic",
        "fastembed",
        "bun",
        "deno",
        "uv",
        "ruff",
    ],
}
