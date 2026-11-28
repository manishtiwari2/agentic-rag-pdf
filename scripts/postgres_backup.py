"""Back up or restore the PostgreSQL service in the Compose deployment."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DATABASE_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def backup(postgres: list[str], archive: Path) -> None:
    if archive.exists():
        raise FileExistsError(f"Backup already exists: {archive}")
    stream = NamedTemporaryFile(dir=archive.parent, prefix=".pg-backup-", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            subprocess.run(
                [*postgres, "sh", "-ec",
                 'exec pg_dump --no-password --format=custom --no-privileges '
                 '--username "$POSTGRES_USER" --dbname "$POSTGRES_DB"'],
                stdout=stream, check=True,
            )
        os.link(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def restore(postgres: list[str], archive: Path, database: str) -> None:
    if not DATABASE_NAME.fullmatch(database):
        raise ValueError("Use a lowercase PostgreSQL database name of up to 63 letters, digits, or underscores.")
    with archive.open("rb") as stream:
        subprocess.run([*postgres, "pg_restore", "--list"], stdin=stream, stdout=subprocess.DEVNULL, check=True)
        stream.seek(0)
        subprocess.run(
            [*postgres, "createdb", "--no-password", "--username=rag_admin", "--maintenance-db=postgres",
             "--template=template0", "--", database],
            check=True,
        )
        subprocess.run(
            [*postgres, "pg_restore", "--no-password", "--username=rag_admin", "--dbname", database,
             "--single-transaction", "--no-owner", "--no-privileges"],
            stdin=stream, check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--project-name")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("backup").add_argument("archive", type=Path)
    restore_command = commands.add_parser("restore")
    restore_command.add_argument("archive", type=Path)
    restore_command.add_argument("--database", required=True)
    args = parser.parse_args()

    postgres = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml")]
    if args.env_file:
        postgres.extend(["--env-file", str(args.env_file)])
    if args.project_name:
        postgres.extend(["--project-name", args.project_name])
    postgres.extend(["exec", "-T", "postgres"])
    try:
        if args.action == "backup":
            backup(postgres, args.archive)
            print(f"Backup written to {args.archive}")
        else:
            restore(postgres, args.archive, args.database)
            print(f"Backup restored to {args.database}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"PostgreSQL {args.action} failed. {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
