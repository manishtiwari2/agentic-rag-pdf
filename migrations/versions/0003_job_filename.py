"""Keep the source filename on jobs after the staged bytes are removed."""

from alembic import op
import sqlalchemy as sa

revision = "0003_job_filename"
down_revision = "0002_ingestion_uploads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("filename", sa.Text()))


def downgrade() -> None:
    op.drop_column("jobs", "filename")
