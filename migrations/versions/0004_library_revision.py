"""Track cache invalidation in the document transaction."""

from alembic import op
import sqlalchemy as sa

revision = "0004_library_revision"
down_revision = "0003_job_filename"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("library_revision", sa.Uuid()))


def downgrade() -> None:
    op.drop_column("users", "library_revision")
