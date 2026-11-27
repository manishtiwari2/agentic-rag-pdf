"""Store pending uploads and link completed jobs to document versions."""

from alembic import op
import sqlalchemy as sa

revision = "0002_ingestion_uploads"
down_revision = "0001_persistent_library"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("result_version_id", sa.Uuid()))
    op.create_foreign_key(
        "fk_jobs_user_id_document_versions", "jobs", "document_versions",
        ["user_id", "result_version_id"], ["user_id", "id"], ondelete="CASCADE",
    )
    op.create_table(
        "job_uploads",
        sa.Column("user_id", sa.Uuid(), primary_key=True),
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("file_sha256", sa.String(64), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("version_label", sa.Text()),
        sa.ForeignKeyConstraint(["user_id", "id"], ["jobs.user_id", "jobs.id"], ondelete="CASCADE"),
        sa.CheckConstraint("size_bytes = octet_length(data) AND size_bytes > 0", name=op.f("ck_job_uploads_size")),
    )
    op.execute("ALTER TABLE job_uploads ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE job_uploads FORCE ROW LEVEL SECURITY")
    op.execute("""
CREATE POLICY owner_scope ON job_uploads
    USING (user_id = nullif(current_setting('rag.user_id', true), '')::uuid)
""")


def downgrade() -> None:
    op.drop_table("job_uploads")
    op.drop_constraint("fk_jobs_user_id_document_versions", "jobs", type_="foreignkey")
    op.drop_column("jobs", "result_version_id")
