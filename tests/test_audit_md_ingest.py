def test_markdown_heading_split_with_frontmatter_and_timestamps():
    from borge.audit.ingest_md import load_markdown_dump
    from borge.audit.ingest import MemoryRecord
    recs = load_markdown_dump("tests/fixtures/audit_memory.md", split="heading")
    assert recs and all(isinstance(r, MemoryRecord) for r in recs)
    joined = " ".join(r.text for r in recs).lower()
    assert "vegetarian" in joined and "steak" in joined
    # frontmatter `modified` recovered as timestamp for sections lacking an inline date
    assert any(r.timestamp.startswith("2026-05-20") for r in recs)
    # ids unique + stable (slug from heading)
    assert len({r.id for r in recs}) == len(recs)
    assert all(r.id for r in recs)


def test_heading_split_assigns_goal_role_to_deadline_sections():
    """Goal-keyword headings get role='goal'; others keep the frontmatter role."""
    from borge.audit.ingest_md import load_markdown_dump
    recs = load_markdown_dump("tests/fixtures/audit_memory.md", split="heading")
    by_slug = {r.id.split("#", 1)[1]: r for r in recs}
    # "Deadline A" and "Deadline B" contain the word "deadline" → goal role
    assert by_slug["deadline-a"].role == "goal"
    assert by_slug["deadline-b"].role == "goal"
    # Personal facts inherit the frontmatter default ("user")
    assert by_slug["allergy"].role == "user"
    assert by_slug["diet"].role == "user"
    # goal and user roles coexist → goal_relevance and self_relevance are independent
    roles = {r.role for r in recs}
    assert "goal" in roles and "user" in roles
