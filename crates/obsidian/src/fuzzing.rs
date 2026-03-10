pub fn split_frontmatter(content: &str) -> (Option<&str>, &str) {
    crate::frontmatter::split_frontmatter(content)
}
